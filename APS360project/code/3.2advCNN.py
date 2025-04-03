
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
# Use new autocast API
from torch.amp import autocast
from torch.cuda.amp import GradScaler
# Ensure all PIL-related modules are imported at the global level
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import multiprocessing
import copy
import matplotlib.font_manager as fm
from matplotlib import rcParams
import warnings

# Set up Chinese font support
def setup_chinese_font():
    try:
        # Try to set font that supports Chinese
        if os.path.exists('SimHei.ttf'):
            plt.rcParams['font.sans-serif'] = ['SimHei'] 
        else:
            # Fall back to Arial
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display
        # Ignore font warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    except Exception as e:
        print(f"Error setting Chinese font: {e}")
        # Fall back to English
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable cuDNN benchmarking for performance

# Define data paths
data_root = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Preprocess"
train_dir = os.path.join(data_root, "train")
val_dir = os.path.join(data_root, "val")
test_dir = os.path.join(data_root, "test")

# Get emotion categories
def get_emotion_categories():
    return sorted(os.listdir(train_dir))

# Calculate class weights for handling imbalance
def calculate_class_weights(data_dir, emotion_categories, accuracy_dict=None):
    class_counts = {}
    num_classes = len(emotion_categories)
    
    # Calculate sample counts for each class
    for emotion in emotion_categories:
        class_path = os.path.join(data_dir, emotion)
        class_counts[emotion] = len(os.listdir(class_path))
    
    total_samples = sum(class_counts.values())
    # If accuracy dictionary is provided, combine accuracy and sample count for weights
    if accuracy_dict:
        # Combine inverse accuracy and inverse sample count
        class_weights = {emotion: (total_samples / (num_classes * count)) * 
                                  (1.0 / (accuracy_dict.get(emotion, 0.5) + 0.1)) 
                        for emotion, count in class_counts.items()}
    else:
        # Based only on sample count
        class_weights = {emotion: total_samples / (num_classes * count) 
                        for emotion, count in class_counts.items()}
    
    # Normalize weights
    weight_sum = sum(class_weights.values())
    norm_class_weights = {k: v / weight_sum * num_classes for k, v in class_weights.items()}
    
    # Convert to tensor format for loss function
    weights_tensor = torch.FloatTensor([norm_class_weights[emotion] for emotion in emotion_categories])
    
    return norm_class_weights, weights_tensor

# Define transforms targeting low-accuracy categories
def get_transforms(mode='train', target_classes=None):
    # Base transforms for all classes
    base_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),  # Resize
            transforms.RandomCrop(224),     # Random crop
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    # Enhanced transforms for low-accuracy categories
    enhanced_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.7),  # Increase flip probability
            transforms.RandomRotation(30),  # Increase rotation angle
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),  # Enhanced affine transform
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.2),  # Enhanced color changes
            transforms.RandomGrayscale(p=0.05),  # Occasional grayscale
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2),  # Increase random erasing probability
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    
    if mode == 'val' or mode == 'test':
        return base_transforms['val']
    else:
        return base_transforms['train'], enhanced_transforms['train']

# Multiprocessing-compatible dataset
class MultiprocessEmotionDataset(Dataset):
    def __init__(self, data_dir, emotion_categories, transform=None, enhanced_transform=None, 
                 target_classes=None, adaptive_augment=True, mixer_prob=0.2):
        self.data_dir = data_dir
        self.emotion_categories = emotion_categories
        self.transform = transform  # Base transforms
        self.enhanced_transform = enhanced_transform  # Enhanced transforms
        self.target_classes = target_classes or []  # Classes needing special treatment
        self.adaptive_augment = adaptive_augment
        self.mixer_prob = mixer_prob  # Mix-up probability
        
        # Read all image file paths and corresponding labels
        self.samples = []
        self.targets = []
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(emotion_categories)}
        
        # Organize samples by class
        self.samples_by_class = {emotion: [] for emotion in emotion_categories}
        
        for emotion in emotion_categories:
            emotion_dir = os.path.join(data_dir, emotion)
            emotion_idx = self.emotion_to_idx[emotion]
            image_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            self.samples_by_class[emotion] = image_files
            self.samples.extend(image_files)
            self.targets.extend([emotion_idx] * len(image_files))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        target = self.targets[idx]
        emotion = self.emotion_categories[target]
        
        try:
            # Read image
            img = Image.open(img_path).convert('RGB')
            
            # Apply special enhancement for low-accuracy categories
            if self.adaptive_augment and emotion in self.target_classes and self.enhanced_transform is not None:
                # Randomly choose to use enhanced transforms
                if random.random() < 0.8:  # 80% probability to use enhanced transforms
                    # Apply manual enhancement
                    if random.random() < 0.5:
                        # Moderate blur to simulate unclear images
                        radius = random.uniform(0.1, 0.7)
                        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
                    
                    # Adjust contrast
                    if random.random() < 0.7:
                        enhancer = ImageEnhance.Contrast(img)
                        factor = random.uniform(0.7, 1.3)
                        img = enhancer.enhance(factor)
                    
                    # Adjust brightness
                    if random.random() < 0.7:
                        enhancer = ImageEnhance.Brightness(img)
                        factor = random.uniform(0.7, 1.3)
                        img = enhancer.enhance(factor)
                    
                    # Rotation
                    if random.random() < 0.6:
                        angle = random.uniform(-30, 30)
                        img = img.rotate(angle, resample=Image.BILINEAR, expand=False)
                    
                    # Apply enhanced transforms
                    img = self.enhanced_transform(img)
                else:
                    # Apply base transforms
                    img = self.transform(img)
                
                # Apply mix-up augmentation (MixUp)
                if random.random() < self.mixer_prob:
                    try:
                        # Randomly select another image from the same class
                        same_class_samples = self.samples_by_class[emotion]
                        if len(same_class_samples) > 1:
                            other_img_path = random.choice([s for s in same_class_samples if s != img_path])
                            other_img = Image.open(other_img_path).convert('RGB')
                            
                            # Apply the same transforms to the second image
                            if random.random() < 0.8:
                                other_img = self.enhanced_transform(other_img)
                            else:
                                other_img = self.transform(other_img)
                            
                            # Mix the two images
                            alpha = random.uniform(0.3, 0.7)
                            img = alpha * img + (1 - alpha) * other_img
                    except Exception as e:
                        print(f"Error in mix-up: {e}")
                        # Ignore errors, continue with original image
            else:
                # Apply base transforms for other categories
                img = self.transform(img)
                
            return img, target
            
        except Exception as e:
            print(f"Error reading image {img_path}: {e}")
            # Return a placeholder image and label
            dummy_img = torch.zeros(3, 224, 224)
            return dummy_img, target

# Advanced balanced sampler for difficult classes
class AdvancedBalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset, class_accuracies=None, beta=0.9999):
        # Get indices for each class
        indices_per_class = {}
        for idx, target in enumerate(dataset.targets):
            emotion = dataset.emotion_categories[target]
            if emotion not in indices_per_class:
                indices_per_class[emotion] = []
            indices_per_class[emotion].append(idx)

        # If no accuracy provided, use uniform weights
        if class_accuracies is None:
            class_weights = {emotion: 1.0 for emotion in dataset.emotion_categories}
        else:
            # Use effective number sampling based on sample count and accuracy
            # beta closer to 1 means more balanced; beta closer to 0 means closer to original distribution
            n_samples = len(dataset)
            n_classes = len(dataset.emotion_categories)
            
            # Calculate sample count for each class
            class_counts = {emotion: len(indices) for emotion, indices in indices_per_class.items()}
            
            # Calculate target sampling count (based on effective number and accuracy)
            effective_num = {emotion: 1.0 - beta**count for emotion, count in class_counts.items()}
            weights = {emotion: (1.0 - beta) / (effective_num[emotion] + 1e-10) for emotion in class_counts}
            
            # Further adjust weights based on accuracy
            weights = {emotion: weight * (1.0/(class_accuracies.get(emotion, 0.5) + 0.1)) 
                      for emotion, weight in weights.items()}
                
            # Normalize
            total_weight = sum(weights.values())
            class_weights = {emotion: weight/total_weight * n_classes for emotion, weight in weights.items()}
            
            print("Sampler class weights:")
            for emotion, weight in class_weights.items():
                print(f"{emotion}: {weight:.4f}")
        
        # Calculate weight for each sample
        weights = []
        for idx, target in enumerate(dataset.targets):
            emotion = dataset.emotion_categories[target]
            weights.append(class_weights[emotion])
            
        # Initialize sampler
        super().__init__(weights=torch.DoubleTensor(weights), num_samples=len(dataset), replacement=True)

# Create data loaders
def create_dataloaders(batch_size=64, num_workers=4, emotion_categories=None, class_accuracies=None):
    # Determine target classes based on accuracy
    if class_accuracies:
        target_classes = [emotion for emotion, acc in class_accuracies.items() if acc < 0.6]
        print(f"Target optimization classes: {target_classes}")
    else:
        target_classes = ["Confusion", "Contempt", "Disgust"]  # Default target classes
    
    # Get base and enhanced transforms
    base_transforms, enhanced_transforms = get_transforms(mode='train', target_classes=target_classes)
    val_test_transform = get_transforms(mode='val')
    
    # Create datasets
    train_dataset = MultiprocessEmotionDataset(
        train_dir, emotion_categories, 
        transform=base_transforms, 
        enhanced_transform=enhanced_transforms,
        target_classes=target_classes, 
        adaptive_augment=True,
        mixer_prob=0.15  # Mix-up probability
    )
    
    val_dataset = MultiprocessEmotionDataset(
        val_dir, emotion_categories, 
        transform=val_test_transform, 
        enhanced_transform=None,
        target_classes=None, 
        adaptive_augment=False
    )
    
    test_dataset = MultiprocessEmotionDataset(
        test_dir, emotion_categories, 
        transform=val_test_transform, 
        enhanced_transform=None,
        target_classes=None, 
        adaptive_augment=False
    )
    
    # Calculate class distribution in training set
    train_class_counts = {}
    for target in train_dataset.targets:
        emotion = emotion_categories[target]
        train_class_counts[emotion] = train_class_counts.get(emotion, 0) + 1
    
    print("Training set class distribution:")
    for emotion, count in sorted(train_class_counts.items()):
        print(f"{emotion}: {count} images")
    
    # Create balanced sampler (if accuracy information is provided)
    if class_accuracies:
        # Use advanced balanced sampler
        sampler = AdvancedBalancedSampler(train_dataset, class_accuracies)
        shuffle = False  # Can't use shuffle with sampler
    else:
        sampler = None
        shuffle = True
    
    # Create data loaders - using multiprocessing
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=num_workers,  # Use 4 worker processes
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,  # Each worker prefetches 2 batches
        persistent_workers=True if num_workers > 0 else False,  # Keep worker processes alive
        drop_last=True  # Drop incomplete final batch, helps with batch normalization
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

# Attention modules
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Create spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        x_out = self.conv1(x_cat)
        attention = self.sigmoid(x_out)
        
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        out = avg_out + max_out
        attention = self.sigmoid(out)
        
        return x * attention

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# Enhanced emotion recognition model
class EnhancedEmotionModel(nn.Module):
    def __init__(self, num_classes, dropout_rates=[0.5, 0.4, 0.3], backbone='efficientnet_b3'):
        super(EnhancedEmotionModel, self).__init__()
        
        # Choose base model
        if backbone == 'efficientnet_b2':
            self.base_model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
            last_channel = self.base_model.classifier[1].in_features
        elif backbone == 'efficientnet_b3':
            self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
            last_channel = self.base_model.classifier[1].in_features
        elif backbone == 'resnet50':
            self.base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            last_channel = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove original classification head
        if hasattr(self.base_model, 'classifier'):
            self.base_model.classifier = nn.Identity()
        
        # Add attention module
        self.cbam = CBAM(channels=last_channel, reduction=16, kernel_size=7)
        
        # Global pooling after feature extraction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Complex classification head - deeper MLP, stronger regularization
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rates[0]),
            nn.Linear(last_channel, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[1]),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_rates[2]),
            nn.Linear(512, num_classes)
        )
        
        # Specialized head for difficult-to-classify categories
        self.specialized_classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),  # Use GELU activation function
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 3)  # Assuming 3 difficult-to-classify categories
        )
        
        # Class indicators for deciding which classification head to use
        self.difficult_classes = [2, 1, 3]  # Assuming these are Contempt, Confusion, Disgust indices
        
    def forward(self, x):
        # Feature extraction
        features = self.base_model.features(x) if hasattr(self.base_model, 'features') else self.base_model(x)
        
        # Apply attention mechanism
        features = self.cbam(features)
        
        # Global pooling
        features = self.avg_pool(features)
        features = torch.flatten(features, 1)
        
        # Main classifier output
        main_output = self.classifier(features)
        
        # For difficult-to-classify categories, additionally use specialized classifier
        specialized_output = self.specialized_classifier(features)
        
        # Return feature vector for visualization
        return main_output, features

# Focal Loss implementation - for hard-to-classify samples
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Label smoothing cross entropy
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        if self.weight is not None:
            true_dist = true_dist * self.weight.unsqueeze(0)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

# Mixed loss function - combining Focal Loss and label smoothing
class MixedLoss(nn.Module):
    def __init__(self, classes, alpha=None, gamma=2.0, smoothing=0.1, mix_ratio=0.5):
        super(MixedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.label_smoothing = LabelSmoothingLoss(classes=classes, smoothing=smoothing, weight=alpha)
        self.mix_ratio = mix_ratio
        
    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        smooth = self.label_smoothing(inputs, targets)
        return self.mix_ratio * focal + (1 - self.mix_ratio) * smooth

# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        # Fix here - use new autocast format
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, targets in tqdm(dataloader, desc="Validating"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate probabilities
                probs = F.softmax(outputs, dim=1)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_targets, all_probs

# Improved gradual unfreezing training function
def train_with_gradual_unfreezing(model, train_loader, val_loader, criterion, device, 
                                   num_epochs=30, stages=3, patience=5, eval_interval=1):
    # Initialize
    best_val_acc = 0.0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    patience_counter = 0
    scaler = GradScaler()  # Mixed precision training
    
    # Stage 1: Train only the classification head
    print("\n===== Stage 1: Training classification head =====")
    # Freeze feature extractor
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    # Set optimizer
    optimizer = optim.AdamW(
        list(model.classifier.parameters()) + 
        list(model.specialized_classifier.parameters()) + 
        list(model.cbam.parameters()), 
        lr=0.001, 
        weight_decay=0.01
    )
    
    # Learning rate scheduler - use OneCycleLR
    steps_per_epoch = len(train_loader)
    epochs_per_stage = num_epochs // stages
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.001, 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs_per_stage,
        pct_start=0.3
    )
    
    # Train stage 1
    for epoch in range(epochs_per_stage):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        
        # Periodic validation
        if (epoch + 1) % eval_interval == 0 or epoch == epochs_per_stage - 1:
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Stage 1 - Epoch {epoch+1}/{epochs_per_stage} - "
                  f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
                  f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"Saved new best model, validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: {patience} epochs without improvement")
                    model.load_state_dict(best_model_state)
                    break
    
    # Stage 2: Partially unfreeze feature extractor
    print("\n===== Stage 2: Partially unfreezing feature extractor =====")
    
    # Get layers to unfreeze based on model type
    if hasattr(model.base_model, 'features'):
        # For EfficientNet models
        layers_to_unfreeze = list(model.base_model.features)[-6:]
    else:
        # For ResNet models
        backbone_layers = list(model.base_model.children())
        layers_to_unfreeze = backbone_layers[-3:-1]  # Usually includes the last few blocks
    
    # Unfreeze selected layers
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Set optimizer with lower learning rates
    param_groups = [
        {'params': model.classifier.parameters(), 'lr': 0.0005},
        {'params': model.specialized_classifier.parameters(), 'lr': 0.0005},
        {'params': model.cbam.parameters(), 'lr': 0.0005}
    ]
    
    # Add parameter groups for unfrozen layers
    for i, layer in enumerate(layers_to_unfreeze):
        param_groups.append({
            'params': layer.parameters(), 
            'lr': 0.0001 if i < len(layers_to_unfreeze)//2 else 0.0002
        })
    
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    
    # Update learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.0005, 0.0005, 0.0005] + [0.0001 if i < len(layers_to_unfreeze)//2 else 0.0002 for i in range(len(layers_to_unfreeze))], 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs_per_stage,
        pct_start=0.3
    )
    
    # Reset early stopping counter
    patience_counter = 0
    
    # Train stage 2
    for epoch in range(epochs_per_stage):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        
        # Periodic validation
        if (epoch + 1) % eval_interval == 0 or epoch == epochs_per_stage - 1:
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Stage 2 - Epoch {epoch+1}/{epochs_per_stage} - "
                  f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
                  f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"Saved new best model, validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: {patience} epochs without improvement")
                    model.load_state_dict(best_model_state)
                    break
    
    # Stage 3: Completely unfreeze model and fine-tune
    print("\n===== Stage 3: Completely unfreezing model =====")
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Set optimizer with very small learning rates
    optimizer = optim.AdamW([
        {'params': model.classifier.parameters(), 'lr': 0.0001},
        {'params': model.specialized_classifier.parameters(), 'lr': 0.0001},
        {'params': model.cbam.parameters(), 'lr': 0.0001},
        {'params': model.base_model.parameters(), 'lr': 0.00002}
    ], weight_decay=0.01)
    
    # Update learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[0.0001, 0.0001, 0.0001, 0.00002], 
        steps_per_epoch=steps_per_epoch, 
        epochs=epochs_per_stage,
        pct_start=0.3
    )
    
    # Reset early stopping counter
    patience_counter = 0
    
    # Train stage 3
    for epoch in range(epochs_per_stage):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        
        # Periodic validation
        if (epoch + 1) % eval_interval == 0 or epoch == epochs_per_stage - 1:
            val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Stage 3 - Epoch {epoch+1}/{epochs_per_stage} - "
                  f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
                  f"Val loss: {val_loss:.4f}, Val accuracy: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
                print(f"Saved new best model, validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping: {patience} epochs without improvement")
                    model.load_state_dict(best_model_state)
                    break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    return model, history, best_val_acc

# Improved single epoch training function
def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use tqdm progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for inputs, targets in progress_bar:
        # Move data to GPU
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        
        # Mixed precision forward pass - fix here, use new autocast format
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
        
        # Mixed precision backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        
        # Gradient clipping to prevent explosion
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Update learning rate
        scheduler.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        current_loss = running_loss / total
        current_acc = correct / total
        progress_bar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
        
        # Free unnecessary memory
        torch.cuda.empty_cache()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Process batch results in parallel (using Pool for standalone tasks)
def parallel_process_results(func, data_list, num_processes=4):
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.map(func, data_list)
    return results

# Draw training history
def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 5))
    
    # Draw loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Training loss')
    plt.plot(history['val_loss'], 'r-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Validation epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Draw accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], 'b-', label='Training accuracy')
    plt.plot(history['val_acc'], 'r-', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Validation epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

# Draw confusion matrix
def plot_confusion_matrix(targets, predictions, class_names, save_dir):
    cm = confusion_matrix(targets, predictions)
    
    # Calculate normalized confusion matrix (row normalized)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Draw confusion matrix
    plt.figure(figsize=(14, 12))
    
    # Original confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Sample Count)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Normalized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Proportion)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

# Calculate metrics for each class
def calculate_per_class_metrics(targets, predictions, class_names):
    # Calculate accuracy for each class
    cm = confusion_matrix(targets, predictions)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    print("\nAccuracy for each emotion category:")
    for i, emotion in enumerate(class_names):
        print(f"{emotion}: {class_accuracy[i]:.4f}")
        
    # Calculate precision, recall, and F1 score
    report = classification_report(targets, predictions, target_names=class_names, digits=4)
    print("\nClassification report:")
    print(report)
    
    # Save classification report to file
    with open(os.path.join(data_root, 'improved_classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Draw accuracy for each class
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, class_accuracy)
    
    # Add value labels to each bar
    for bar, acc in zip(bars, class_accuracy):
        plt.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.01, 
                f'{acc:.4f}', 
                ha='center', va='bottom')
    
    plt.title('Accuracy for Each Emotion Category')
    plt.xlabel('Emotion Category')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(data_root, 'class_accuracy.png'))
    plt.close()
    
    # Draw F1 score for each class
    f1_scores = []
    for i, emotion in enumerate(class_names):
        # Calculate F1 score for this class
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, f1_scores)
    
    # Add value labels to each bar
    for bar, f1 in zip(bars, f1_scores):
        plt.text(bar.get_x() + bar.get_width()/2., 
                bar.get_height() + 0.01, 
                f'{f1:.4f}', 
                ha='center', va='bottom')
    
    plt.title('F1 Score for Each Emotion Category')
    plt.xlabel('Emotion Category')
    plt.ylabel('F1 Score')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(data_root, 'class_f1_scores.png'))
    plt.close()

# Draw ROC curves
def plot_roc_curves(targets, probabilities, class_names, save_dir):
    # Convert targets to one-hot encoding
    n_classes = len(class_names)
    y_test = np.zeros((len(targets), n_classes))
    for i, target in enumerate(targets):
        y_test[i, target] = 1
    
    # Calculate ROC curve and AUC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, class_name in enumerate(class_names):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], np.array(probabilities)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calculate macro-average and micro-average ROC curves
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Draw all ROC curves
    plt.figure(figsize=(14, 12))
    
    # Draw diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Draw ROC curve for each class
    for i, class_name in enumerate(class_names):
        plt.plot(fpr[i], tpr[i], lw=2,
                label=f'{class_name} (AUC = {roc_auc[i]:.4f})')
    
    # Draw macro-average ROC curve
    plt.plot(fpr["macro"], tpr["macro"], 
            label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})',
            color='navy', linestyle=':', linewidth=4)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Emotion Categories')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curves.png'))
    plt.close()
    
    # Separately draw ROC curves for difficult classes
    plt.figure(figsize=(10, 8))
    
    # Draw diagonal line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Indices of difficult classes
    difficult_classes = ['Confusion', 'Contempt', 'Disgust']
    difficult_indices = [class_names.index(c) for c in difficult_classes if c in class_names]
    
    for idx in difficult_indices:
        class_name = class_names[idx]
        plt.plot(fpr[idx], tpr[idx], lw=2,
                label=f'{class_name} (AUC = {roc_auc[idx]:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Difficult Classes')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'difficult_classes_roc.png'))
    plt.close()

# Main function
def main():
    # Set up Chinese font
    setup_chinese_font()
    
    # Set random seed
    set_seed()
    
    # Get emotion categories
    emotion_categories = get_emotion_categories()
    num_classes = len(emotion_categories)
    print(f"Emotion categories: {emotion_categories}, Category count: {num_classes}")
    
    # Use latest accuracy information to optimize training
    class_accuracies = {
        'Anger': 0.7362,
        'Confusion': 0.4353,
        'Contempt': 0.4518,
        'Disgust': 0.4653,
        'Happiness': 0.8254,
        'Neutral': 0.7044,
        'Sadness': 0.6738,
        'Surprise': 0.7254
    }
    
    # Set training parameters
    batch_size = 32  # Reduce batch size for better generalization
    num_workers = 4  # Use 4 worker processes
    num_epochs = 60  # Increase training epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=batch_size, 
        num_workers=num_workers,
        emotion_categories=emotion_categories,
        class_accuracies=class_accuracies
    )
    
    # Create model - use stronger backbone
    model = EnhancedEmotionModel(
        num_classes=num_classes,
        dropout_rates=[0.5, 0.4, 0.3],
        backbone='efficientnet_b3'  # Use stronger backbone
    )
    model = model.to(device)
    
    # Set class weights based on accuracy
    _, weights = calculate_class_weights(
        train_dir, 
        emotion_categories, 
        accuracy_dict=class_accuracies
    )
    weights = weights.to(device)
    
    # Increase focus on low-accuracy categories
    gamma = 2.5  # Increase gamma value to emphasize difficult-to-classify samples
    
    # Use improved mixed loss function
    criterion = MixedLoss(
        classes=num_classes,
        alpha=weights,
        gamma=gamma,
        smoothing=0.1,
        mix_ratio=0.8  # Increase Focal Loss proportion to focus on difficult-to-classify samples
    )
    
    # Gradual unfreezing training
    model, history, best_val_acc = train_with_gradual_unfreezing(
        model, train_loader, val_loader, criterion, device, 
        num_epochs=num_epochs, stages=3, patience=8, eval_interval=1
    )
    
    # Save best model
    model_save_dir = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel\emotion"
    # 确保目标目录存在  
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'improved_emotion_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    # Plot training history
    plot_training_history(history, data_root)
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc, test_preds, test_targets, test_probs = validate(
        model, test_loader, criterion, device
    )
    print(f"Test set accuracy: {test_acc:.4f}")
    
    # Plot confusion matrix and calculate class metrics
    plot_confusion_matrix(test_targets, test_preds, emotion_categories, data_root)
    calculate_per_class_metrics(test_targets, test_preds, emotion_categories)
    
    # Plot ROC curves to evaluate model performance for each category
    plot_roc_curves(test_targets, test_probs, emotion_categories, data_root)

# This is the critical part - ensure multiprocessing starts correctly
if __name__ == '__main__':
    # Add multiprocessing support
    multiprocessing.freeze_support()
    
    # Run main function
    main()