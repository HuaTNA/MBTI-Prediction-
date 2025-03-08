import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FacialEmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(FacialEmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)  
        self.bn1 = nn.BatchNorm2d(32)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  
        self.bn2 = nn.BatchNorm2d(64)  
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)  
        self.bn3 = nn.BatchNorm2d(128)  
        self.fc1 = nn.Linear(128 * 28 * 28, 512)  
        self.fc2 = nn.Linear(512, num_classes)  

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = torch.flatten(x, start_dim=1)  
        x = F.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, optimizer_type="Adam"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 选择损失函数
    criterion = nn.CrossEntropyLoss()

    # 选择优化器
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type. Choose 'Adam', 'SGD' or 'RMSprop'.")

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}")

    print("Training complete!")



num_epochs = 20         # 训练轮数
learning_rate = 0.0005  # 学习率
batch_size = 32         # 批次大小
optimizer_type = "Adam" # 选择优化器：'Adam', 'SGD', 'RMSprop'



from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理（调整大小、归一化）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),          
    transforms.Normalize([0.5], [0.5])  
])

# 载入数据集
train_dataset = datasets.ImageFolder(root=r"H:/CODE/APS360/Data/Face_emotion/Split/train", transform=transform)
val_dataset = datasets.ImageFolder(root=r"H:/CODE/APS360/Data/Face_emotion/Split/val", transform=transform)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 获取类别数量
num_classes = len(train_dataset.classes)
print("Number of classes:", num_classes)




# 初始化模型
model = FacialEmotionCNN(num_classes=num_classes)

# 训练模型（支持超参数调整）
train_model(model, train_loader, val_loader, num_epochs, learning_rate, optimizer_type)
