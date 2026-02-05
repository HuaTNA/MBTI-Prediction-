# -*- coding: utf-8 -*-
"""
MBTI Personality Type Prediction Model (No WordNet Dependency)
This version resolves zero_division warnings and improves handling of rare types.
"""

import os
import re
import string
import pandas as pd
import numpy as np
import pickle
import json
import time
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline

# Global configuration
MODEL_DIR = r"C:\Users\lnasl\Desktop\APS360\APS360\Model"
BEST_MODEL_DIR = r"C:\Users\lnasl\Desktop\APS360\APS360\Model\TrainedModel\text\ml"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# 1. Robust text preprocessing (no WordNet dependency)
def robust_text_preprocessing(text):
    """
    Comprehensive text preprocessing without relying on NLTK WordNet.

    Args:
        text (str): Raw text to process.

    Returns:
        str: Cleaned and processed text.
    """
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\S+@\S+', ' email ', text)
    text = re.sub(r'\d+', ' number ', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    words = text.split()
    processed_words = []

    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'all', 'any', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're',
        've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn',
        'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn',
        'wasn', 'weren', 'won', 'wouldn', 'am', 'is', 'are', 'was', 'were',
        'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
        'did', 'doing'
    }

    for word in words:
        if word in stop_words or len(word) <= 2:
            continue

        # Simple stemming
        if word.endswith('ing') and len(word) > 4:
            word = word[:-3]
        elif word.endswith('ed') and len(word) > 3:
            word = word[:-2]
        elif word.endswith('ly') and len(word) > 3:
            word = word[:-2]
        elif word.endswith('ies') and len(word) > 4:
            word = word[:-3] + 'y'
        elif word.endswith('es') and len(word) > 3:
            word = word[:-2]
        elif word.endswith('s') and not word.endswith('ss') and len(word) > 2:
            word = word[:-1]

        processed_words.append(word)

    text = ' '.join(processed_words)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2. Data loading and analysis
def load_data(file_path):
    """
    Load the MBTI dataset and ensure correct formatting.

    Args:
        file_path (str): Path to CSV or Excel file.

    Returns:
        DataFrame: Loaded dataset.
    """
    print(f"Loading data from: {file_path}")

    try:
        if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"Trying alternative encodings: {e}")
            for encoding in ['cp1252', 'ISO-8859-1', 'cp850']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    print(f"Successfully loaded using {encoding}")
                    break
                except:
                    continue

    if 'type' not in df.columns or 'posts' not in df.columns:
        type_col = None
        post_col = None

        # Try detecting MBTI and text columns
        for col in df.columns:
            sample_values = df[col].astype(str).str.upper().tolist()[:20]
            mbti_count = sum(1 for val in sample_values if re.match(r'^[IE][NS][TF][JP]$', val.strip()))
            if mbti_count > len(sample_values) * 0.5:
                type_col = col
                print(f"Detected MBTI column: {col}")
                break

        text_lens = {col: df[col].astype(str).str.len().mean() for col in df.columns if col != type_col}
        if text_lens:
            post_col = max(text_lens, key=text_lens.get)
            print(f"Detected text column: {post_col}")

        if type_col and post_col:
            df = df[[type_col, post_col]]
            df.columns = ['type', 'posts']
        else:
            raise ValueError("Could not identify MBTI and text columns")

    print(f"Dataset shape: {df.shape}")
    print(f"Null values:\n{df.isnull().sum()}")
    df = df.fillna({'type': '', 'posts': ''})
    return df

def analyze_mbti_data(df):
    """
    Perform exploratory analysis on MBTI data.

    Args:
        df (DataFrame): Raw dataset.

    Returns:
        DataFrame: Valid filtered dataset.
    """
    df['type'] = df['type'].str.upper().str.strip()
    valid_df = df[df['type'].str.match(r'^[IE][NS][TF][JP]$')]
    invalid_count = len(df) - len(valid_df)
    print(f"Valid MBTI types: {len(valid_df)} / {len(df)} (Removed {invalid_count} invalid rows)")

    type_counts = Counter(valid_df['type'])
    total = len(valid_df)
    print("\nMBTI Type Distribution:")
    for mbti_type, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"{mbti_type}: {count} ({(count / total) * 100:.2f}%)")

    dimensions = {'I/E': {'I': 0, 'E': 0}, 'N/S': {'N': 0, 'S': 0},
                  'T/F': {'T': 0, 'F': 0}, 'J/P': {'J': 0, 'P': 0}}
    for mbti_type in valid_df['type']:
        dimensions['I/E'][mbti_type[0]] += 1
        dimensions['N/S'][mbti_type[1]] += 1
        dimensions['T/F'][mbti_type[2]] += 1
        dimensions['J/P'][mbti_type[3]] += 1

    print("\nDimension Distribution:")
    for dim, counts in dimensions.items():
        total_dim = sum(counts.values())
        for trait, count in counts.items():
            print(f"{dim} - {trait}: {count} ({(count / total_dim) * 100:.2f}%)")

    valid_df['text_length'] = valid_df['posts'].astype(str).str.len()
    print(f"\nText Length Statistics:")
    print(f"Average length: {valid_df['text_length'].mean():.2f}")
    print(f"Max length: {valid_df['text_length'].max()}")
    print(f"Min length: {valid_df['text_length'].min()}")
    return valid_df


# 3. Data balancing and augmentation
def enhance_class_balance(df, target_ratio=0.7):
    """
    Improve dataset balance by generating synthetic samples for rare types.

    Args:
        df (DataFrame): Original dataset.
        target_ratio (float): Target proportion relative to the majority class.

    Returns:
        DataFrame: Balanced dataset.
    """
    print("Enhancing dataset balance...")

    type_counts = Counter(df['type'])
    max_count = max(type_counts.values())
    target_count = int(max_count * target_ratio)

    print(f"Largest class count: {max_count}")
    print(f"Target class count: {target_count}")

    balanced_data = [df]

    for mbti_type, count in type_counts.items():
        if count < target_count:
            print(f"Augmenting class {mbti_type}: {count} → {target_count}")
            type_data = df[df['type'] == mbti_type]
            samples_needed = target_count - count
            enhanced_samples = []

            for _ in range(samples_needed):
                sample = type_data.sample(1).iloc[0]
                processed_text = sample['posts']
                words = processed_text.split()

                if len(words) >= 10:
                    split_point = len(words) // 3
                    if split_point > 0:
                        first_part = words[:split_point]
                        np.random.shuffle(first_part)
                        words = first_part + words[split_point:]

                    mbti_keywords = get_mbti_keywords(mbti_type)
                    selected_keywords = np.random.choice(
                        mbti_keywords, size=min(3, len(mbti_keywords)), replace=False
                    )

                    for keyword in selected_keywords:
                        insert_pos = np.random.randint(0, len(words))
                        words.insert(insert_pos, keyword)

                enhanced_text = ' '.join(words)
                enhanced_sample = sample.copy()
                enhanced_sample['posts'] = enhanced_text
                enhanced_samples.append(enhanced_sample)

            if enhanced_samples:
                balanced_data.append(pd.DataFrame(enhanced_samples))

    balanced_df = pd.concat(balanced_data, ignore_index=True)
    print(f"Balanced dataset size: {balanced_df.shape}")
    return balanced_df


def get_mbti_keywords(mbti_type):
    """
    Return a list of characteristic keywords associated with a given MBTI type.

    Args:
        mbti_type (str): MBTI type (e.g., "INTJ").

    Returns:
        list: List of relevant keywords.
    """
    dimension_keywords = {
        'I': ['introvert', 'quiet', 'reflect', 'alone', 'private', 'inner', 'depth', 'focus', 'thought'],
        'E': ['extrovert', 'social', 'talk', 'people', 'engage', 'outgoing', 'active', 'external', 'interact'],
        'N': ['intuitive', 'abstract', 'future', 'imagine', 'possibility', 'pattern', 'meaning', 'theory', 'concept'],
        'S': ['sensing', 'detail', 'present', 'practical', 'concrete', 'reality', 'fact', 'experience', 'observation'],
        'T': ['thinking', 'logic', 'analysis', 'objective', 'principle', 'rational', 'critique', 'reason', 'system'],
        'F': ['feeling', 'value', 'harmony', 'empathy', 'personal', 'compassion', 'ethic', 'human', 'subjective'],
        'J': ['judging', 'plan', 'organize', 'structure', 'decide', 'control', 'certain', 'schedule', 'complete'],
        'P': ['perceiving', 'flexible', 'adapt', 'explore', 'option', 'spontaneous', 'open', 'process', 'possibility']
    }

    keywords = []
    for letter in mbti_type:
        if letter in dimension_keywords:
            keywords.extend(dimension_keywords[letter])
    return keywords


# 4. Feature preparation
def prepare_features(df):
    """
    Prepare text features and labels for training.

    Args:
        df (DataFrame): Preprocessed dataset.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights)
    """
    print("Preparing features...")

    df['processed_posts'] = df['posts'].apply(robust_text_preprocessing)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['type'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_posts'], y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        use_idf=True,
        stop_words='english'
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"Training features shape: {X_train_tfidf.shape}")
    print(f"Testing features shape: {X_test_tfidf.shape}")

    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    n_classes = len(class_counts)
    class_weights = {i: total_samples / (n_classes * count) for i, count in enumerate(class_counts)}

    print("Class weights:")
    for i, weight in class_weights.items():
        mbti_type = label_encoder.inverse_transform([i])[0]
        print(f"  {mbti_type}: {weight:.2f}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer, label_encoder, class_weights


# 5. Model training and evaluation
def train_and_evaluate_models(X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights):
    """
    Train and evaluate multiple models.

    Args:
        X_train, X_test, y_train, y_test: Training and testing sets.
        vectorizer: TF-IDF vectorizer.
        label_encoder: Label encoder.
        class_weights: Computed class weights.

    Returns:
        tuple: (results dictionary, best model name)
    """
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight=class_weights,
            solver='liblinear',
            multi_class='ovr',
            random_state=42
        ),
        'LinearSVC': LinearSVC(
            max_iter=2000,
            C=1.0,
            class_weight=class_weights,
            dual=False,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=40,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} model...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        print(f"{name} training time: {train_time:.2f} s")
        print(f"{name} accuracy: {accuracy:.4f}")
        print(f"{name} F1 score: {f1:.4f}")

        target_names = label_encoder.classes_
        report = classification_report(
            y_test, y_pred, target_names=target_names, digits=4, zero_division=0
        )
        print(f"Classification report:\n{report}")

        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'f1_score': f1,
            'train_time': train_time,
            'report': classification_report(
                y_test, y_pred,
                target_names=target_names,
                digits=4,
                zero_division=0,
                output_dict=True
            )
        }

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")

    return results, best_model_name


# 6. Detailed prediction analysis
def analyze_predictions(model, X_test, y_test, label_encoder):
    """
    Analyze detailed performance of the trained model.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.
        label_encoder: Label encoder.

    Returns:
        dict: Performance metrics.
    """
    print("\nAnalyzing model predictions...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = label_encoder.classes_

    class_performance = {}
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        class_performance[class_name] = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(cm[i, :].sum())
        }

    worst_classes = sorted(class_performance.items(), key=lambda x: x[1]['f1_score'])[:3]
    print("Lowest-performing MBTI types:")
    for class_name, metrics in worst_classes:
        print(f"  {class_name}: F1={metrics['f1_score']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    return {
        'confusion_matrix': cm.tolist(),
        'class_performance': class_performance
    }
# 7. Save and load models
def save_model(model, vectorizer, label_encoder, model_name, metadata=None):
    """
    Save a trained model and related components.

    Args:
        model: Trained model instance.
        vectorizer: TF-IDF vectorizer.
        label_encoder: Label encoder.
        model_name: Directory name to store this model under MODEL_DIR.
        metadata: Optional extra metadata dict.

    Returns:
        str: Path where the model is saved.
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    with open(os.path.join(model_path, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(model_path, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    config = {
        'model_name': model_name,
        'classes': label_encoder.classes_.tolist(),
        'feature_count': vectorizer.max_features,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    if metadata:
        config.update(metadata)

    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Model saved to: {model_path}")
    return model_path


def save_best_model(results, best_model_name, vectorizer, label_encoder, analysis=None):
    """
    Save the best model and its artifacts into the dedicated BEST_MODEL_DIR.

    Args:
        results: Dict of model results from training.
        best_model_name: Name of the best-performing model.
        vectorizer: TF-IDF vectorizer.
        label_encoder: Label encoder.
        analysis: Optional detailed analysis dict.

    Returns:
        str: Path to BEST_MODEL_DIR where the best model is saved.
    """
    best_model = results[best_model_name]['model']

    metadata = {
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'train_time': results[best_model_name]['train_time'],
        'report': results[best_model_name]['report']
    }
    if analysis:
        metadata['analysis'] = analysis

    if os.path.exists(BEST_MODEL_DIR):
        import shutil
        shutil.rmtree(BEST_MODEL_DIR)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    with open(os.path.join(BEST_MODEL_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(best_model, f)

    with open(os.path.join(BEST_MODEL_DIR, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    with open(os.path.join(BEST_MODEL_DIR, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    config = {
        'model_name': best_model_name,
        'classes': label_encoder.classes_.tolist(),
        'feature_count': vectorizer.max_features,
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    with open(os.path.join(BEST_MODEL_DIR, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    with open(os.path.join(BEST_MODEL_DIR, 'report.json'), 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"Best model saved to: {BEST_MODEL_DIR}")
    return BEST_MODEL_DIR


def load_best_model():
    """
    Load the best saved model from BEST_MODEL_DIR.

    Returns:
        tuple: (model, vectorizer, label_encoder, config)
    """
    if not os.path.exists(BEST_MODEL_DIR):
        raise FileNotFoundError(f"Best model directory not found: {BEST_MODEL_DIR}")

    with open(os.path.join(BEST_MODEL_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    with open(os.path.join(BEST_MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)

    with open(os.path.join(BEST_MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    with open(os.path.join(BEST_MODEL_DIR, 'config.json'), 'r') as f:
        config = json.load(f)

    return model, vectorizer, label_encoder, config


# 8. Advanced prediction utilities
def predict_mbti_type(text, model=None, vectorizer=None, label_encoder=None):
    """
    Predict MBTI type from input text. Supports paragraph input.

    Args:
        text (str): Input text.
        model, vectorizer, label_encoder: Optional components; if not provided, the best model is loaded.

    Returns:
        dict: Prediction result containing type, confidence, and dimension analysis.
    """
    if model is None or vectorizer is None or label_encoder is None:
        try:
            model, vectorizer, label_encoder, _ = load_best_model()
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    processed_text = robust_text_preprocessing(text)
    text_vector = vectorizer.transform([processed_text])

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_vector)[0]
        prediction = int(np.argmax(proba))
        confidence = float(proba[prediction])
    else:
        prediction = int(model.predict(text_vector)[0])
        confidence = 1.0

    mbti_type = label_encoder.inverse_transform([prediction])[0]

    ie_score, ns_score, tf_score, jp_score = analyze_mbti_dimensions(text, processed_text)

    alt_type = ''
    alt_type += 'I' if ie_score < 0 else 'E'
    alt_type += 'N' if ns_score > 0 else 'S'
    alt_type += 'T' if tf_score < 0 else 'F'
    alt_type += 'J' if jp_score < 0 else 'P'

    result = {
        'mbti_type': mbti_type,
        'confidence': confidence,
        'alternative_type': alt_type,
        'dimension_analysis': {
            'IE': {'score': ie_score, 'preference': 'I' if ie_score < 0 else 'E'},
            'NS': {'score': ns_score, 'preference': 'N' if ns_score > 0 else 'S'},
            'TF': {'score': tf_score, 'preference': 'T' if tf_score < 0 else 'F'},
            'JP': {'score': jp_score, 'preference': 'J' if jp_score < 0 else 'P'}
        }
    }

    if hasattr(model, 'predict_proba'):
        top_indices = proba.argsort()[-3:][::-1]
        result['top_predictions'] = [
            {'type': label_encoder.inverse_transform([idx])[0], 'confidence': float(proba[idx])}
            for idx in top_indices
        ]

    return result


def analyze_mbti_dimensions(text, processed_text=None):
    """
    Heuristic scoring across the four MBTI dimensions based on keyword presence.

    Args:
        text (str): Original text.
        processed_text (str): Preprocessed text (optional).

    Returns:
        tuple: (IE, NS, TF, JP) scores.
    """
    if processed_text is None:
        processed_text = robust_text_preprocessing(text)

    dimension_keywords = {
        'I': {'introvert': 2, 'quiet': 1, 'reflect': 1, 'alone': 1, 'private': 1, 'inner': 1,
              'depth': 1, 'focus': 1, 'thought': 1, 'peace': 1, 'solitude': 2, 'individual': 1},
        'E': {'extrovert': 2, 'social': 1, 'talk': 1, 'people': 1, 'engage': 1, 'outgoing': 1,
              'active': 1, 'external': 1, 'interact': 1, 'energetic': 1, 'group': 1, 'party': 1},
        'N': {'intuitive': 2, 'abstract': 1, 'future': 1, 'imagine': 1, 'possibility': 1, 'pattern': 1,
              'meaning': 1, 'theory': 1, 'concept': 1, 'insight': 1, 'innovative': 1, 'vision': 1},
        'S': {'sensing': 2, 'detail': 1, 'present': 1, 'practical': 1, 'concrete': 1, 'reality': 1,
              'fact': 1, 'experience': 1, 'observation': 1, 'specific': 1, 'actual': 1, 'tangible': 1},
        'T': {'thinking': 2, 'logic': 1, 'analysis': 1, 'objective': 1, 'principle': 1, 'rational': 1,
              'critique': 1, 'reason': 1, 'system': 1, 'truth': 1, 'fair': 1, 'consistent': 1},
        'F': {'feeling': 2, 'value': 1, 'harmony': 1, 'empathy': 1, 'personal': 1, 'compassion': 1,
              'ethic': 1, 'human': 1, 'subjective': 1, 'emotion': 1, 'care': 1, 'moral': 1},
        'J': {'judging': 2, 'plan': 1, 'organize': 1, 'structure': 1, 'decide': 1, 'control': 1,
              'certain': 1, 'schedule': 1, 'complete': 1, 'deadline': 1, 'goal': 1, 'closure': 1},
        'P': {'perceiving': 2, 'flexible': 1, 'adapt': 1, 'explore': 1, 'option': 1, 'spontaneous': 1,
              'open': 1, 'process': 1, 'possibility': 1, 'casual': 1, 'flow': 1, 'discover': 1}
    }

    ie_score = 0   # negative favors I, positive favors E
    ns_score = 0   # positive favors N, negative favors S
    tf_score = 0   # negative favors T, positive favors F
    jp_score = 0   # negative favors J, positive favors P

    words = processed_text.split()

    for word in words:
        for keyword, weight in dimension_keywords['I'].items():
            if keyword in word:
                ie_score -= weight
        for keyword, weight in dimension_keywords['E'].items():
            if keyword in word:
                ie_score += weight

        for keyword, weight in dimension_keywords['N'].items():
            if keyword in word:
                ns_score += weight
        for keyword, weight in dimension_keywords['S'].items():
            if keyword in word:
                ns_score -= weight

        for keyword, weight in dimension_keywords['T'].items():
            if keyword in word:
                tf_score -= weight
        for keyword, weight in dimension_keywords['F'].items():
            if keyword in word:
                tf_score += weight

        for keyword, weight in dimension_keywords['J'].items():
            if keyword in word:
                jp_score -= weight
        for keyword, weight in dimension_keywords['P'].items():
            if keyword in word:
                jp_score += weight

    def normalize_score(score, words_count):
        if words_count == 0:
            return 0
        normalized = score / (words_count ** 0.5)
        return max(min(normalized, 10), -10)

    words_count = len(words)
    ie_score = normalize_score(ie_score, words_count)
    ns_score = normalize_score(ns_score, words_count)
    tf_score = normalize_score(tf_score, words_count)
    jp_score = normalize_score(jp_score, words_count)

    return ie_score, ns_score, tf_score, jp_score


# 9. End-to-end training pipeline
def train_mbti_prediction_model(file_path):
    """
    End-to-end training procedure for the MBTI prediction model.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        dict: Summary of training results.
    """
    print("=" * 80)
    print("Starting MBTI Personality Type Prediction Model Training")
    print("=" * 80)

    raw_df = load_data(file_path)
    valid_df = analyze_mbti_data(raw_df)
    balanced_df = enhance_class_balance(valid_df)

    X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights = prepare_features(balanced_df)

    results, best_model_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, vectorizer, label_encoder, class_weights
    )

    best_model = results[best_model_name]['model']
    analysis = analyze_predictions(best_model, X_test, y_test, label_encoder)

    for name, result in results.items():
        model_metadata = {
            'accuracy': result['accuracy'],
            'f1_score': result['f1_score'],
            'train_time': result['train_time'],
            'report': result['report']
        }
        save_model(result['model'], vectorizer, label_encoder, name, model_metadata)

    best_model_path = save_best_model(results, best_model_name, vectorizer, label_encoder, analysis)

    print("\nTraining completed.")
    print(f"Best model ({best_model_name}) saved to: {best_model_path}")

    return {
        'best_model': best_model_name,
        'accuracy': results[best_model_name]['accuracy'],
        'f1_score': results[best_model_name]['f1_score'],
        'best_model_path': best_model_path
    }


# 10. Quick test and demo
def test_model_with_examples():
    """
    Quick sanity check using a few example texts.
    """
    print("\nTesting model predictions...")

    try:
        model, vectorizer, label_encoder, config = load_best_model()
        print(f"Loaded model successfully. Model type: {config['model_name']}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    test_texts = [
        (
            "I like spending time alone, reflecting on the universe and the meaning of life. "
            "I am always looking for deeper understanding and connections between seemingly "
            "unrelated concepts. I enjoy theoretical discussions more than practical matters."
        ),
        (
            "I love being around people and organizing social events. I am very practical "
            "and focus on immediate results. I believe in traditions and value stability. "
            "I prefer clear rules and established procedures."
        ),
        (
            "I make decisions based on logic and objective analysis. I enjoy solving complex "
            "problems and finding efficient solutions. I value competence and intellectual "
            "discussions. I tend to focus on concepts rather than details."
        )
    ]

    for i, text in enumerate(test_texts):
        result = predict_mbti_type(text, model, vectorizer, label_encoder)
        print(f"\nExample {i + 1}:")
        print(f"Predicted MBTI type: {result['mbti_type']} (Confidence: {result['confidence'] * 100:.2f}%)")
        print(f"Alternative type (by dimension heuristic): {result['alternative_type']}")
        print("Dimension analysis:")
        for dim, analysis in result['dimension_analysis'].items():
            print(f"  {dim}: {analysis['score']:.2f} (leans {analysis['preference']})")
        if 'top_predictions' in result:
            print("Top-3 types by probability:")
            for pred in result['top_predictions']:
                print(f"  {pred['type']}: {pred['confidence'] * 100:.2f}%")

    print("\nDemo completed.")


# Entry point
if __name__ == "__main__":
    file_path = r"C:\Users\lnasl\Desktop\APS360\APS360\Data\Text\mbti_1.csv"

    print("Select an option:")
    print("1. Train a new model")
    print("2. Test an existing best model")

    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        results = train_mbti_prediction_model(file_path)
        print(f"Training results: {results}")
        test_model_with_examples()
    elif choice == "2":
        test_model_with_examples()
    else:
        print("Invalid option")
