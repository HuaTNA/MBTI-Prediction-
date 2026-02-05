"""
Text Processing Utilities
Functions for preprocessing text data for MBTI analysis
"""

import re


def robust_text_preprocessing(text):
    """
    Preprocess text for MBTI analysis

    Performs comprehensive text cleaning including:
    - URL and email removal
    - HTML tag stripping
    - Number normalization
    - Punctuation removal
    - Stop word filtering
    - Simple stemming

    Args:
        text: Input text string to preprocess

    Returns:
        str: Cleaned and preprocessed text
    """
    # Ensure text is a string
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' email ', text)

    # Replace numbers with 'number' token
    text = re.sub(r'\d+', ' number ', text)

    # Handle punctuation - remove all non-word characters
    text = re.sub(r'[^\w\s]', ' ', text)

    # Simple word stemming
    words = text.split()
    processed_words = []

    # English stop words list
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
        # Skip stop words
        if word in stop_words:
            continue

        # Skip very short words
        if len(word) <= 2:
            continue

        # Simple stemming rules
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

    # Rejoin words
    text = ' '.join(processed_words)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
