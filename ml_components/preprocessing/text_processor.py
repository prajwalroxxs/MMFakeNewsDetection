"""
Text preprocessing utilities for fake news detection
"""
import re
import string
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Handles text preprocessing for fake news detection"""
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.setup_nltk()
        
        # Initialize BERT tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            self.tokenizer = None
        
        # Initialize lemmatizer and stop words
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def setup_nltk(self):
        """Download required NLTK data"""
        nltk_downloads = [
            'punkt', 'stopwords', 'wordnet', 'omw-1.4', 
            'vader_lexicon', 'averaged_perceptron_tagger'
        ]
        
        for download in nltk_downloads:
            try:
                nltk.data.find(f'tokenizers/{download}')
            except (LookupError, OSError):
                try:
                    nltk.download(download, quiet=True)
                except:
                    logger.warning(f"Could not download {download}")
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        if not isinstance(text, str):
            return ""
        
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text"""
        if not isinstance(text, str):
            return ""
        
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def extract_linguistic_features(self, text: str) -> Dict:
        """Extract linguistic features from text"""
        if not isinstance(text, str) or not text.strip():
            return self._empty_features()
        
        features = {}
        
        # Basic statistics
        features['char_count'] = len(text)
        features['word_count'] = len(word_tokenize(text))
        features['sentence_count'] = len(sent_tokenize(text))
        features['avg_word_length'] = np.mean([len(word) for word in word_tokenize(text)]) if features['word_count'] > 0 else 0
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['uppercase_count'] = sum(1 for c in text if c.isupper())
        features['uppercase_ratio'] = features['uppercase_count'] / len(text) if len(text) > 0 else 0
        
        # Sentiment analysis
        try:
            blob = TextBlob(text)
            features['sentiment_polarity'] = blob.sentiment.polarity
            features['sentiment_subjectivity'] = blob.sentiment.subjectivity
        except:
            features['sentiment_polarity'] = 0
            features['sentiment_subjectivity'] = 0
        
        # Readability (simple approximation)
        features['difficult_words'] = sum(1 for word in word_tokenize(text) if len(word) > 6)
        features['difficult_words_ratio'] = features['difficult_words'] / features['word_count'] if features['word_count'] > 0 else 0
        
        return features
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary"""
        return {
            'char_count': 0, 'word_count': 0, 'sentence_count': 0,
            'avg_word_length': 0, 'avg_sentence_length': 0,
            'exclamation_count': 0, 'question_count': 0,
            'uppercase_count': 0, 'uppercase_ratio': 0,
            'sentiment_polarity': 0, 'sentiment_subjectivity': 0,
            'difficult_words': 0, 'difficult_words_ratio': 0
        }
    
    def tokenize_for_bert(self, texts: List[str], max_length: int = 512) -> Dict:
        """Tokenize texts for BERT model"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='np'
        )
    
    def preprocess_batch(self, texts: List[str], include_features: bool = True) -> Dict:
        """Preprocess a batch of texts"""
        processed_texts = []
        linguistic_features = []
        
        for text in texts:
            # Clean text
            cleaned = self.clean_text(text)
            processed_texts.append(cleaned)
            
            # Extract features if requested
            if include_features:
                features = self.extract_linguistic_features(text)
                linguistic_features.append(features)
        
        result = {'processed_texts': processed_texts}
        
        if include_features:
            result['linguistic_features'] = linguistic_features
            # Convert to DataFrame for easier handling
            result['features_df'] = pd.DataFrame(linguistic_features)
        
        # Tokenize for BERT if tokenizer is available
        if self.tokenizer is not None:
            try:
                tokenized = self.tokenize_for_bert(processed_texts)
                result['tokenized'] = tokenized
            except Exception as e:
                logger.warning(f"Could not tokenize texts: {e}")
        
        return result