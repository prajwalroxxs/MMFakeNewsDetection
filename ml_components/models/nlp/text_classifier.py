"""
NLP-based fake news detection model
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTFakeNewsClassifier:
    """BERT-based fake news classifier"""
    
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.is_trained = False
        
        self.setup_model()
    
    def setup_model(self):
        """Initialize BERT model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Create BERT-based classification model
            bert_model = TFAutoModel.from_pretrained(self.model_name)
            
            # Build classification head
            input_ids = tf.keras.Input(shape=(self.max_length,), dtype=tf.int32, name='input_ids')
            attention_mask = tf.keras.Input(shape=(self.max_length,), dtype=tf.int32, name='attention_mask')
            
            bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = bert_output.pooler_output
            
            # Add dropout and classification layer
            dropout = tf.keras.layers.Dropout(0.3)(pooled_output)
            output = tf.keras.layers.Dense(1, activation='sigmoid', name='classification')(dropout)
            
            self.model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
            
            # Compile model
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            logger.info(f"BERT model initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Error initializing BERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    def prepare_data(self, texts: List[str]) -> Dict:
        """Prepare text data for BERT"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized")
        
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors='tf'
        )
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask']
        }
    
    def train(self, texts: List[str], labels: List[int], validation_split: float = 0.2, epochs: int = 3) -> Dict:
        """Train the BERT model"""
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            # Prepare data
            data = self.prepare_data(texts)
            labels_array = np.array(labels)
            
            # Split data
            train_data, val_data, train_labels, val_labels = train_test_split(
                [data['input_ids'], data['attention_mask']], 
                labels_array, 
                test_size=validation_split, 
                random_state=42,
                stratify=labels_array
            )
            
            train_input_ids, train_attention_mask = train_data[0], train_data[1]
            val_input_ids, val_attention_mask = val_data[0], val_data[1]
            
            # Train model
            history = self.model.fit(
                [train_input_ids, train_attention_mask],
                train_labels,
                validation_data=([val_input_ids, val_attention_mask], val_labels),
                epochs=epochs,
                batch_size=16,
                verbose=1
            )
            
            self.is_trained = True
            
            # Evaluate on validation set
            val_predictions = self.model.predict([val_input_ids, val_attention_mask])
            val_pred_binary = (val_predictions > 0.5).astype(int).flatten()
            
            # Calculate metrics
            val_accuracy = np.mean(val_pred_binary == val_labels)
            val_auc = roc_auc_score(val_labels, val_predictions)
            
            results = {
                'training_history': history.history,
                'validation_accuracy': val_accuracy,
                'validation_auc': val_auc,
                'classification_report': classification_report(val_labels, val_pred_binary, output_dict=True)
            }
            
            logger.info(f"Training completed. Validation accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, texts: List[str]) -> Dict:
        """Make predictions on new texts"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        try:
            # Prepare data
            data = self.prepare_data(texts)
            
            # Make predictions
            predictions = self.model.predict([data['input_ids'], data['attention_mask']])
            
            # Convert to binary predictions
            binary_predictions = (predictions > 0.5).astype(int).flatten()
            confidence_scores = predictions.flatten()
            
            results = {
                'predictions': binary_predictions.tolist(),
                'confidence_scores': confidence_scores.tolist(),
                'fake_probability': confidence_scores.tolist(),  # Probability of being fake
                'real_probability': (1 - confidence_scores).tolist()  # Probability of being real
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_model(self, save_path: str):
        """Save trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(save_path, 'bert_model'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(save_path, 'tokenizer'))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'max_length': self.max_length,
            'is_trained': self.is_trained
        }
        
        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load trained model"""
        try:
            # Load metadata
            with open(os.path.join(load_path, 'metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            
            self.model_name = metadata['model_name']
            self.max_length = metadata['max_length']
            self.is_trained = metadata['is_trained']
            
            # Load model
            self.model = tf.keras.models.load_model(os.path.join(load_path, 'bert_model'))
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_path, 'tokenizer'))
            
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

class FeatureBasedClassifier:
    """Feature-based fake news classifier using linguistic features"""
    
    def __init__(self, classifier_type: str = 'random_forest'):
        self.classifier_type = classifier_type
        self.classifier = None
        self.is_trained = False
        
        self.setup_classifier()
    
    def setup_classifier(self):
        """Initialize the classifier"""
        if self.classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
        elif self.classifier_type == 'logistic_regression':
            self.classifier = LogisticRegression(
                random_state=42,
                class_weight='balanced',
                max_iter=1000
            )
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def train(self, features_df: pd.DataFrame, labels: List[int]) -> Dict:
        """Train the feature-based classifier"""
        try:
            # Prepare features
            X = features_df.fillna(0)  # Handle missing values
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train classifier
            self.classifier.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate
            train_predictions = self.classifier.predict(X_train)
            test_predictions = self.classifier.predict(X_test)
            
            train_accuracy = np.mean(train_predictions == y_train)
            test_accuracy = np.mean(test_predictions == y_test)
            
            # Get feature importance (if available)
            feature_importance = None
            if hasattr(self.classifier, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, self.classifier.feature_importances_))
                # Sort by importance
                feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            results = {
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(y_test, test_predictions, output_dict=True),
                'feature_importance': feature_importance
            }
            
            logger.info(f"Feature-based training completed. Test accuracy: {test_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during feature-based training: {e}")
            raise
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """Make predictions using features"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            X = features_df.fillna(0)
            
            predictions = self.classifier.predict(X)
            
            # Get prediction probabilities if available
            if hasattr(self.classifier, 'predict_proba'):
                probabilities = self.classifier.predict_proba(X)
                fake_prob = probabilities[:, 1]  # Assuming class 1 is fake
                real_prob = probabilities[:, 0]  # Assuming class 0 is real
            else:
                fake_prob = predictions.astype(float)
                real_prob = 1 - predictions.astype(float)
            
            results = {
                'predictions': predictions.tolist(),
                'fake_probability': fake_prob.tolist(),
                'real_probability': real_prob.tolist(),
                'confidence_scores': np.max([fake_prob, real_prob], axis=0).tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def save_model(self, save_path: str):
        """Save the trained classifier"""
        if self.classifier is None:
            raise ValueError("No classifier to save")
        
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'feature_classifier.pkl'), 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'classifier_type': self.classifier_type,
                'is_trained': self.is_trained
            }, f)
        
        logger.info(f"Feature-based classifier saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the trained classifier"""
        try:
            with open(os.path.join(load_path, 'feature_classifier.pkl'), 'rb') as f:
                data = pickle.load(f)
            
            self.classifier = data['classifier']
            self.classifier_type = data['classifier_type']
            self.is_trained = data['is_trained']
            
            logger.info(f"Feature-based classifier loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            raise