"""
Computer Vision models for fake news detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageManipulationDetector:
    """CNN-based image manipulation detection model"""
    
    def __init__(self, input_shape: Tuple[int, int, int] = (224, 224, 3)):
        self.input_shape = input_shape
        self.model = None
        self.is_trained = False
        
        self.build_model()
    
    def build_model(self):
        """Build CNN model for manipulation detection"""
        try:
            model = models.Sequential([
                # First convolutional block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                
                # Fourth convolutional block
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.MaxPooling2D(2, 2),
                layers.BatchNormalization(),
                
                # Global average pooling instead of flatten
                layers.GlobalAveragePooling2D(),
                
                # Dense layers
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.3),
                
                # Output layer
                layers.Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info("Image manipulation detection model built successfully")
            
        except Exception as e:
            logger.error(f"Error building CNN model: {e}")
            raise
    
    def train(self, images: np.ndarray, labels: np.ndarray, validation_split: float = 0.2, epochs: int = 10) -> Dict:
        """Train the manipulation detection model"""
        if self.model is None:
            raise ValueError("Model not built")
        
        try:
            # Ensure images are in correct format
            if images.max() > 1.0:
                images = images.astype(np.float32) / 255.0
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=validation_split, random_state=42, stratify=labels
            )
            
            # Data augmentation
            train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            )
            
            # Train model
            history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=32),
                steps_per_epoch=len(X_train) // 32,
                epochs=epochs,
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            self.is_trained = True
            
            # Evaluate
            val_predictions = self.model.predict(X_val)
            val_pred_binary = (val_predictions > 0.5).astype(int).flatten()
            
            val_accuracy = np.mean(val_pred_binary == y_val)
            val_auc = roc_auc_score(y_val, val_predictions)
            
            results = {
                'training_history': history.history,
                'validation_accuracy': val_accuracy,
                'validation_auc': val_auc,
                'classification_report': classification_report(y_val, val_pred_binary, output_dict=True)
            }
            
            logger.info(f"Image manipulation training completed. Val accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during image manipulation training: {e}")
            raise
    
    def predict(self, images: np.ndarray) -> Dict:
        """Predict image manipulation"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained")
        
        try:
            # Ensure images are in correct format
            if images.max() > 1.0:
                images = images.astype(np.float32) / 255.0
            
            predictions = self.model.predict(images)
            binary_predictions = (predictions > 0.5).astype(int).flatten()
            confidence_scores = predictions.flatten()
            
            results = {
                'predictions': binary_predictions.tolist(),
                'manipulation_probability': confidence_scores.tolist(),
                'authentic_probability': (1 - confidence_scores).tolist(),
                'confidence_scores': confidence_scores.tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during image manipulation prediction: {e}")
            raise
    
    def save_model(self, save_path: str):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save(os.path.join(save_path, 'manipulation_detector.h5'))
        
        # Save metadata
        metadata = {
            'input_shape': self.input_shape,
            'is_trained': self.is_trained
        }
        
        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"Image manipulation model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the trained model"""
        try:
            # Load metadata
            with open(os.path.join(load_path, 'metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            
            self.input_shape = metadata['input_shape']
            self.is_trained = metadata['is_trained']
            
            # Load model
            self.model = tf.keras.models.load_model(os.path.join(load_path, 'manipulation_detector.h5'))
            
            logger.info(f"Image manipulation model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

class OCRFeatureClassifier:
    """Classifier for OCR-extracted text features from images"""
    
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
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def extract_ocr_features(self, ocr_results: List[Dict]) -> pd.DataFrame:
        """Extract features from OCR results"""
        features = []
        
        for ocr_result in ocr_results:
            feature_dict = {
                'has_text': ocr_result.get('has_text', False),
                'text_length': len(ocr_result.get('text', '')),
                'word_count': ocr_result.get('word_count', 0),
                'confidence': ocr_result.get('confidence', 0),
                'text_density': ocr_result.get('word_count', 0) / max(len(ocr_result.get('text', '')), 1),
                'high_confidence': ocr_result.get('confidence', 0) > 80,
                'suspicious_patterns': self._detect_suspicious_patterns(ocr_result.get('text', ''))
            }
            
            features.append(feature_dict)
        
        return pd.DataFrame(features)
    
    def _detect_suspicious_patterns(self, text: str) -> int:
        """Detect suspicious patterns in extracted text"""
        if not text:
            return 0
        
        suspicious_count = 0
        text_lower = text.lower()
        
        # Check for excessive capitalization
        if sum(1 for c in text if c.isupper()) / len(text) > 0.3:
            suspicious_count += 1
        
        # Check for clickbait patterns
        clickbait_words = ['shocking', 'amazing', 'unbelievable', 'incredible', 'secret', 'exposed']
        for word in clickbait_words:
            if word in text_lower:
                suspicious_count += 1
        
        # Check for excessive punctuation
        punct_count = sum(1 for c in text if c in '!?')
        if punct_count > 3:
            suspicious_count += 1
        
        return suspicious_count
    
    def train(self, ocr_results: List[Dict], labels: List[int]) -> Dict:
        """Train the OCR feature classifier"""
        try:
            # Extract features
            features_df = self.extract_ocr_features(ocr_results)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features_df, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train classifier
            self.classifier.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate
            test_predictions = self.classifier.predict(X_test)
            test_accuracy = np.mean(test_predictions == y_test)
            
            # Get feature importance
            feature_importance = dict(zip(features_df.columns, self.classifier.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
            
            results = {
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(y_test, test_predictions, output_dict=True),
                'feature_importance': feature_importance
            }
            
            logger.info(f"OCR feature training completed. Test accuracy: {test_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during OCR feature training: {e}")
            raise
    
    def predict(self, ocr_results: List[Dict]) -> Dict:
        """Predict using OCR features"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            features_df = self.extract_ocr_features(ocr_results)
            
            predictions = self.classifier.predict(features_df)
            probabilities = self.classifier.predict_proba(features_df)
            
            fake_prob = probabilities[:, 1]
            real_prob = probabilities[:, 0]
            
            results = {
                'predictions': predictions.tolist(),
                'fake_probability': fake_prob.tolist(),
                'real_probability': real_prob.tolist(),
                'confidence_scores': np.max(probabilities, axis=1).tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error during OCR prediction: {e}")
            raise
    
    def save_model(self, save_path: str):
        """Save the trained classifier"""
        if self.classifier is None:
            raise ValueError("No classifier to save")
        
        os.makedirs(save_path, exist_ok=True)
        
        with open(os.path.join(save_path, 'ocr_classifier.pkl'), 'wb') as f:
            pickle.dump({
                'classifier': self.classifier,
                'classifier_type': self.classifier_type,
                'is_trained': self.is_trained
            }, f)
        
        logger.info(f"OCR classifier saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the trained classifier"""
        try:
            with open(os.path.join(load_path, 'ocr_classifier.pkl'), 'rb') as f:
                data = pickle.load(f)
            
            self.classifier = data['classifier']
            self.classifier_type = data['classifier_type']
            self.is_trained = data['is_trained']
            
            logger.info(f"OCR classifier loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading OCR classifier: {e}")
            raise