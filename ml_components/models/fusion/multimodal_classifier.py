"""
Multi-modal fusion system for fake news detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pickle
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiModalFusionClassifier:
    """Combines text and image predictions for final classification"""
    
    def __init__(self, fusion_method: str = 'weighted_average'):
        """
        Initialize multimodal fusion classifier
        
        Args:
            fusion_method: 'weighted_average', 'voting', 'meta_learner'
        """
        self.fusion_method = fusion_method
        self.meta_classifier = None
        self.text_weight = 0.6  # Default weights
        self.image_weight = 0.4
        self.is_trained = False
        
        if fusion_method == 'meta_learner':
            self.setup_meta_classifier()
    
    def setup_meta_classifier(self):
        """Setup meta-classifier for fusion"""
        self.meta_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        )
    
    def weighted_average_fusion(self, text_predictions: Dict, image_predictions: Dict) -> Dict:
        """Combine predictions using weighted average"""
        try:
            text_probs = np.array(text_predictions.get('fake_probability', []))
            image_probs = np.array(image_predictions.get('fake_probability', []))
            
            # Handle cases where one modality is missing
            if len(text_probs) == 0 and len(image_probs) == 0:
                raise ValueError("No predictions available from either modality")
            elif len(text_probs) == 0:
                # Only image predictions available
                combined_probs = image_probs
                self.text_weight, self.image_weight = 0.0, 1.0
            elif len(image_probs) == 0:
                # Only text predictions available
                combined_probs = text_probs
                self.text_weight, self.image_weight = 1.0, 0.0
            else:
                # Both modalities available
                combined_probs = (self.text_weight * text_probs + 
                                self.image_weight * image_probs)
            
            combined_predictions = (combined_probs > 0.5).astype(int)
            
            results = {
                'predictions': combined_predictions.tolist(),
                'fake_probability': combined_probs.tolist(),
                'real_probability': (1 - combined_probs).tolist(),
                'confidence_scores': combined_probs.tolist(),
                'fusion_method': 'weighted_average',
                'text_weight': self.text_weight,
                'image_weight': self.image_weight
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in weighted average fusion: {e}")
            raise
    
    def voting_fusion(self, text_predictions: Dict, image_predictions: Dict) -> Dict:
        """Combine predictions using majority voting"""
        try:
            text_preds = np.array(text_predictions.get('predictions', []))
            image_preds = np.array(image_predictions.get('predictions', []))
            
            text_probs = np.array(text_predictions.get('fake_probability', []))
            image_probs = np.array(image_predictions.get('fake_probability', []))
            
            # Handle missing modalities
            if len(text_preds) == 0 and len(image_preds) == 0:
                raise ValueError("No predictions available from either modality")
            elif len(text_preds) == 0:
                combined_predictions = image_preds
                combined_probs = image_probs
            elif len(image_preds) == 0:
                combined_predictions = text_preds
                combined_probs = text_probs
            else:
                # Majority voting
                combined_predictions = ((text_preds + image_preds) >= 1).astype(int)
                combined_probs = (text_probs + image_probs) / 2
            
            results = {
                'predictions': combined_predictions.tolist(),
                'fake_probability': combined_probs.tolist(),
                'real_probability': (1 - combined_probs).tolist(),
                'confidence_scores': np.abs(combined_probs - 0.5).tolist(),
                'fusion_method': 'voting'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in voting fusion: {e}")
            raise
    
    def meta_learner_fusion(self, text_predictions: Dict, image_predictions: Dict, 
                          additional_features: Optional[Dict] = None) -> Dict:
        """Use meta-learner for fusion"""
        if not self.is_trained or self.meta_classifier is None:
            raise ValueError("Meta-classifier not trained")
        
        try:
            # Prepare features for meta-classifier
            features = self._prepare_meta_features(text_predictions, image_predictions, additional_features)
            
            # Make predictions
            predictions = self.meta_classifier.predict(features)
            probabilities = self.meta_classifier.predict_proba(features)
            
            fake_probs = probabilities[:, 1]
            
            results = {
                'predictions': predictions.tolist(),
                'fake_probability': fake_probs.tolist(),
                'real_probability': (1 - fake_probs).tolist(),
                'confidence_scores': np.max(probabilities, axis=1).tolist(),
                'fusion_method': 'meta_learner'
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in meta-learner fusion: {e}")
            raise
    
    def _prepare_meta_features(self, text_predictions: Dict, image_predictions: Dict, 
                              additional_features: Optional[Dict] = None) -> np.ndarray:
        """Prepare features for meta-classifier"""
        features_list = []
        
        # Text features
        text_fake_prob = text_predictions.get('fake_probability', [])
        text_confidence = text_predictions.get('confidence_scores', [])
        
        # Image features  
        image_fake_prob = image_predictions.get('fake_probability', [])
        image_confidence = image_predictions.get('confidence_scores', [])
        
        # Determine the number of samples
        n_samples = max(len(text_fake_prob), len(image_fake_prob))
        
        for i in range(n_samples):
            sample_features = []
            
            # Text features
            if i < len(text_fake_prob):
                sample_features.extend([text_fake_prob[i], text_confidence[i]])
            else:
                sample_features.extend([0.5, 0.0])  # Default values
            
            # Image features
            if i < len(image_fake_prob):
                sample_features.extend([image_fake_prob[i], image_confidence[i]])
            else:
                sample_features.extend([0.5, 0.0])  # Default values
            
            # Additional features
            if additional_features:
                for key, values in additional_features.items():
                    if i < len(values):
                        sample_features.append(values[i])
                    else:
                        sample_features.append(0.0)
            
            features_list.append(sample_features)
        
        return np.array(features_list)
    
    def train_meta_classifier(self, training_data: List[Dict], labels: List[int]) -> Dict:
        """Train the meta-classifier using predictions from individual models"""
        if self.fusion_method != 'meta_learner':
            raise ValueError("Meta-classifier training only available for meta_learner fusion method")
        
        try:
            # Prepare training features
            all_features = []
            for data in training_data:
                text_pred = data.get('text_predictions', {})
                image_pred = data.get('image_predictions', {})
                additional = data.get('additional_features', {})
                
                features = self._prepare_meta_features(text_pred, image_pred, additional)
                all_features.append(features[0])  # Assuming single sample per data point
            
            X = np.array(all_features)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train meta-classifier
            self.meta_classifier.fit(X_train, y_train)
            self.is_trained = True
            
            # Evaluate
            test_predictions = self.meta_classifier.predict(X_test)
            test_accuracy = np.mean(test_predictions == y_test)
            
            # Get feature importance
            feature_names = ['text_fake_prob', 'text_confidence', 'image_fake_prob', 'image_confidence']
            feature_importance = dict(zip(feature_names, self.meta_classifier.feature_importances_))
            
            results = {
                'test_accuracy': test_accuracy,
                'classification_report': classification_report(y_test, test_predictions, output_dict=True),
                'feature_importance': feature_importance
            }
            
            logger.info(f"Meta-classifier training completed. Test accuracy: {test_accuracy:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error training meta-classifier: {e}")
            raise
    
    def fuse_predictions(self, text_predictions: Dict, image_predictions: Dict, 
                        additional_features: Optional[Dict] = None) -> Dict:
        """Main fusion method dispatcher"""
        try:
            if self.fusion_method == 'weighted_average':
                return self.weighted_average_fusion(text_predictions, image_predictions)
            elif self.fusion_method == 'voting':
                return self.voting_fusion(text_predictions, image_predictions)
            elif self.fusion_method == 'meta_learner':
                return self.meta_learner_fusion(text_predictions, image_predictions, additional_features)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
                
        except Exception as e:
            logger.error(f"Error in fusion: {e}")
            raise
    
    def update_weights(self, text_weight: float, image_weight: float):
        """Update fusion weights for weighted average method"""
        if text_weight + image_weight != 1.0:
            logger.warning("Weights do not sum to 1.0, normalizing...")
            total = text_weight + image_weight
            text_weight /= total
            image_weight /= total
        
        self.text_weight = text_weight
        self.image_weight = image_weight
        
        logger.info(f"Updated weights: text={text_weight:.3f}, image={image_weight:.3f}")
    
    def save_model(self, save_path: str):
        """Save the fusion model"""
        os.makedirs(save_path, exist_ok=True)
        
        model_data = {
            'fusion_method': self.fusion_method,
            'text_weight': self.text_weight,
            'image_weight': self.image_weight,
            'is_trained': self.is_trained,
            'meta_classifier': self.meta_classifier
        }
        
        with open(os.path.join(save_path, 'fusion_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Fusion model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load the fusion model"""
        try:
            with open(os.path.join(load_path, 'fusion_model.pkl'), 'rb') as f:
                model_data = pickle.load(f)
            
            self.fusion_method = model_data['fusion_method']
            self.text_weight = model_data['text_weight']
            self.image_weight = model_data['image_weight']
            self.is_trained = model_data['is_trained']
            self.meta_classifier = model_data['meta_classifier']
            
            logger.info(f"Fusion model loaded from {load_path}")
            
        except Exception as e:
            logger.error(f"Error loading fusion model: {e}")
            raise

class FakeNewsDetectionSystem:
    """Complete multi-modal fake news detection system"""
    
    def __init__(self):
        self.text_classifier = None
        self.image_classifier = None
        self.ocr_classifier = None
        self.fusion_classifier = None
        
        self.text_processor = None
        self.image_processor = None
        
        self.is_initialized = False
    
    def initialize_components(self, text_classifier, image_classifier, ocr_classifier, 
                            fusion_classifier, text_processor, image_processor):
        """Initialize all system components"""
        self.text_classifier = text_classifier
        self.image_classifier = image_classifier
        self.ocr_classifier = ocr_classifier
        self.fusion_classifier = fusion_classifier
        
        self.text_processor = text_processor
        self.image_processor = image_processor
        
        self.is_initialized = True
        logger.info("Fake news detection system initialized")
    
    def analyze_content(self, text: str = None, image=None) -> Dict:
        """Analyze content for fake news detection"""
        if not self.is_initialized:
            raise ValueError("System not initialized")
        
        results = {
            'text_analysis': {},
            'image_analysis': {},
            'final_prediction': {},
            'confidence_breakdown': {},
            'detailed_analysis': {}
        }
        
        try:
            # Text analysis
            if text and self.text_classifier:
                text_results = self._analyze_text(text)
                results['text_analysis'] = text_results
            
            # Image analysis
            if image is not None and (self.image_classifier or self.ocr_classifier):
                image_results = self._analyze_image(image)
                results['image_analysis'] = image_results
            
            # Fusion
            if results['text_analysis'] or results['image_analysis']:
                fusion_results = self._fuse_results(results['text_analysis'], results['image_analysis'])
                results['final_prediction'] = fusion_results
            
            # Generate detailed analysis
            results['detailed_analysis'] = self._generate_detailed_analysis(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            results['error'] = str(e)
            return results
    
    def _analyze_text(self, text: str) -> Dict:
        """Analyze text content"""
        try:
            # Preprocess text
            processed = self.text_processor.preprocess_batch([text], include_features=True)
            
            # Get predictions from text classifier
            predictions = self.text_classifier.predict([text])
            
            return {
                'predictions': predictions,
                'linguistic_features': processed.get('features_df', pd.DataFrame()).to_dict('records')[0] if not processed.get('features_df', pd.DataFrame()).empty else {},
                'processed_text': processed['processed_texts'][0]
            }
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {'error': str(e)}
    
    def _analyze_image(self, image) -> Dict:
        """Analyze image content"""
        try:
            # Process image
            processed = self.image_processor.process_image_batch([image])
            
            results = {}
            
            # Image manipulation detection
            if self.image_classifier and len(processed['preprocessed_images']) > 0:
                img_array = np.array([processed['preprocessed_images'][0]])
                manipulation_pred = self.image_classifier.predict(img_array)
                results['manipulation_detection'] = manipulation_pred
            
            # OCR analysis
            if self.ocr_classifier and len(processed['ocr_results']) > 0:
                ocr_pred = self.ocr_classifier.predict(processed['ocr_results'])
                results['ocr_analysis'] = ocr_pred
                results['extracted_text'] = processed['ocr_results'][0]
            
            results['image_features'] = processed['image_features'][0] if processed['image_features'] else {}
            
            return results
            
        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            return {'error': str(e)}
    
    def _fuse_results(self, text_results: Dict, image_results: Dict) -> Dict:
        """Fuse text and image analysis results"""
        try:
            # Prepare predictions for fusion
            text_pred = text_results.get('predictions', {})
            
            # Combine image predictions (manipulation + OCR)
            image_pred = {}
            if 'manipulation_detection' in image_results:
                manip_pred = image_results['manipulation_detection']
                image_pred = manip_pred
            
            if 'ocr_analysis' in image_results:
                ocr_pred = image_results['ocr_analysis']
                if image_pred:
                    # Average the predictions
                    fake_prob = (np.array(image_pred['fake_probability']) + 
                               np.array(ocr_pred['fake_probability'])) / 2
                    image_pred['fake_probability'] = fake_prob.tolist()
                else:
                    image_pred = ocr_pred
            
            # Perform fusion
            if text_pred and image_pred:
                fusion_result = self.fusion_classifier.fuse_predictions(text_pred, image_pred)
            elif text_pred:
                fusion_result = text_pred
                fusion_result['fusion_method'] = 'text_only'
            elif image_pred:
                fusion_result = image_pred
                fusion_result['fusion_method'] = 'image_only'
            else:
                fusion_result = {'error': 'No valid predictions available'}
            
            return fusion_result
            
        except Exception as e:
            logger.error(f"Error in result fusion: {e}")
            return {'error': str(e)}
    
    def _generate_detailed_analysis(self, results: Dict) -> Dict:
        """Generate detailed analysis report"""
        analysis = {
            'summary': {},
            'confidence_factors': [],
            'risk_indicators': [],
            'recommendations': []
        }
        
        try:
            final_pred = results.get('final_prediction', {})
            
            if 'predictions' in final_pred and len(final_pred['predictions']) > 0:
                is_fake = final_pred['predictions'][0]
                confidence = final_pred.get('confidence_scores', [0])[0]
                fake_prob = final_pred.get('fake_probability', [0])[0]
                
                analysis['summary'] = {
                    'classification': 'FAKE' if is_fake else 'REAL',
                    'confidence': confidence,
                    'fake_probability': fake_prob,
                    'real_probability': 1 - fake_prob
                }
                
                # Add confidence factors
                if results.get('text_analysis'):
                    analysis['confidence_factors'].append('Text analysis completed')
                
                if results.get('image_analysis'):
                    analysis['confidence_factors'].append('Image analysis completed')
                
                # Add risk indicators
                if fake_prob > 0.7:
                    analysis['risk_indicators'].append('High probability of fake content')
                elif fake_prob > 0.5:
                    analysis['risk_indicators'].append('Moderate probability of fake content')
                
                # Add recommendations
                if is_fake:
                    analysis['recommendations'].append('Verify information from multiple credible sources')
                    analysis['recommendations'].append('Check for official statements or corrections')
                else:
                    analysis['recommendations'].append('Content appears authentic but continue to verify important claims')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating detailed analysis: {e}")
            return {'error': str(e)}