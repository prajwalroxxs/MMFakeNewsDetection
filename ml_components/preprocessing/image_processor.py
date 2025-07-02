"""
Image preprocessing utilities for fake news detection
"""
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import pytesseract
from typing import Dict, List, Tuple, Optional
import base64
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Handles image preprocessing for fake news detection"""
    
    def __init__(self):
        self.setup_tesseract()
    
    def setup_tesseract(self):
        """Setup tesseract for OCR"""
        try:
            # Try to configure tesseract path if needed
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR is available")
        except Exception as e:
            logger.warning(f"Tesseract OCR setup issue: {e}")
    
    def load_image(self, image_input) -> Optional[np.ndarray]:
        """Load image from various input formats"""
        try:
            if isinstance(image_input, str):
                if image_input.startswith('data:image'):
                    # Handle base64 encoded image
                    return self.base64_to_image(image_input)
                elif os.path.exists(image_input):
                    # Handle file path
                    return cv2.imread(image_input)
                else:
                    logger.error(f"Image file not found: {image_input}")
                    return None
            elif isinstance(image_input, np.ndarray):
                return image_input
            else:
                logger.error(f"Unsupported image input type: {type(image_input)}")
                return None
        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
    
    def base64_to_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to PIL Image
            pil_image = Image.open(io.BytesIO(image_data))
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return opencv_image
        except Exception as e:
            logger.error(f"Error converting base64 to image: {e}")
            return None
    
    def image_to_base64(self, image: np.ndarray) -> str:
        """Convert OpenCV image to base64 string"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(rgb_image)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            base64_string = base64.b64encode(buffer.getvalue()).decode()
            
            return f"data:image/png;base64,{base64_string}"
        except Exception as e:
            logger.error(f"Error converting image to base64: {e}")
            return ""
    
    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for CNN models"""
        try:
            # Resize image
            resized = cv2.resize(image, target_size)
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype(np.float32) / 255.0
            
            return normalized
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return np.zeros((target_size[0], target_size[1], 3))
    
    def extract_text_from_image(self, image: np.ndarray) -> Dict:
        """Extract text from image using OCR"""
        try:
            # Convert to PIL Image for pytesseract
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Extract text
            extracted_text = pytesseract.image_to_string(pil_image)
            
            # Get detailed information
            try:
                data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
                confidence_scores = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            except:
                avg_confidence = 0
            
            return {
                'text': extracted_text.strip(),
                'confidence': avg_confidence,
                'word_count': len(extracted_text.split()),
                'has_text': len(extracted_text.strip()) > 0
            }
        
        except Exception as e:
            logger.error(f"Error extracting text from image: {e}")
            return {
                'text': '',
                'confidence': 0,
                'word_count': 0,
                'has_text': False
            }
    
    def detect_image_manipulation(self, image: np.ndarray) -> Dict:
        """Detect potential image manipulation using Error Level Analysis (ELA)"""
        try:
            # Convert to PIL Image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
            
            # Save image with high quality
            buffer1 = io.BytesIO()
            pil_image.save(buffer1, format='JPEG', quality=95)
            
            # Reload and save with lower quality
            buffer1.seek(0)
            img_95 = Image.open(buffer1)
            buffer2 = io.BytesIO()
            img_95.save(buffer2, format='JPEG', quality=75)
            
            # Reload the lower quality image
            buffer2.seek(0)
            img_75 = Image.open(buffer2)
            
            # Calculate difference (ELA)
            diff = np.array(img_95) - np.array(img_75)
            diff = np.abs(diff)
            
            # Calculate metrics
            mean_diff = np.mean(diff)
            max_diff = np.max(diff)
            std_diff = np.std(diff)
            
            # Simple manipulation detection based on ELA
            manipulation_score = (mean_diff + std_diff) / 255.0  # Normalize to [0, 1]
            
            # Threshold for potential manipulation (this is a simple heuristic)
            is_potentially_manipulated = manipulation_score > 0.1
            
            return {
                'manipulation_score': manipulation_score,
                'is_potentially_manipulated': is_potentially_manipulated,
                'mean_difference': mean_diff,
                'max_difference': max_diff,
                'std_difference': std_diff,
                'analysis_method': 'Error Level Analysis (ELA)'
            }
        
        except Exception as e:
            logger.error(f"Error in manipulation detection: {e}")
            return {
                'manipulation_score': 0.0,
                'is_potentially_manipulated': False,
                'mean_difference': 0.0,
                'max_difference': 0.0,
                'std_difference': 0.0,
                'analysis_method': 'Error Level Analysis (ELA)',
                'error': str(e)
            }
    
    def extract_image_features(self, image: np.ndarray) -> Dict:
        """Extract various features from image"""
        try:
            features = {}
            
            # Basic image properties
            features['height'], features['width'] = image.shape[:2]
            features['channels'] = image.shape[2] if len(image.shape) > 2 else 1
            features['aspect_ratio'] = features['width'] / features['height']
            
            # Color analysis
            if len(image.shape) == 3:
                # Color statistics
                features['mean_red'] = np.mean(image[:, :, 2])  # OpenCV uses BGR
                features['mean_green'] = np.mean(image[:, :, 1])
                features['mean_blue'] = np.mean(image[:, :, 0])
                
                features['std_red'] = np.std(image[:, :, 2])
                features['std_green'] = np.std(image[:, :, 1])
                features['std_blue'] = np.std(image[:, :, 0])
                
                # Color diversity
                unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
                total_pixels = features['height'] * features['width']
                features['color_diversity'] = unique_colors / total_pixels
            
            # Texture analysis using Laplacian variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            features['sharpness'] = laplacian_var
            
            # Brightness and contrast
            features['brightness'] = np.mean(gray)
            features['contrast'] = np.std(gray)
            
            return features
        
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def process_image_batch(self, images: List) -> Dict:
        """Process a batch of images"""
        results = {
            'preprocessed_images': [],
            'ocr_results': [],
            'manipulation_analysis': [],
            'image_features': [],
            'base64_images': []
        }
        
        for image_input in images:
            # Load image
            image = self.load_image(image_input)
            
            if image is not None:
                # Preprocess for ML models
                preprocessed = self.preprocess_image(image)
                results['preprocessed_images'].append(preprocessed)
                
                # OCR text extraction
                ocr_result = self.extract_text_from_image(image)
                results['ocr_results'].append(ocr_result)
                
                # Manipulation analysis
                manipulation_result = self.detect_image_manipulation(image)
                results['manipulation_analysis'].append(manipulation_result)
                
                # Extract features
                features = self.extract_image_features(image)
                results['image_features'].append(features)
                
                # Convert to base64 for storage/transmission
                base64_img = self.image_to_base64(image)
                results['base64_images'].append(base64_img)
            else:
                # Handle failed image loading
                results['preprocessed_images'].append(np.zeros((224, 224, 3)))
                results['ocr_results'].append({'text': '', 'confidence': 0, 'word_count': 0, 'has_text': False})
                results['manipulation_analysis'].append({'manipulation_score': 0.0, 'is_potentially_manipulated': False})
                results['image_features'].append({})
                results['base64_images'].append('')
        
        return results