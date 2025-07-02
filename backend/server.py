from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
import sys
import numpy as np
import base64
import io
from PIL import Image
import cv2

# Add ML components to path
sys.path.append('/app')

# Import ML components
from ml_components.utils.data_utils import DataLoader
from ml_components.preprocessing.text_processor import TextPreprocessor
from ml_components.preprocessing.image_processor import ImagePreprocessor
from ml_components.models.nlp.text_classifier import FeatureBasedClassifier
from ml_components.models.computer_vision.image_classifier import OCRFeatureClassifier
from ml_components.models.fusion.multimodal_classifier import MultiModalFusionClassifier, FakeNewsDetectionSystem


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class FakeNewsAnalysisRequest(BaseModel):
    text: Optional[str] = None
    image_base64: Optional[str] = None

class FakeNewsAnalysisResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text_analysis: Dict[str, Any] = {}
    image_analysis: Dict[str, Any] = {}
    final_prediction: Dict[str, Any] = {}
    detailed_analysis: Dict[str, Any] = {}
    confidence_breakdown: Dict[str, Any] = {}
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# Initialize ML System
ml_system = None
text_processor = None
image_processor = None

def initialize_ml_system():
    """Initialize the ML system components"""
    global ml_system, text_processor, image_processor
    
    try:
        logger.info("Initializing ML components...")
        
        # Initialize preprocessors
        text_processor = TextPreprocessor()
        image_processor = ImagePreprocessor()
        
        # Initialize classifiers
        feature_classifier = FeatureBasedClassifier(classifier_type='random_forest')
        ocr_classifier = OCRFeatureClassifier()
        fusion_classifier = MultiModalFusionClassifier(fusion_method='weighted_average')
        
        # Create sample training data for demo purposes
        sample_train_data()
        
        # Initialize the complete system
        ml_system = FakeNewsDetectionSystem()
        ml_system.initialize_components(
            text_classifier=feature_classifier,
            image_classifier=None,
            ocr_classifier=ocr_classifier,
            fusion_classifier=fusion_classifier,
            text_processor=text_processor,
            image_processor=image_processor
        )
        
        logger.info("ML system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing ML system: {e}")
        return False

def sample_train_data():
    """Create and train with sample data for demonstration"""
    global text_processor, image_processor
    
    try:
        # Sample text data
        fake_texts = [
            "SHOCKING! You won't believe this incredible discovery that doctors don't want you to know!",
            "BREAKING: Miracle cure found! Click here to learn the secret!",
            "Scientists HATE this one simple trick that will change your life forever!",
            "URGENT: Government conspiracy exposed! Share before it's deleted!",
            "AMAZING results with this weird method! Doctors are speechless!"
        ]
        
        real_texts = [
            "The weather forecast shows sunny conditions with temperatures reaching 75 degrees.",
            "Local university researchers publish findings in peer-reviewed journal.",
            "City council announces new infrastructure development plans for downtown area.",
            "Stock market shows moderate gains following quarterly earnings reports.",
            "New study reveals importance of regular exercise for cardiovascular health."
        ]
        
        all_texts = fake_texts + real_texts
        all_labels = [1] * len(fake_texts) + [0] * len(real_texts)
        
        # Train text classifier
        processed = text_processor.preprocess_batch(all_texts, include_features=True)
        features_df = processed['features_df']
        
        feature_classifier = FeatureBasedClassifier(classifier_type='random_forest')
        feature_classifier.train(features_df, all_labels)
        
        # Sample OCR data
        fake_ocr_results = [
            {'text': 'SHOCKING NEWS YOU WONT BELIEVE!!!', 'confidence': 85, 'word_count': 6, 'has_text': True},
            {'text': 'BREAKING: MIRACLE CURE DISCOVERED!', 'confidence': 90, 'word_count': 5, 'has_text': True},
            {'text': 'DOCTORS HATE THIS ONE SIMPLE TRICK', 'confidence': 88, 'word_count': 7, 'has_text': True}
        ]
        
        real_ocr_results = [
            {'text': 'Weather forecast for today', 'confidence': 95, 'word_count': 5, 'has_text': True},
            {'text': 'Local news update', 'confidence': 98, 'word_count': 4, 'has_text': True},
            {'text': 'Scientific research findings', 'confidence': 96, 'word_count': 4, 'has_text': True}
        ]
        
        all_ocr_results = fake_ocr_results + real_ocr_results
        all_ocr_labels = [1] * len(fake_ocr_results) + [0] * len(real_ocr_results)
        
        # Train OCR classifier
        ocr_classifier = OCRFeatureClassifier()
        ocr_classifier.train(all_ocr_results, all_ocr_labels)
        
        # Store trained models globally
        globals()['trained_feature_classifier'] = feature_classifier
        globals()['trained_ocr_classifier'] = ocr_classifier
        
        logger.info("Sample training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in sample training: {e}")

def get_trained_models():
    """Get the trained models"""
    try:
        feature_classifier = globals().get('trained_feature_classifier')
        ocr_classifier = globals().get('trained_ocr_classifier')
        
        if feature_classifier and ocr_classifier:
            return feature_classifier, ocr_classifier
        else:
            # Return default models if training failed
            return FeatureBasedClassifier(), OCRFeatureClassifier()
    except:
        return FeatureBasedClassifier(), OCRFeatureClassifier()

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Multi-Modal Fake News Detection API", "status": "active"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

@api_router.post("/analyze", response_model=FakeNewsAnalysisResponse)
async def analyze_fake_news(request: FakeNewsAnalysisRequest):
    """Analyze content for fake news detection"""
    
    if not request.text and not request.image_base64:
        raise HTTPException(status_code=400, detail="Either text or image must be provided")
    
    try:
        # Get trained models
        feature_classifier, ocr_classifier = get_trained_models()
        fusion_classifier = MultiModalFusionClassifier(fusion_method='weighted_average')
        
        # Initialize system with trained models
        analysis_system = FakeNewsDetectionSystem()
        analysis_system.initialize_components(
            text_classifier=feature_classifier,
            image_classifier=None,
            ocr_classifier=ocr_classifier,
            fusion_classifier=fusion_classifier,
            text_processor=text_processor,
            image_processor=image_processor
        )
        
        # Prepare image if provided
        image = None
        if request.image_base64:
            try:
                # Convert base64 to image
                if ',' in request.image_base64:
                    image_data = request.image_base64.split(',')[1]
                else:
                    image_data = request.image_base64
                
                image_bytes = base64.b64decode(image_data)
                pil_image = Image.open(io.BytesIO(image_bytes))
                image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Perform analysis
        analysis_result = analysis_system.analyze_content(
            text=request.text,
            image=image
        )
        
        # Create response
        response = FakeNewsAnalysisResponse(
            text_analysis=analysis_result.get('text_analysis', {}),
            image_analysis=analysis_result.get('image_analysis', {}),
            final_prediction=analysis_result.get('final_prediction', {}),
            detailed_analysis=analysis_result.get('detailed_analysis', {}),
            confidence_breakdown=analysis_result.get('confidence_breakdown', {})
        )
        
        # Save analysis to database
        analysis_dict = response.dict()
        await db.fake_news_analyses.insert_one(analysis_dict)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fake news analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/analyze/history")
async def get_analysis_history(limit: int = 50):
    """Get recent analysis history"""
    try:
        analyses = await db.fake_news_analyses.find().sort("timestamp", -1).limit(limit).to_list(limit)
        return analyses
    except Exception as e:
        logger.error(f"Error fetching analysis history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history")

@api_router.get("/system/status")
async def get_system_status():
    """Get ML system status"""
    global ml_system, text_processor, image_processor
    
    status = {
        "ml_system_initialized": ml_system is not None and ml_system.is_initialized,
        "text_processor_ready": text_processor is not None,
        "image_processor_ready": image_processor is not None,
        "models_available": {
            "feature_classifier": 'trained_feature_classifier' in globals(),
            "ocr_classifier": 'trained_ocr_classifier' in globals()
        }
    }
    
    return status

@api_router.post("/system/initialize")
async def initialize_system():
    """Initialize or reinitialize the ML system"""
    try:
        success = initialize_ml_system()
        if success:
            return {"message": "ML system initialized successfully", "status": "ready"}
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize ML system")
    except Exception as e:
        logger.error(f"Error initializing system: {e}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

@api_router.post("/analyze/text-only")
async def analyze_text_only(text: str):
    """Analyze text content only"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text content is required")
    
    try:
        feature_classifier, _ = get_trained_models()
        
        # Process text
        processed = text_processor.preprocess_batch([text], include_features=True)
        features_df = processed['features_df']
        
        # Make prediction
        prediction = feature_classifier.predict(features_df)
        
        # Extract linguistic features
        linguistic_features = processed['features_df'].to_dict('records')[0] if not processed['features_df'].empty else {}
        
        result = {
            "text": text,
            "processed_text": processed['processed_texts'][0],
            "prediction": "FAKE" if prediction['predictions'][0] else "REAL",
            "fake_probability": prediction['fake_probability'][0],
            "confidence": prediction['confidence_scores'][0],
            "linguistic_features": linguistic_features
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in text-only analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Text analysis failed: {str(e)}")

@api_router.post("/analyze/image-only")
async def analyze_image_only(file: UploadFile = File(...)):
    """Analyze image content only"""
    try:
        # Read image file
        image_data = await file.read()
        
        # Convert to OpenCV image
        pil_image = Image.open(io.BytesIO(image_data))
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Process image
        processed = image_processor.process_image_batch([image])
        
        # Get OCR results
        ocr_result = processed['ocr_results'][0] if processed['ocr_results'] else {}
        manipulation_result = processed['manipulation_analysis'][0] if processed['manipulation_analysis'] else {}
        image_features = processed['image_features'][0] if processed['image_features'] else {}
        
        # Analyze with OCR classifier if text is found
        ocr_prediction = {}
        if ocr_result.get('has_text', False):
            _, ocr_classifier = get_trained_models()
            ocr_prediction = ocr_classifier.predict([ocr_result])
        
        result = {
            "filename": file.filename,
            "ocr_results": ocr_result,
            "manipulation_analysis": manipulation_result,
            "image_features": image_features,
            "ocr_prediction": ocr_prediction
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in image-only analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Image analysis failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize ML system on startup"""
    logger.info("Starting up the application...")
    success = initialize_ml_system()
    if success:
        logger.info("ML system initialized successfully on startup")
    else:
        logger.warning("ML system initialization failed on startup")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
