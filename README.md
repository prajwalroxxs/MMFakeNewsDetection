# Multi-Modal Fake News Detection System

A comprehensive deep learning research project that combines Natural Language Processing (NLP) and Computer Vision to detect fake news content. This system analyzes both text and images to provide robust classification with confidence scores and detailed analysis reports.

## 🚀 Features

### Multi-Modal Analysis
- **Text Analysis**: BERT-based and feature-based classification
- **Image Analysis**: 
  - Image manipulation detection using Error Level Analysis
  - OCR text extraction and analysis
- **Fusion System**: Combines predictions from all modalities

### Comprehensive Output
- Binary classification (Fake/Real)
- Confidence scores for each prediction
- Detailed analysis reports
- Feature importance insights
- Risk indicators and recommendations

### Research Components
- Jupyter notebook for experimentation
- API endpoints for real-time analysis
- Visualization tools for results analysis
- Performance metrics and evaluation

## 🏗️ Architecture

```
├── ml_components/
│   ├── utils/
│   │   └── data_utils.py          # Dataset loading and management
│   ├── preprocessing/
│   │   ├── text_processor.py      # Text preprocessing and feature extraction
│   │   └── image_processor.py     # Image preprocessing and OCR
│   ├── models/
│   │   ├── nlp/
│   │   │   └── text_classifier.py # BERT and feature-based classifiers
│   │   ├── computer_vision/
│   │   │   └── image_classifier.py # Image manipulation and OCR classifiers
│   │   └── fusion/
│   │       └── multimodal_classifier.py # Multi-modal fusion system
│   └── notebooks/
│       └── fake_news_detection_research.ipynb # Research notebook
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python
- **Machine Learning**: TensorFlow, scikit-learn, transformers
- **Computer Vision**: OpenCV, PIL, pytesseract
- **NLP**: BERT, NLTK, TextBlob
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Database**: MongoDB

## 📊 Model Components

### 1. NLP Component
- **BERT-based Classifier**: Uses pre-trained BERT for semantic understanding
- **Feature-based Classifier**: Extracts linguistic features like:
  - Sentiment polarity and subjectivity
  - Text length and complexity metrics
  - Punctuation patterns
  - Readability scores

### 2. Computer Vision Components

#### Image Manipulation Detection
- Error Level Analysis (ELA) for detecting image tampering
- CNN-based classification for manipulation detection
- Statistical analysis of compression artifacts

#### OCR Text Analysis
- Text extraction from images using pytesseract
- Analysis of extracted text for suspicious patterns
- Confidence scoring based on OCR quality

### 3. Multi-Modal Fusion
- **Weighted Average**: Combines predictions with configurable weights
- **Voting**: Majority voting across modalities
- **Meta-learner**: Trained classifier for optimal fusion

## 🚦 API Endpoints

### Core Analysis
- `POST /api/analyze` - Complete multi-modal analysis
- `POST /api/analyze/text-only` - Text-only analysis
- `POST /api/analyze/image-only` - Image-only analysis

### System Management
- `GET /api/system/status` - Check system status
- `POST /api/system/initialize` - Initialize/reinitialize ML system

### History and Monitoring
- `GET /api/analyze/history` - Get analysis history
- `GET /api/status` - System health checks

## 📚 Research Notebook

The Jupyter notebook (`fake_news_detection_research.ipynb`) provides:

1. **Data Exploration**: Dataset statistics and visualization
2. **Text Analysis**: Preprocessing and feature extraction
3. **Model Training**: Training and evaluation of all components
4. **Performance Analysis**: Comprehensive metrics and visualizations
5. **Research Insights**: Conclusions and future improvements

## 🎯 Usage Examples

### Text Analysis
```python
# Analyze suspicious text
text = "SHOCKING! This one weird trick will make you rich overnight!"
result = await analyze_text_only(text)
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### Multi-Modal Analysis
```python
# Analyze text and image together
request = {
    "text": "Breaking news about amazing discovery!",
    "image_base64": "data:image/png;base64,..."
}
result = await analyze_fake_news(request)
print(f"Final Prediction: {result['final_prediction']}")
```

## 📈 Performance Metrics

The system provides comprehensive evaluation metrics:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall**: Class-specific performance
- **F1-Score**: Balanced performance measure
- **AUC-ROC**: Probability calibration quality
- **Confidence Calibration**: Reliability of confidence scores

## 🔬 Research Applications

- **Social Media Monitoring**: Real-time fake news detection
- **News Verification**: Automated fact-checking assistance
- **Educational Tools**: Media literacy and critical thinking
- **Content Moderation**: Platform safety and trust
- **Research**: Misinformation pattern analysis

## 🚀 Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API Server**:
   ```bash
   sudo supervisorctl restart backend
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   cd /app/ml_components/notebooks
   jupyter notebook fake_news_detection_research.ipynb
   ```

4. **Test the API**:
   ```bash
   curl -X POST "http://localhost:8001/api/analyze/text-only" \
        -H "Content-Type: application/json" \
        -d '{"text": "Your text here"}'
   ```

## 📊 Dataset Sources

The system uses publicly available datasets:
- **LIAR Dataset**: Political statements with truth ratings
- **FakeNewsNet**: Multi-modal fake news dataset
- **Custom Samples**: Synthetic data for demonstration

## 🔧 Configuration

### Model Weights (Fusion)
- Text Weight: 0.6 (default)
- Image Weight: 0.4 (default)
- Configurable via API

### Thresholds
- Classification Threshold: 0.5
- Confidence Threshold: Configurable
- OCR Confidence Minimum: 80%

## 🎓 Research Insights

### Key Findings
1. **Multi-modal Superior**: Combined analysis outperforms single-modal approaches
2. **Linguistic Features**: Sentiment and readability are strong indicators
3. **Image Text**: OCR-extracted text provides valuable signals
4. **Fusion Benefits**: Weighted fusion provides best balance

### Future Improvements
- Advanced image manipulation detection
- Real-time streaming analysis
- External fact-checking integration
- Explainable AI components
- Cross-platform deployment

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@misc{multimodal_fake_news_detection,
  title={Multi-Modal Fake News Detection: Combining NLP and Computer Vision},
  author={Research Team},
  year={2025},
  note={Deep Learning Research Project}
}
```

## 🤝 Contributing

This is a research project demonstrating multi-modal approaches to fake news detection. Contributions and improvements are welcome!

## 📄 License

This project is intended for research and educational purposes.

---

**Note**: This system is designed for research purposes and should be used as part of a comprehensive fact-checking workflow, not as a standalone truth determination tool.
