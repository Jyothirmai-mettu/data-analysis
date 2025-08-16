# 🔥 Calories Burnt Prediction - AI Fitness Tracker

A beautiful and intelligent web application that predicts calories burnt during exercise using advanced machine learning models. Built with Flask, scikit-learn, and modern web technologies.

## ✨ Features

- **🤖 Multiple ML Models**: Random Forest, Gradient Boosting, and Linear Regression
- **🎯 High Accuracy**: Advanced feature scaling and preprocessing for optimal predictions
- **💻 Beautiful UI**: Modern, responsive design with smooth animations
- **📊 Real-time Metrics**: Live model performance comparison and evaluation
- **🔄 Model Retraining**: Ability to retrain models with updated data
- **📱 Mobile Friendly**: Responsive design that works on all devices

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. **Clone or download the project files**
   ```bash
   # Make sure you have these files in your directory:
   # - app.py
   # - templates/index.html
   # - requirements.txt
   # - X_train.csv (your dataset)
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## 📊 Dataset Structure

The system expects your `X_train.csv` file to contain the following features:

| Feature | Description | Range |
|---------|-------------|-------|
| User_ID | Unique user identifier | Any |
| Gender | Gender (1=male, 0=female) | 0 or 1 |
| Age | Age in years | 15-100 |
| Height | Height in centimeters | 100-250 |
| Weight | Weight in kilograms | 30-200 |
| Duration | Exercise duration in minutes | 1-180 |
| Heart_Rate | Heart rate in BPM | 60-200 |
| Body_Temp | Body temperature in °C | 35-42 |
| Calories | Target variable (calories burnt) | Any positive value |

## 🧠 Machine Learning Models

### 1. Random Forest Regressor
- **Best for**: High accuracy predictions
- **Features**: Handles non-linear relationships, robust to outliers
- **Use case**: Primary prediction model

### 2. Gradient Boosting Regressor
- **Best for**: Sequential learning and optimization
- **Features**: Reduces bias and variance, handles complex patterns
- **Use case**: Secondary prediction model

### 3. Linear Regression
- **Best for**: Interpretable results
- **Features**: Simple, fast, explainable
- **Use case**: Baseline comparison model

## 🎯 How to Use

### Making Predictions

1. **Fill in the form** with your exercise parameters:
   - Select your gender
   - Enter your age, height, and weight
   - Input exercise duration and heart rate
   - Provide your body temperature

2. **Click "Predict Calories"** to get instant results

3. **View results** showing:
   - Main prediction (from Random Forest)
   - Comparison across all models
   - Confidence in the prediction

### Understanding Results

- **Prediction Value**: The main calorie prediction in calories
- **Model Comparison**: Shows predictions from all three models
- **Performance Metrics**: R² score, RMSE, and MAE for each model

### Model Performance

- **R² Score**: Higher is better (0-1 scale)
- **RMSE**: Lower is better (Root Mean Square Error)
- **MAE**: Lower is better (Mean Absolute Error)

## 🔧 Advanced Features

### Retraining Models

Click the "Retrain Models" button to:
- Retrain all models with current data
- Update performance metrics
- Improve prediction accuracy

### API Endpoints

- `GET /`: Main application page
- `POST /predict`: Make predictions
- `POST /train`: Retrain models
- `GET /api/performance`: Get model performance metrics

## 📁 Project Structure

```
Calorie_burnt_prediction/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html        # Beautiful web interface
├── requirements.txt       # Python dependencies
├── X_train.csv           # Your training dataset
├── calories.csv          # Calories dataset
├── exercise.csv          # Exercise dataset
└── README.md            # This file
```

## 🛠️ Technical Details

### Backend (Flask)
- **Framework**: Flask web framework
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

### Frontend
- **HTML5**: Semantic markup
- **CSS3**: Modern styling with gradients and animations
- **JavaScript**: Async/await for API calls
- **Responsive Design**: Mobile-first approach

### Machine Learning Pipeline
1. **Data Loading**: Load and validate CSV data
2. **Preprocessing**: Handle missing values and scale features
3. **Model Training**: Train multiple regression models
4. **Evaluation**: Calculate performance metrics
5. **Prediction**: Make real-time predictions
6. **Model Persistence**: Save trained models

## 🚀 Deployment

### Local Development
```bash
python app.py
```

### Production Deployment
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

## 📈 Performance Optimization

- **Feature Scaling**: StandardScaler for optimal model performance
- **Model Selection**: Automatic selection of best performing model
- **Caching**: Models are trained once and reused
- **Async Processing**: Non-blocking prediction requests

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Format**: Ensure your CSV has the correct column names and data types

3. **Port Conflicts**: Change the port in `app.py` if 5000 is busy
   ```python
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

4. **Memory Issues**: For large datasets, consider reducing model complexity

### Performance Tips

- Use the Random Forest model for best accuracy
- Ensure your input data is within the expected ranges
- Retrain models periodically for better performance
- Monitor model performance metrics regularly

## 🤝 Contributing

Feel free to contribute to this project by:
- Improving the UI/UX
- Adding new machine learning models
- Optimizing performance
- Adding new features

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Flask and scikit-learn
- Beautiful UI inspired by modern design principles
- Dataset provided for calories prediction

---

**Ready to predict your calories? Start the application and begin your fitness journey! 🚀💪**
