#!/usr/bin/env python3
"""
Calories Prediction Web Application
Professional Theme + Subtle Fitness Background
"""

import http.server
import socketserver
import urllib.parse
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import pickle

# Global variables for model and scaler
model = None
scaler = None
model_performance = {}

class CaloriesPredictionHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            # Generate HTML content
            html_content = self.generate_html()
            self.wfile.write(html_content.encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                prediction = self.make_prediction(data)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                response = {
                    'success': True,
                    'prediction': round(prediction, 2),
                    'message': f'Predicted calories burnt: {round(prediction, 2)} calories'
                }
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'success': False, 'error': str(e)}
                self.wfile.write(json.dumps(response).encode())
        
        elif self.path == '/train':
            try:
                success = train_model()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                if success:
                    response = {
                        'success': True,
                        'message': 'Model trained successfully!',
                        'performance': model_performance
                    }
                else:
                    response = {'success': False, 'error': 'Failed to train model'}
                self.wfile.write(json.dumps(response).encode())
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'success': False, 'error': str(e)}
                self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.end_headers()
    
    def make_prediction(self, data):
        global model, scaler
        if model is None or scaler is None:
            raise Exception("Model not trained yet")
        
        gender = 1 if data['gender'] == 'male' else 0
        features = np.array([[gender, float(data['age']), float(data['height']),
                              float(data['weight']), float(data['duration']),
                              float(data['heart_rate']), float(data['body_temp'])]])
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return prediction
    
    def generate_html(self):
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Calories Burnt Prediction</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f0f4f8, #d9e4ec);
            min-height: 100vh;
            color: #333;
            position: relative;
        }
        body::before {
            content: "";
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: url('https://tse4.mm.bing.net/th/id/OIP.JTsah-euYtghDD7lOsDo6wHaE7?rs=1&pid=ImgDetMain&o=7&rm=3') 
            opacity: 0.08;
            z-index: 0;
        }
        .overlay {
            position: relative;
            z-index: 1;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1000px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 2.5rem;
            color: #0077b6;
        }
        .header p {
            font-size: 1.2rem;
            color: #444;
        }
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .card {
            background: #ffffffcc;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            backdrop-filter: blur(6px);
        }
        .card h2 { color: #0077b6; margin-bottom: 20px; }
        .form-group { margin-bottom: 15px; }
        .form-group label { color: #333; font-weight: bold; }
        .form-group input, .form-group select {
            width: 100%; padding: 10px; border: 1px solid #ccc;
            border-radius: 8px; background: #f9f9f9; color: #333;
        }
        .form-group input:focus, .form-group select:focus { border-color: #0077b6; outline: none; }
        .btn {
            background: linear-gradient(135deg, #00c6ff 0%, #0072ff 100%);
            color: white; border: none; padding: 12px 25px;
            border-radius: 8px; font-size: 16px; cursor: pointer; width: 100%;
        }
        .btn:hover { opacity: 0.9; }
        .results {
            background: #e6f7ff;
            color: #0077b6; border-radius: 12px; padding: 20px;
            margin-top: 20px; text-align: center; display: none;
        }
        .prediction-value { font-size: 2rem; font-weight: bold; }
        .error {
            background: #ffcccc; padding: 15px; margin-top: 20px;
            border-radius: 8px; display: none; text-align: center;
            color: #b00020;
        }
        .stats-section { grid-column: 1/-1; margin-top: 40px; }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr));
            gap: 20px; margin-top: 20px;
        }
        .stat-card {
            background: #fafafa;
            border-radius: 12px; padding: 20px; text-align: center;
            box-shadow: 0 3px 8px rgba(0,0,0,0.05);
        }
        .stat-value { font-size: 1.8rem; color: #0077b6; font-weight: bold; }
        .stat-label { font-size: 0.9rem; color: #555; }
        @media (max-width:768px){
            .main-content { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="overlay">
        <div class="container">
            <div class="header">
                <h1>🔥 Calories Burnt Predictor</h1>
                <p>AI-powered fitness tracking with machine learning</p>
            </div>
            <div class="main-content">
                <div class="card">
                    <h2>🏋️ Predict Calories</h2>
                    <form id="predictionForm">
                        <div class="form-group"><label>Gender</label>
                            <select id="gender" name="gender" required>
                                <option value="">Select Gender</option>
                                <option value="male">Male</option>
                                <option value="female">Female</option>
                            </select>
                        </div>
                        <div class="form-group"><label>Age (years)</label>
                            <input type="number" id="age" name="age" required>
                        </div>
                        <div class="form-group"><label>Height (cm)</label>
                            <input type="number" id="height" name="height" required>
                        </div>
                        <div class="form-group"><label>Weight (kg)</label>
                            <input type="number" id="weight" name="weight" required>
                        </div>
                        <div class="form-group"><label>Duration (minutes)</label>
                            <input type="number" id="duration" name="duration" required>
                        </div>
                        <div class="form-group"><label>Heart Rate (bpm)</label>
                            <input type="number" id="heart_rate" name="heart_rate" required>
                        </div>
                        <div class="form-group"><label>Body Temp (°C)</label>
                            <input type="number" id="body_temp" name="body_temp" required>
                        </div>
                        <button type="submit" class="btn">🚀 Predict Calories</button>
                    </form>
                    <div class="error" id="error"></div>
                    <div class="results" id="results">
                        <h3>📈 Prediction Results</h3>
                        <div class="prediction-value" id="predictionValue">0</div>
                        <p>calories burnt</p>
                    </div>
                </div>
            </div>
            <div class="card stats-section">
                <h2>📊 Dataset Overview</h2>
                <div class="stats-grid">
                    <div class="stat-card"><div class="stat-value">15,002</div><div class="stat-label">Records</div></div>
                    <div class="stat-card"><div class="stat-value">7</div><div class="stat-label">Features</div></div>
                    <div class="stat-card"><div class="stat-value">1</div><div class="stat-label">Model</div></div>
                    <div class="stat-card"><div class="stat-value">AI</div><div class="stat-label">Powered</div></div>
                </div>
            </div>
        </div>
    </div>
<script>
const form=document.getElementById('predictionForm');
const results=document.getElementById('results');
const errorBox=document.getElementById('error');
form.addEventListener('submit',async e=>{
 e.preventDefault();
 const data={
  gender:form.gender.value,
  age:form.age.value,
  height:form.height.value,
  weight:form.weight.value,
  duration:form.duration.value,
  heart_rate:form.heart_rate.value,
  body_temp:form.body_temp.value
 };
 try{
  const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  const result=await res.json();
  if(result.success){
    document.getElementById('predictionValue').textContent=result.prediction;
    results.style.display='block'; errorBox.style.display='none';
  } else { showError(result.error); }
 }catch(err){ showError('Network error'); }
});
function showError(msg){ errorBox.textContent=msg; errorBox.style.display='block'; results.style.display='none'; }
</script>
</body>
</html>"""

def load_and_prepare_data():
    try:
        df = pd.read_csv('X_train.csv')
        if 'Unnamed: 0' in df.columns: df = df.drop('Unnamed: 0', axis=1)
        if df['Gender'].dtype == 'object':
            df['Gender'] = df['Gender'].map({'male': 1, 'female': 0})
        X = df.drop(['User_ID','Calories'],axis=1)
        y = df['Calories']
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        global scaler; scaler=StandardScaler()
        return scaler.fit_transform(X_train), scaler.transform(X_test), y_train,y_test
    except Exception as e:
        print("Error loading data:",e); return None,None,None,None

def train_model():
    global model,model_performance
    X_train,X_test,y_train,y_test=load_and_prepare_data()
    if X_train is None: return False
    model=RandomForestRegressor(n_estimators=100,random_state=42)
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    mse=mean_squared_error(y_test,y_pred)
    rmse=np.sqrt(mse)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    model_performance={'MSE':round(mse,2),'RMSE':round(rmse,2),'MAE':round(mae,2),'R²':round(r2,3)}
    with open('model.pkl','wb') as f: pickle.dump(model,f)
    with open('scaler.pkl','wb') as f: pickle.dump(scaler,f)
    return True

def main():
    print("🚀 Starting Calories Prediction Web Application")
    if train_model(): print("✅ Model trained successfully!", model_performance)
    else: print("❌ Training failed"); return
    PORT=8000
    with socketserver.TCPServer(("",PORT),CaloriesPredictionHandler) as httpd:
        print(f"🌐 Running at http://localhost:{PORT}")
        try: httpd.serve_forever()
        except KeyboardInterrupt: print("🛑 Server stopped"); httpd.shutdown()

if __name__=="__main__":
    main()
