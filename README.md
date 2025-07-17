README.md
==========

Heart-Disease Risk Prediction  
--------------------------------

This mini–project contains two main scripts:

1. **`train_logistic.py`** &nbsp;– trains a Logistic-Regression model and saves the fitted preprocessing + model pipeline.  
2. **`heart_disease_app.py`** &nbsp;– a Streamlit web UI that loads the pipeline and predicts the risk of heart disease for user-entered patient data.

Folder structure after cloning / downloading
```
├── Heart Prediction Quantum Dataset.csv      # source data  (must be in the root folder)
├── train_logistic.py                         # model-training script
├── heart_disease_app.py                      # Streamlit UI
├── requirements.txt                          # Python dependencies
└── output/
      └── logreg_pipeline.pkl                 # created by train_logistic.py
```

Prerequisites
-------------
• Python ≥ 3.9  
• Internet connection to install PyPI packages

(Optional but recommended) create and activate a virtual environment:
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

1. Install dependencies
-----------------------
```bash
pip install -r requirements.txt
```

2. Train the model
------------------
```bash
python train_model.py
```
What it does  
• Reads `Heart Prediction Quantum Dataset.csv`  
• Fits a logistic-regression pipeline (with standard-scaling) on **all** available rows  
• Creates `output/` (if it doesn’t exist) and writes `model_name.pkl`

You should see something like:
```
✅ All models trained and saved to ./output/
```

3. Launch the web application
-----------------------------
```bash
streamlit run heart_disease_app.py
```
A browser tab will open at `http://localhost:8501` showing the Heart-Disease Risk Predictor UI.  
Enter the six features (Age, Gender, Blood Pressure, etc.) and press **Predict** to obtain:

• Probability of heart disease  
• Predicted class (1 = disease, 0 = no disease)  
• A progress bar visualization of the probability

Troubleshooting
---------------
• “Model file not found” – Make sure you ran `train_model.py` before starting the app, and that `output/model_name.pkl` exists.  
• Port already in use – Start Streamlit on another port:  
  ```bash
  streamlit run heart_disease_app.py --server.port 8502
  ```  
• To exit the virtual environment: `deactivate`

Have fun predicting!