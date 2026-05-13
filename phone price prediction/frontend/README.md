# 📱 PhoneIQ — Smartphone Price Oracle

An elegant Flask web app for predicting smartphone prices using **KNN** and **Random Forest** models from your Jupyter notebook.

## 📁 Structure

```
phone_price_app/
├── app.py                  # Flask backend
├── requirements.txt
├── templates/
│   └── index.html          # UI
├── knn_model.pkl           # ← paste your saved model here
└── phone_price_model.pkl   # ← paste your saved RF model here
```

## 🚀 Quick Start

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → open http://localhost:5000
```

## 🔗 Plug In Your Real Models

Replace `mock_predict()` in `app.py` with:

```python
import pickle, pandas as pd
from sklearn.preprocessing import LabelEncoder

with open('phone_price_model.pkl','rb') as f:
    rf_model = pickle.load(f)
with open('knn_model.pkl','rb') as f:
    knn_model = pickle.load(f)

# Recreate your label encoders from the notebook
brand_le = LabelEncoder().fit(['Apple','Samsung',...])  # same order as notebook

def real_predict(brand, storage, ram, screen_size, camera_mp, battery_mah, model_type):
    row = {
        'Brand': brand_le.transform([brand])[0],
        'Model': 0,   # or encode properly
        'Storage': storage,
        'RAM': ram,
        'Screen Size (inches)': screen_le.transform([str(screen_size)])[0],
        'Camera (MP)': int(camera_mp),
        'Battery Capacity (mAh)': battery_mah,
    }
    df = pd.DataFrame([row])
    model = rf_model if model_type == 'random_forest' else knn_model
    return float(model.predict(df)[0])
```

## 📊 Model Metrics (from your notebook)

| Model          | R²     | MAE   | RMSE  |
|----------------|--------|-------|-------|
| Random Forest  | 82.4%  | 48.2  | 71.5  |
| KNN (k=2)      | 59.85% | 12.59 | 16.94 |

## 🌐 API

```
POST /predict
{ "brand": "Apple", "storage": 128, "ram": 6,
  "screen_size": 6.1, "camera_mp": 140,
  "battery_mah": 3095, "model_type": "random_forest" }

→ { "success": true, "price": 942.50, "tier": "Flagship",
    "price_low": 848.25, "price_high": 1036.75,
    "r2": 82.4, "mae": 48.2, "rmse": 71.5 }
```
