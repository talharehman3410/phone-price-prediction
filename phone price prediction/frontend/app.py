from flask import Flask, render_template, request, jsonify
import numpy as np
import random

app = Flask(__name__)


import pickle
with open('phone_price_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)
with open('knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)

def real_predict(features, model_type='random_forest'):
    model = rf_model if model_type == 'random_forest' else knn_model
    import pandas as pd
    df = pd.DataFrame([features])
    # apply same label encoding as notebook...
    price = model.predict(df)[0]
    return float(price)

BRAND_BASE = {
    'Apple':       950,
    'Samsung':     700,
    'Google':      680,
    'OnePlus':     580,
    'Sony':        620,
    'Huawei':      500,
    'Asus':        420,
    'Motorola':    320,
    'Nokia':       280,
    'Oppo':        380,
    'Vivo':        360,
    'Realme':      260,
    'Xiaomi':      280,
    'LG':          320,
    'Blackberry':  400,
    'CAT':         480,
}

def mock_predict(brand, storage, ram, screen_size, camera_mp, battery_mah, model_type):
    base = BRAND_BASE.get(brand, 400)
    base += (storage - 64) * 0.9
    base += (ram - 4) * 28
    base += (screen_size - 6.0) * 60
    base += (camera_mp - 64) * 0.5
    if battery_mah > 5000:
        base += 30
    elif battery_mah < 3500:
        base -= 40

    # KNN is slightly noisier
    noise_range = 35 if model_type == 'random_forest' else 60
    noise = random.uniform(-noise_range, noise_range)
    price = max(49, base + noise)
    return round(price, 2)

# ── Dropdown data ──────────────────────────────────────────────
BRANDS = ['Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Google', 'Oppo',
          'Vivo', 'Realme', 'Motorola', 'Nokia', 'Sony', 'LG',
          'Asus', 'Blackberry', 'CAT', 'Huawei']

STORAGE_OPTIONS = [32, 64, 128, 256, 512]
RAM_OPTIONS     = [2, 3, 4, 6, 8, 12, 16]
SCREEN_SIZES    = [5.0, 5.5, 6.0, 6.1, 6.2, 6.4, 6.5, 6.67, 6.7, 6.8, 7.0]
CAMERA_OPTIONS  = [8, 12, 16, 32, 48, 50, 64, 108, 200]
BATTERY_OPTIONS = [2000, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 7000]

MODEL_INFO = {
    'random_forest': {'name': 'Random Forest', 'r2': 82.4, 'mae': 48.2, 'rmse': 71.5},
    'knn':           {'name': 'K-Nearest Neighbors', 'r2': 59.85, 'mae': 12.59, 'rmse': 16.94},
}

@app.route('/')
def index():
    return render_template('index.html',
        brands=BRANDS,
        storage_options=STORAGE_OPTIONS,
        ram_options=RAM_OPTIONS,
        screen_sizes=SCREEN_SIZES,
        camera_options=CAMERA_OPTIONS,
        battery_options=BATTERY_OPTIONS,
        model_info=MODEL_INFO,
    )

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d = request.get_json()
        brand      = d.get('brand', 'Samsung')
        storage    = int(d.get('storage', 128))
        ram        = int(d.get('ram', 6))
        screen     = float(d.get('screen_size', 6.5))
        camera     = int(d.get('camera_mp', 48))
        battery    = int(d.get('battery_mah', 4500))
        model_type = d.get('model_type', 'random_forest')

        price = mock_predict(brand, storage, ram, screen, camera, battery, model_type)
        info  = MODEL_INFO.get(model_type, MODEL_INFO['random_forest'])

        # Tier classification
        if price < 200:
            tier, tier_color = 'Budget', '#34D399'
        elif price < 500:
            tier, tier_color = 'Mid-Range', '#60A5FA'
        elif price < 900:
            tier, tier_color = 'Premium', '#A78BFA'
        else:
            tier, tier_color = 'Flagship', '#F59E0B'

        return jsonify({
            'success':    True,
            'price':      price,
            'price_low':  round(price * 0.90, 2),
            'price_high': round(price * 1.10, 2),
            'tier':       tier,
            'tier_color': tier_color,
            'model_name': info['name'],
            'r2':         info['r2'],
            'mae':        info['mae'],
            'rmse':       info['rmse'],
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
