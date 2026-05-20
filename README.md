# 🌱 AI-Powered Crop Recommendation System and Market Advisory

An AI-driven decision support system that recommends the most suitable crops for farmers based on soil health, weather conditions, and market data.  
Built with **Streamlit** for an easy-to-use web and mobile-friendly interface.  

🚀 Live Demo: [AI-Powered Crop Recommendation](
https://ai-powered-crop-recommendation-yogita-1327.streamlit.app)

---


## ✨ Features

- 📊 **Soil & Weather Input** → Farmers can enter Nitrogen, Phosphorus, Potassium, pH, rainfall, temperature, and humidity values.
- 🌱 **Crop Recommendation** → Suggests the best crop for given soil and climate conditions.
- 💰 **Market Advisory** → Provides approximate market prices for the recommended crop.
- 🌐 **Multilingual Support** → English + Hindi (extendable to more languages in future).
- 📱 **Mobile Ready** → Works on phones; can be converted into an APK for Android.
- 🎨 **Custom Styling** → Agriculture-themed UI with CSS support.

---

## 🏗️ Project Structure

AI-Crop-Recommendation/
│
├── 📂 data/
│   ├── crop_data.csv            # Dataset used for training the model
│   ├── market_prices.csv        # Approx market price dataset
│   └── soil_samples.csv         # Sample soil data (optional)
│
├── 📂 models/
│   ├── crop_model.pkl           # Trained ML model for crop prediction
│   └── scaler.pkl               # StandardScaler / preprocessing model
│
├── 📂 app/
│   ├── main.py                  # Streamlit main app file
│   ├── utils.py                 # Helper functions for prediction & advisory
│   ├── ui_components.py         # Reusable UI widgets, cards, styles
│   ├── language_support.py      # Multilingual translations (English/Hindi)
│   └── market_advisory.py       # Logic for suggesting market price
│
├── 📂 assets/
│   ├── styles.css               # Custom CSS for UI theming
│   ├── banner.png               # App banner / hero image
│   └── icons/                   # Icons for crops, weather, UI elements
│
├── 📂 deployment/
│   ├── requirements.txt         # Python dependencies
│   ├── Procfile                 # For deployment on Streamlit Cloud/Heroku
│   └── Dockerfile               # (Optional) Containerization support
│
├── README.md                    # Project documentation  
├── LICENSE                      # Project license (MIT/GPL etc.)
└── .gitignore                   # Ignore unnecessary files in GitHub

🌦️ Real-Time Integrations

Live Weather API Integration (OpenWeather / Weatherbit) → Auto-fetch temperature, humidity, rainfall instead of manual input.

Market Price API Integration → Can be extended to fetch real mandi prices from government sources (Agmarknet API).

🧩 Smart Farming Add-Ons

Fertilizer Recommendation System → Suggests nutrient requirements tailored to the recommended crop.

Pest & Disease Alert Module (future-ready) → Placeholder for integrating image-based detection using CNNs.

Water Requirement Calculator → Shows weekly irrigation needs for the suggested crop.
