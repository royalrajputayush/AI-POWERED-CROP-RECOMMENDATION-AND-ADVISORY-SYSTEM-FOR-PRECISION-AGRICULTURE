import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Set page configurations
st.set_page_config(
    page_title="AGRIMIND AI-Powered Crop Recommendation & Advisory",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Load Local CSS ----------------
def local_css(file_name):
    """Load a local CSS file into Streamlit."""
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ CSS file not found. Using default styling.")

# Apply custom CSS
local_css("bpp_style.css")

def get_dataset_mtime():
    path1 = "data/crop_recommendation.csv"
    path2 = "crop_recommendation.csv"
    if os.path.exists(path1):
        return os.path.getmtime(path1)
    elif os.path.exists(path2):
        return os.path.getmtime(path2)
    return 0

# ---------------- Load datasets & Train ML model ----------------
@st.cache_resource
def load_and_train_models(mtime):
    """Load the dataset and train Random Forest and XGBoost models."""
    try:
        df = pd.read_csv("data/crop_recommendation.csv")
    except FileNotFoundError:
        # Fallback if the path is slightly different
        df = pd.read_csv("crop_recommendation.csv")
        
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Train scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Random Forest (with OOB evaluation for accuracy)
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42, oob_score=True)
    rf_model.fit(X_scaled, y)
    rf_accuracy = rf_model.oob_score_
    
    # 2. XGBoost (with Label Encoding)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train-test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    xgb_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_accuracy = xgb_model.score(X_test, y_test)
    
    # Retrain on full dataset for prediction
    xgb_model_full = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model_full.fit(X_scaled, y_encoded)
    
    # Calculate crop means for optimal comparison
    crop_means = df.groupby('label').mean().to_dict(orient='index')
    
    models = {
        "Random Forest": {
            "model": rf_model,
            "accuracy": rf_accuracy
        },
        "XGBoost": {
            "model": xgb_model_full,
            "accuracy": xgb_accuracy,
            "encoder": le
        }
    }
    
    return models, scaler, crop_means, df

try:
    mtime = get_dataset_mtime()
    models, scaler, crop_means, dataset = load_and_train_models(mtime)
    model_loaded = True
except Exception as e:
    st.error(f"Error training machine learning model: {e}")
    model_loaded = False

# Load market prices
try:
    market_data = pd.read_csv("data/market_prices.csv")
except FileNotFoundError:
    try:
        market_data = pd.read_csv("market_prices.csv")
    except FileNotFoundError:
        market_data = pd.DataFrame({"crop": [], "price_per_quintal": []})

# ---------------- Translation Dictionary ----------------
translations = {
    "English": {
        "title": "🌱 AGRIMIND Crop Advisory Dashboard",
        "subtitle": "AI-Powered Decisions for Sustainable and Profitable Farming",
        "tab_dashboard": "📋 Custom Advisory Report",
        "tab_analytics": "📊 Soil & Crop Analytics",
        "tab_faqs": "📖 Farming Guide & FAQs",
        "sec_soil": "Soil Nutrients",
        "sec_climate": "Climate Conditions",
        "n_label": "Nitrogen (N) - Soil Nutrient (mg/kg)",
        "p_label": "Phosphorus (P) - Soil Nutrient (mg/kg)",
        "k_label": "Potassium (K) - Soil Nutrient (mg/kg)",
        "ph_label": "Soil pH level (Acidity/Alkalinity)",
        "temp_label": "Air Temperature (°C)",
        "hum_label": "Air Humidity (%)",
        "rain_label": "Rainfall (mm)",
        "btn_recommend": "🔍 Generate Custom Advisory Report",
        "rec_crop": "Recommended Crop",
        "confidence": "Recommendation Confidence",
        "market_advisory": "Market Price Advisory",
        "market_desc": "Current average market price per quintal for the recommended crop:",
        "potential_yield": "Estimated Potential Yield",
        "potential_income": "Estimated Potential Gross Income",
        "fertilizer_advisory": "Soil Nutrients & Fertilizer Advisory",
        "irrigation_advisory": "Water & Irrigation Advisory",
        "optimal_compare": "Comparison: Input vs. Optimal Conditions",
        "param": "Parameter",
        "your_value": "Your Value",
        "optimal_value": "Optimal Value",
        "status": "Status",
        "leaf_health_card": "🌱 Check Leaf Health (External Link)",
        "leaf_health_desc": "Scan plant leaves for stress, disease, or pests using computer vision.",
        "faqs_title": "Frequently Asked Questions",
        "quick_stats": "Quick Dataset Statistics",
        "samples_count": "Total Dataset Samples",
        "crops_supported": "Crops Supported",
        "accuracy_title": "Model Performance",
        "accuracy_desc": "Select a classifier to use for recommendations and view its validation accuracy on our dataset.",
        "model_selection": "🤖 Classifier Model",
        "choose_classifier": "Choose Model Type",
    },
    "Hindi": {
        "title": "🌱 एग्रीमाइंड फसल सलाहकार डैशबोर्ड",
        "subtitle": "सतत और लाभदायक खेती के लिए एआई-संचालित निर्णय",
        "tab_dashboard": "📋 कस्टम सलाहकार रिपोर्ट",
        "tab_analytics": "📊 मिट्टी और फसल विश्लेषण",
        "tab_faqs": "📖 खेती गाइड और सामान्य प्रश्न",
        "sec_soil": "मिट्टी के पोषक तत्व",
        "sec_climate": "जलवायु की स्थिति",
        "n_label": "नाइट्रोजन (N) - मिट्टी का पोषक तत्व (मिलीग्राम/किग्रा)",
        "p_label": "फास्फोरस (P) - मिट्टी का पोषक तत्व (मिलीग्राम/किग्रा)",
        "k_label": "पोटेशियम (K) - मिट्टी का पोषक तत्व (मिलीग्राम/किग्रा)",
        "ph_label": "मिट्टी का pH स्तर (अम्लता/क्षारीयता)",
        "temp_label": "हवा का तापमान (°C)",
        "hum_label": "हवा की आर्द्रता (%)",
        "rain_label": "वर्षा (मिमी)",
        "btn_recommend": "🔍 कस्टम सलाहकार रिपोर्ट तैयार करें",
        "rec_crop": "अनुशंसित फसल",
        "confidence": "सिफारिश विश्वसनीयता",
        "market_advisory": "बाजार मूल्य सलाहकार",
        "market_desc": "अनुशंसित फसल के लिए वर्तमान औसत बाजार मूल्य प्रति क्विंटल:",
        "potential_yield": "अनुमानित संभावित उपज",
        "potential_income": "अनुमानित संभावित सकल आय",
        "fertilizer_advisory": "मिट्टी के पोषक तत्व और उर्वरक सलाह",
        "irrigation_advisory": "पानी और सिंचाई सलाह",
        "optimal_compare": "तुलना: इनपुट बनाम अनुकूलतम स्थितियां",
        "param": "पैरामीटर",
        "your_value": "आपका मान",
        "optimal_value": "अनुकूलतम मान",
        "status": "स्थिति",
        "leaf_health_card": "🌱 पत्ती स्वास्थ्य की जांच करें (बाहरी लिंक)",
        "leaf_health_desc": "कंप्यूटर विज़न का उपयोग करके तनाव, बीमारी या कीटों के लिए पौधों की पत्तियों को स्कैन करें।",
        "faqs_title": "अक्सर पूछे जाने वाले प्रश्न",
        "quick_stats": "त्वरित डेटासेट आंकड़े",
        "samples_count": "कुल डेटासेट नमूने",
        "crops_supported": "समर्थित फसलें",
        "accuracy_title": "मॉडल प्रदर्शन",
        "accuracy_desc": "सिफारिशों के लिए उपयोग करने के लिए एक क्लासिफायर चुनें और हमारे डेटासेट पर उसकी सत्यापन सटीकता देखें।",
        "model_selection": "🤖 क्लासिफायर मॉडल",
        "choose_classifier": "मॉडल का प्रकार चुनें",
    }
}

# ---------------- Crop Information Dictionary ----------------
crop_info = {
    "English": {
        "rice": {"name": "🌾 Rice", "desc": "A water-loving cereal crop that thrives in hot, humid tropical regions with high rainfall.", "yield": "3.5 - 4.5 tonnes/hectare"},
        "maize": {"name": "🌽 Maize (Corn)", "desc": "A versatile grain crop requiring moderate climate, well-drained soil, and steady sunlight.", "yield": "5.0 - 6.5 tonnes/hectare"},
        "chickpea": {"name": "🧆 Chickpea (Chana)", "desc": "A dry-climate pulse crop that requires very low humidity and thrives in sandy loam soil.", "yield": "1.2 - 1.8 tonnes/hectare"},
        "kidneybeans": {"name": "🫘 Kidney Beans (Rajma)", "desc": "A cool-season legume crop requiring mild temperatures and moderate moisture.", "yield": "1.5 - 2.2 tonnes/hectare"},
        "pigeonpeas": {"name": "🫛 Pigeonpeas (Arhar/Tur)", "desc": "A drought-resistant pulse crop well-suited for arid and semi-arid tropical soils.", "yield": "1.0 - 1.5 tonnes/hectare"},
        "mothbeans": {"name": "🫘 Moth Beans", "desc": "An extremely drought-tolerant pulse crop grown in arid and sandy soils.", "yield": "0.6 - 1.0 tonnes/hectare"},
        "mungbean": {"name": "🫛 Mungbean", "desc": "A short-duration legume crop that enriches soil nitrogen and requires mild warmth.", "yield": "0.8 - 1.2 tonnes/hectare"},
        "blackgram": {"name": "🫘 Black Gram (Urad)", "desc": "A nutritious legume that grows well in loamy soils and hot, humid climates.", "yield": "0.7 - 1.1 tonnes/hectare"},
        "lentil": {"name": "🧆 Lentil (Masur)", "desc": "A cool-season pulse crop that requires well-drained, moderately fertile soil.", "yield": "1.0 - 1.4 tonnes/hectare"},
        "pomegranate": {"name": "🍎 Pomegranate", "desc": "A drought-hardy fruit crop thriving in warm, dry-summer climates and alkaline soils.", "yield": "12 - 15 tonnes/hectare"},
        "banana": {"name": "🍌 Banana", "desc": "A high-yielding tropical fruit crop that requires deep organic-rich soil and high humidity.", "yield": "30 - 40 tonnes/hectare"},
        "mango": {"name": "🥭 Mango", "desc": "The king of fruits, needing dry tropical/subtropical climates with deep well-drained loam soil.", "yield": "8 - 12 tonnes/hectare"},
        "grapes": {"name": "🍇 Grapes", "desc": "A high-value fruit crop that thrives in cool, dry-summer climates with high potassium soils.", "yield": "15 - 20 tonnes/hectare"},
        "watermelon": {"name": "🍉 Watermelon", "desc": "A warm-season vining fruit requiring sandy well-drained soils and low moisture during ripening.", "yield": "25 - 35 tonnes/hectare"},
        "muskmelon": {"name": "🍈 Muskmelon", "desc": "A sweet desert vine fruit that requires dry heat, sandy soils, and light watering.", "yield": "18 - 25 tonnes/hectare"},
        "apple": {"name": "🍎 Apple", "desc": "A temperate fruit crop requiring cool winter chilling hours and well-aerated organic soil.", "yield": "10 - 15 tonnes/hectare"},
        "orange": {"name": "🍊 Orange", "desc": "A citrus fruit crop that requires mild winters, sunny summers, and well-drained acidic soil.", "yield": "15 - 22 tonnes/hectare"},
        "papaya": {"name": "🥭 Papaya", "desc": "A fast-growing tropical fruit tree that prefers frost-free climates and rich soil.", "yield": "40 - 55 tonnes/hectare"},
        "coconut": {"name": "🥥 Coconut", "desc": "A coastal palm crop that thrives in sandy soils, high humidity, and warm sea breezes.", "yield": "80 - 100 nuts/tree/year"},
        "cotton": {"name": "☁️ Cotton", "desc": "A warm-season fiber crop requiring high nitrogen nutrients, stable heat, and medium moisture.", "yield": "2.0 - 2.8 tonnes/hectare"},
        "jute": {"name": "🌾 Jute", "desc": "A natural fiber crop grown in alluvial soils under warm temperatures and high humidity.", "yield": "2.2 - 2.8 tonnes/hectare"},
        "coffee": {"name": "☕ Coffee", "desc": "A shade-loving highland plantation crop requiring cool humid climates and organic rich soil.", "yield": "1.2 - 1.8 tonnes/hectare"},
        "wheat": {"name": "🌾 Wheat", "desc": "A major cereal crop preferring cool dry weather during growth and warm sunny days for ripening.", "yield": "3.0 - 4.5 tonnes/hectare"},
        "barley": {"name": "🌾 Barley", "desc": "A highly resilient winter cereal crop that thrives in temperate dry climates and alkali soils.", "yield": "2.5 - 3.8 tonnes/hectare"},
        "sugarcane": {"name": "🎋 Sugarcane", "desc": "A high-water cash crop requiring high nitrogen soil, steady tropical heat, and rich drainage.", "yield": "70 - 100 tonnes/hectare"},
        "ragi": {"name": "🌾 Ragi (Finger Millet)", "desc": "A nutrient-rich dry millet crop that thrives in marginal soils and semi-arid zones.", "yield": "1.5 - 2.5 tonnes/hectare"},
        "bajra": {"name": "🌾 Bajra (Pearl Millet)", "desc": "A hardy, drought-tolerant crop suited for sandy soils and very low rainfall areas.", "yield": "1.2 - 2.0 tonnes/hectare"},
        "jowar": {"name": "🌾 Jowar (Sorghum)", "desc": "A versatile food-and-fodder dryland crop suited for clay-loam soil with medium moisture.", "yield": "1.8 - 2.8 tonnes/hectare"},
        "mustard": {"name": "🌼 Mustard", "desc": "An oilseed crop grown in cool winters, requiring high sulfur/nitrogen and low watering.", "yield": "1.2 - 2.0 tonnes/hectare"},
        "soybean": {"name": "🫘 Soybean", "desc": "A high-protein oilseed crop that enriches nitrogen and requires warm humid climates.", "yield": "2.0 - 3.0 tonnes/hectare"},
        "groundnut": {"name": "🥜 Groundnut (Peanut)", "desc": "A legume crop that develops pods underground, requiring loose sandy soils and moderate warmth.", "yield": "1.8 - 2.5 tonnes/hectare"},
        "sunflower": {"name": "🌻 Sunflower", "desc": "A deep-rooting oilseed crop that prefers well-aerated soils and moderate tropical sun.", "yield": "1.5 - 2.2 tonnes/hectare"},
        "potato": {"name": "🥔 Potato", "desc": "A high-potassium tuber crop requiring cool temperatures, loose acidic soil, and high nitrogen.", "yield": "20 - 30 tonnes/hectare"},
        "tomato": {"name": "🍅 Tomato", "desc": "A warm-season vegetable crop requiring stable watering, nitrogen/potassium, and mild air humidity.", "yield": "25 - 40 tonnes/hectare"},
        "onion": {"name": "🧅 Onion", "desc": "A shallow-rooted bulb vegetable requiring mild temperatures and well-drained loam soil.", "yield": "15 - 25 tonnes/hectare"},
        "tea": {"name": "🍵 Tea", "desc": "A premium plantation crop grown on sloped hillsides with highly acidic soil and heavy mist/rain.", "yield": "1.5 - 2.5 tonnes/hectare"},
    },
    "Hindi": {
        "rice": {"name": "🌾 चावल (धान)", "desc": "एक जल-प्रेमी अनाज की फसल जो भारी वर्षा वाले गर्म, आर्द्र उष्णकटिबंधीय क्षेत्रों में पनपती है।", "yield": "3.5 - 4.5 टन/हेक्टेयर"},
        "maize": {"name": "🌽 मक्का", "desc": "एक बहुमुखी अनाज की फसल जिसके लिए मध्यम जलवायु, अच्छी जल निकासी वाली मिट्टी और निरंतर धूप की आवश्यकता होती है।", "yield": "5.0 - 6.5 टन/हेक्टेयर"},
        "chickpea": {"name": "🧆 चना", "desc": "एक शुष्क जलवायु वाली दलहन फसल जिसे बहुत कम आर्द्रता की आवश्यकता होती है और बलुई दोमट मिट्टी में पनपती है।", "yield": "1.2 - 1.8 टन/हेक्टेयर"},
        "kidneybeans": {"name": "🫘 राजमा", "desc": "एक ठंडे मौसम की फलदार फसल जिसके लिए हल्के तापमान और मध्यम नमी की आवश्यकता होती है।", "yield": "1.5 - 2.2 टन/हेक्टेयर"},
        "pigeonpeas": {"name": "🫛 अरहर (तुअर)", "desc": "एक सूखा-प्रतिरोधी दलहन फसल जो शुष्क और अर्ध-शुष्क उष्णकटिबंधीय मिट्टी के लिए उपयुक्त है।", "yield": "1.0 - 1.5 टन/हेक्टेयर"},
        "mothbeans": {"name": "🫘 मोठ बाजरा", "desc": "एक अत्यंत सूखा-सहिष्णु दलहन फसल जो शुष्क और रेतीली मिट्टी में उगाई जाती है।", "yield": "0.6 - 1.0 टन/हेक्टेयर"},
        "mungbean": {"name": "🫛 मूंग", "desc": "एक कम अवधि की फलदार फसल जो मिट्टी में नाइट्रोजन बढ़ाती है और हल्की गर्मी की आवश्यकता होती है।", "yield": "0.8 - 1.2 टन/हेक्टेयर"},
        "blackgram": {"name": "🫘 उड़द", "desc": "एक पौष्टिक दलहन जो दोमट मिट्टी और गर्म, आर्द्र जलवायु में अच्छी तरह बढ़ती है।", "yield": "0.7 - 1.1 टन/हेक्टेयर"},
        "lentil": {"name": "🧆 मसूर", "desc": "एक ठंडे मौसम की दलहन फसल जिसके लिए अच्छी जल निकासी वाली और मध्यम उपजाऊ मिट्टी की आवश्यकता होती है।", "yield": "1.0 - 1.4 टन/हेक्टेयर"},
        "pomegranate": {"name": "🍎 अनार", "desc": "एक सूखा-सहिष्णु फल फसल जो गर्म, शुष्क-गर्मी की जलवायु और क्षारीय मिट्टी में पनपती है।", "yield": "12 - 15 टन/हेक्टेयर"},
        "banana": {"name": "🍌 केला", "desc": "एक उच्च उपज देने वाली उष्णकटिबंधीय फल फसल जिसके लिए गहरी जैविक-समृद्ध मिट्टी और उच्च आर्द्रता की आवश्यकता होती है।", "yield": "30 - 40 टन/हेक्टेयर"},
        "mango": {"name": "🥭 आम", "desc": "फलों का राजा, जिसे गहरी अच्छी जल निकासी वाली दोमट मिट्टी के साथ शुष्क उष्णकटिबंधीय/उपोष्णकटिबंधीय जलवायु की आवश्यकता होती है।", "yield": "8 - 12 टन/हेक्टेयर"},
        "grapes": {"name": "🍇 अंगूर", "desc": "एक उच्च मूल्य वाली फल फसल जो उच्च पोटेशियम वाली मिट्टी के साथ ठंडी, शुष्क-गर्मी वाली जलवायु में पनपती है।", "yield": "15 - 20 टन/हेक्टेयर"},
        "watermelon": {"name": "🍉 तरबूज", "desc": "गर्म मौसम का फल जिसके लिए रेतीली जल निकासी वाली मिट्टी और पकने के दौरान कम नमी की आवश्यकता होती है।", "yield": "25 - 35 टन/हेक्टेयर"},
        "muskmelon": {"name": "🍈 खरबूजा", "desc": "एक मीठा फल जिसे शुष्क गर्मी, रेतीली मिट्टी और हल्की सिंचाई की आवश्यकता होती है।", "yield": "18 - 25 टन/हेक्टेयर"},
        "apple": {"name": "🍎 सेब", "desc": "एक समशीतोष्ण फल फसल जिसके लिए ठंडे सर्दियों के मौसम और अच्छी तरह हवादार जैविक मिट्टी की आवश्यकता होती है।", "yield": "10 - 15 टन/हेक्टेयर"},
        "orange": {"name": "🍊 संतरा", "desc": "एक साइट्रस फल फसल जिसे हल्की सर्दियों, धूप वाली गर्मियों और अच्छी जल निकासी वाली अम्लीय मिट्टी की आवश्यकता होती है।", "yield": "15 - 22 टन/हेक्टेयर"},
        "papaya": {"name": "🥭 पपीता", "desc": "एक तेजी से बढ़ने वाला उष्णकटिबंधीय फल का पेड़ जो पाले से मुक्त जलवायु और समृद्ध मिट्टी पसंद करता है।", "yield": "40 - 55 टन/हेक्टेयर"},
        "coconut": {"name": "🥥 नारियल", "desc": "एक तटीय ताड़ की फसल जो रेतीली मिट्टी, उच्च आर्द्रता और गर्म समुद्री हवाओं में पनपती है।", "yield": "80 - 100 नारियल/पेड़/वर्ष"},
        "cotton": {"name": "☁️ कपास", "desc": "एक गर्म मौसम की रेशेदार फसल जिसे उच्च नाइट्रोजन पोषक तत्वों, स्थिर गर्मी और मध्यम नमी की आवश्यकता होती है।", "yield": "2.0 - 2.8 टन/हेक्टेयर"},
        "jute": {"name": "🌾 जूट", "desc": "एक प्राकृतिक रेशेदार फसल जो गर्म तापमान और उच्च आर्द्रता के तहत जलोढ़ मिट्टी में उगाई जाती है।", "yield": "2.2 - 2.8 टन/हेक्टेयर"},
        "coffee": {"name": "☕ कॉफी", "desc": "एक छाया-प्रेमी हाइलैंड वृक्षारोपण फसल जिसके लिए ठंडी आर्द्र जलवायु और जैविक समृद्ध मिट्टी की आवश्यकता होती है।", "yield": "1.2 - 1.8 टन/हेक्टेयर"},
        "wheat": {"name": "🌾 गेहूं", "desc": "एक प्रमुख अनाज फसल जो विकास के दौरान ठंडी शुष्क जलवायु और पकने के लिए गर्म धूप पसंद करती है।", "yield": "3.0 - 4.5 टन/हेक्टेयर"},
        "barley": {"name": "🌾 जौ", "desc": "एक अत्यधिक लचीली सर्दियों की अनाज की फसल जो समशीतोष्ण शुष्क जलवायु और क्षारीय मिट्टी में पनपती है।", "yield": "2.5 - 3.8 टन/हेक्टेयर"},
        "sugarcane": {"name": "🎋 गन्ना", "desc": "एक उच्च पानी की नकदी फसल जिसके लिए उच्च नाइट्रोजन मिट्टी, उष्णकटिबंधीय गर्मी और अच्छी जल निकासी की आवश्यकता होती है।", "yield": "70 - 100 टन/हेक्टेयर"},
        "ragi": {"name": "🌾 रागी (मडुआ)", "desc": "एक पोषक तत्वों से भरपूर शुष्क बाजरा फसल जो सीमांत मिट्टी और अर्ध-शुष्क क्षेत्रों में पनपती है।", "yield": "1.5 - 2.5 टन/हेक्टेयर"},
        "bajra": {"name": "🌾 बाजरा", "desc": "रेतीली मिट्टी और बहुत कम वर्षा वाले क्षेत्रों के लिए उपयुक्त एक कठोर, सूखा-सहिष्णु फसल।", "yield": "1.2 - 2.0 टन/हेक्टेयर"},
        "jowar": {"name": "🌾 ज्वार", "desc": "मध्यम नमी वाली मिट्टी के लिए उपयुक्त एक बहुमुखी खाद्य और चारा शुष्क भूमि फसल।", "yield": "1.8 - 2.8 टन/हेक्टेयर"},
        "mustard": {"name": "🌼 सरसों", "desc": "ठंडी सर्दियों में उगाई जाने वाली तिलहन फसल जिसे उच्च सल्फर/नाइट्रोजन और कम पानी की आवश्यकता होती है।", "yield": "1.2 - 2.0 टन/हेक्टेयर"},
        "soybean": {"name": "🫘 सोयाबीन", "desc": "एक उच्च प्रोटीन तिलहन फसल जो नाइट्रोजन बढ़ाती है और गर्म आर्द्र जलवायु की आवश्यकता होती है।", "yield": "2.0 - 3.0 टन/हेक्टेयर"},
        "groundnut": {"name": "🥜 मूंगफली", "desc": "एक फलदार फसल जो जमीन के नीचे बढ़ती है, जिसके लिए ढीली रेतीली मिट्टी और मध्यम गर्मी की आवश्यकता होती है।", "yield": "1.8 - 2.5 टन/हेक्टेयर"},
        "sunflower": {"name": "🌻 सूरजमुखी", "desc": "एक गहरी जड़ों वाली तिलहन फसल जो अच्छी तरह से हवादार मिट्टी और मध्यम उष्णकटिबंधीय धूप पसंद करती है।", "yield": "1.5 - 2.2 टन/हेक्टेयर"},
        "potato": {"name": "🥔 आलू", "desc": "एक उच्च पोटेशियम कंद फसल जिसके लिए ठंडे तापमान, ढीली अम्लीय मिट्टी और उच्च नाइट्रोजन की आवश्यकता होती है।", "yield": "20 - 30 टन/हेक्टेयर"},
        "tomato": {"name": "🍅 टमाटर", "desc": "एक गर्म मौसम की सब्जी फसल जिसे स्थिर पानी, नाइट्रोजन/पोटेशियम और मध्यम हवा की आर्द्रता की आवश्यकता होती है।", "yield": "25 - 40 टन/हेक्टेयर"},
        "onion": {"name": "🧅 प्याज", "desc": "एक कम गहरी जड़ों वाली कंद सब्जी जिसके लिए मध्यम तापमान और अच्छी जल निकासी वाली दोमट मिट्टी की आवश्यकता होती है।", "yield": "15 - 25 टन/हेक्टेयर"},
        "tea": {"name": "🍵 चाय", "desc": "पहाड़ी ढलानों पर उगाई जाने वाली एक बागानी फसल जिसे अत्यधिक अम्लीय मिट्टी और भारी धुंध/बारिश की आवश्यकता होती है।", "yield": "1.5 - 2.5 टन/हेक्टेयर"},
    }
}

# ---------------- Fertilizer Advisor Engine ----------------
def generate_fertilizer_advices(user_nutrients, optimal_nutrients, lang):
    advices = []
    
    # N check
    n_diff = user_nutrients["N"] - optimal_nutrients["N"]
    if n_diff < -15:
        if lang == "English":
            advices.append("⚠️ **Nitrogen (N) is low**: Apply nitrogen-rich fertilizers like **Urea** or **Ammonium Nitrate**. Adding well-rotted farmyard manure also helps build soil nitrogen over time.")
        else:
            advices.append("⚠️ **नाइट्रोजन (N) कम है**: **यूरिया** या **अमोनियम नाइट्रेट** जैसे नाइट्रोजन युक्त उर्वरक डालें। जैविक सड़ी खाद डालने से भी समय के साथ मिट्टी में नाइट्रोजन बढ़ाने में मदद मिलती है।")
    elif n_diff > 30:
        if lang == "English":
            advices.append("ℹ️ **Nitrogen (N) is high**: Avoid adding nitrogen fertilizers. Excess nitrogen causes vegetative foliage growth but delays crop maturation and increases pest risk.")
        else:
            advices.append("ℹ️ **नाइट्रोजन (N) अधिक है**: नाइट्रोजन उर्वरक डालने से बचें। अत्यधिक नाइट्रोजन से पौधे तो बढ़ेंगे लेकिन फसल पकने में देरी होगी और कीड़ों का खतरा बढ़ेगा।")
            
    # P check
    p_diff = user_nutrients["P"] - optimal_nutrients["P"]
    if p_diff < -10:
        if lang == "English":
            advices.append("⚠️ **Phosphorus (P) is low**: Apply **DAP (Diammonium Phosphate)** or **SSP (Single Superphosphate)**. Phosphorus is crucial for healthy root structure and early seedling vigor.")
        else:
            advices.append("⚠️ **फास्फोरस (P) कम है**: **DAP (डाई-अमोनियम फॉस्फेट)** या **SSP (सिंगल सुपर फॉस्फेट)** डालें। फास्फोरस मजबूत जड़ों और शुरुआती पौधों के विकास के लिए आवश्यक है।")
            
    # K check
    k_diff = user_nutrients["K"] - optimal_nutrients["K"]
    if k_diff < -10:
        if lang == "English":
            advices.append("⚠️ **Potassium (K) is low**: Suggest applying **MOP (Muriate of Potash)** or Potassium Sulfate. Potassium helps build disease immunity and improves grain/fruit weight and flavor.")
        else:
            advices.append("⚠️ **पोटेशियम (K) कम है**: **MOP (म्यूरिएट ऑफ पोटाश)** या पोटेशियम सल्फेट डालने का सुझाव दिया जाता है। पोटेशियम रोगों से लड़ने की क्षमता और फल के वजन व स्वाद में सुधार करता है।")
            
    # pH check
    ph = user_nutrients["ph"]
    opt_ph = optimal_nutrients["ph"]
    if ph < 5.5:
        if lang == "English":
            advices.append(f"⚠️ **Soil is acidic (pH {ph:.1f})**: Below optimal pH ({opt_ph:.1f}). Apply **agricultural lime (calcium carbonate)** to reduce acidity and make locked nutrients bio-available.")
        else:
            advices.append(f"⚠️ **मिट्टी अम्लीय है (pH {ph:.1f})**: अनुकूलतम स्तर ({opt_ph:.1f}) से कम। अम्लता को कम करने और बंद पोषक तत्वों को उपलब्ध कराने के लिए **कृषि चूना** डालें।")
    elif ph > 7.5:
        if lang == "English":
            advices.append(f"⚠️ **Soil is alkaline (pH {ph:.1f})**: Above optimal pH ({opt_ph:.1f}). Suggest adding **agricultural gypsum** or organic mulch (like pine needles or peat) to slowly lower the pH level.")
        else:
            advices.append(f"⚠️ **मिट्टी क्षारीय है (pH {ph:.1f})**: अनुकूलतम स्तर ({opt_ph:.1f}) से अधिक। pH को धीरे-धीरे कम करने के लिए **जिप्सम** या जैविक कम्पोस्ट/मल्च डालें।")
            
    if not advices:
        if lang == "English":
            advices.append("✅ **Soil nutrients are well-balanced!** No critical nutrient adjustments are needed. Maintain organic carbon with seasonal compost.")
        else:
            advices.append("✅ **मिट्टी के पोषक तत्व संतुलित हैं!** किसी महत्वपूर्ण पोषक तत्व सुधार की आवश्यकता नहीं है। जैविक तत्वों को बनाए रखने के लिए कम्पोस्ट डालते रहें।")
            
    return advices

# ---------------- Irrigation Advisor Engine ----------------
def generate_irrigation_advices(user_rainfall, crop_name, optimal_rainfall, lang):
    advices = []
    rain_diff = user_rainfall - optimal_rainfall
    
    if rain_diff < -30:
        deficit = abs(rain_diff)
        if lang == "English":
            advices.append(f"⚠️ **Rainfall deficit**: Current rainfall is **{user_rainfall:.1f} mm**, which is **{deficit:.1f} mm below** the optimal crop requirement of **{optimal_rainfall:.1f} mm**.")
            advices.append("💧 **Recommendation**: Set up supplemental irrigation. **Drip irrigation** is highly recommended to minimize evaporation loss, or **sprinkler irrigation** during early vegetative stages.")
        else:
            advices.append(f"⚠️ **वर्षा की कमी**: वर्तमान वर्षा **{user_rainfall:.1f} मिमी** है, जो फसल की अनुकूलतम आवश्यकता **{optimal_rainfall:.1f} मिमी** से **{deficit:.1f} मिमी कम** है।")
            advices.append("💧 **सलाह**: पूरक सिंचाई की व्यवस्था करें। वाष्पीकरण नुकसान को कम करने के लिए **ड्रिप सिंचाई** या शुरुआती चरणों में **फव्वारा सिंचाई** की सलाह दी जाती है।")
    elif rain_diff > 50:
        excess = rain_diff
        if lang == "English":
            advices.append(f"ℹ️ **Excessive rainfall**: Current rainfall is **{user_rainfall:.1f} mm**, which is **{excess:.1f} mm above** the optimal requirement of **{optimal_rainfall:.1f} mm**.")
            advices.append("🚜 **Recommendation**: Clean and establish field drainage channels to prevent waterlogging. Standing water can cause root hypoxia, root rot, and spread fungal diseases.")
        else:
            advices.append(f"ℹ️ **अत्यधिक वर्षा**: वर्तमान वर्षा **{user_rainfall:.1f} मिमी** है, जो अनुकूलतम आवश्यकता **{optimal_rainfall:.1f} मिमी** से **{excess:.1f} मिमी अधिक** है।")
            advices.append("🚜 **सलाह**: जलभराव को रोकने के लिए खेत में जल निकासी नालियों की सफाई करें। रुका हुआ पानी जड़ों को सड़ा सकता है और कवक रोगों को फैला सकता है।")
    else:
        if lang == "English":
            advices.append(f"✅ **Rainfall is optimal**: Current rainfall of **{user_rainfall:.1f} mm** fits the crop requirements perfectly.")
            advices.append("🌾 **Recommendation**: Follow standard watering intervals. No supplemental emergency watering needed. Monitor weather forecasts for sudden rains.")
        else:
            advices.append(f"✅ **वर्षा अनुकूल है**: वर्तमान वर्षा **{user_rainfall:.1f} मिमी** फसल की आवश्यकताओं के बिल्कुल अनुकूल है।")
            advices.append("🌾 **सलाह**: मानक सिंचाई अंतराल का पालन करें। किसी आपातकालीन पूरक सिंचाई की आवश्यकता नहीं है। अचानक बारिश के लिए मौसम के पूर्वानुमान पर नज़र रखें।")
            
    return advices

# ---------------- Streamlit UI Layout ----------------
# Sidebar
st.sidebar.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: #059669; margin: 0;">🌐 AGRIMIND AI</h2>
        <small style="color: #64748b;">Smart Crop Advisory & Market Price</small>
    </div>
    """,
    unsafe_allow_html=True
)

lang = st.sidebar.radio("🌐 Select Language / भाषा चुनें", ["English", "Hindi"], horizontal=True)

st.sidebar.markdown("---")

# Display Stats in Sidebar
if model_loaded:
    st.sidebar.subheader(translations[lang]["quick_stats"])
    st.sidebar.metric(translations[lang]["samples_count"], f"{dataset.shape[0]} 📊")
    st.sidebar.metric(translations[lang]["crops_supported"], f"{dataset['label'].nunique()} 🌱")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader(translations[lang]["model_selection"])
    selected_model_name = st.sidebar.selectbox(
        translations[lang]["choose_classifier"],
        options=["Random Forest", "XGBoost"],
        index=0
    )
    
    accuracy = models[selected_model_name]["accuracy"]
    st.sidebar.markdown(f"<small>{translations[lang]['accuracy_desc']}</small>", unsafe_allow_html=True)
    st.sidebar.markdown(f"<div class='badge-value'>🎯 {selected_model_name} Accuracy: {accuracy:.2%}</div>", unsafe_allow_html=True)

# Main Title & Subheader
st.markdown(
    f"""
    <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); 
                padding: 30px; border-radius: 20px; color: white; margin-bottom: 30px; 
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);">
        <h1 style="color: #34d399; margin: 0 0 10px 0; font-size: 32px;">{translations[lang]["title"]}</h1>
        <p style="margin: 0; font-size: 16px; opacity: 0.9; font-weight: 300;">{translations[lang]["subtitle"]}</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    translations[lang]["tab_dashboard"], 
    translations[lang]["tab_analytics"], 
    translations[lang]["tab_faqs"]
])

# Under tab 1
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown(f"### 🧪 {translations[lang]['sec_soil']}")
        
        N = st.slider(translations[lang]["n_label"], min_value=0, max_value=160, value=75, step=1)
        P = st.slider(translations[lang]["p_label"], min_value=5, max_value=150, value=65, step=1)
        K = st.slider(translations[lang]["k_label"], min_value=5, max_value=260, value=75, step=1)
        pH = st.slider(translations[lang]["ph_label"], min_value=3.5, max_value=9.5, value=6.5, step=0.1)
        
    with col2:
        st.markdown(f"### 🌦️ {translations[lang]['sec_climate']}")
        
        temperature = st.slider(translations[lang]["temp_label"], min_value=5.0, max_value=45.0, value=25.0, step=0.5)
        humidity = st.slider(translations[lang]["hum_label"], min_value=10.0, max_value=100.0, value=70.0, step=1.0)
        rainfall = st.slider(translations[lang]["rain_label"], min_value=20.0, max_value=320.0, value=120.0, step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calculate Button
    if st.button(translations[lang]["btn_recommend"]):
        if model_loaded:
            # Scale and predict
            input_vector = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
            input_scaled = scaler.transform(input_vector)
            
            # Get selected model info
            model_info = models[selected_model_name]
            model_obj = model_info["model"]
            
            if selected_model_name == "XGBoost":
                encoder = model_info["encoder"]
                pred_encoded = model_obj.predict(input_scaled)[0]
                prediction = encoder.inverse_transform([pred_encoded])[0]
                prediction_proba = model_obj.predict_proba(input_scaled)[0]
                classes = encoder.classes_
            else:
                prediction = model_obj.predict(input_scaled)[0]
                prediction_proba = model_obj.predict_proba(input_scaled)[0]
                classes = model_obj.classes_
            
            top_idx = np.argmax(prediction_proba)
            confidence_score = prediction_proba[top_idx]
            
            # Fetch Crop Details
            pred_lower = prediction.lower()
            crop_meta = crop_info[lang].get(pred_lower, {"name": prediction.capitalize(), "desc": "", "yield": "N/A"})
            
            # Get optimal crop conditions for comparison
            optimal = crop_means.get(pred_lower, {"N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, "ph": pH, "rainfall": rainfall})
            
            # Fetch Market price
            price_row = market_data[market_data["crop"].str.lower() == pred_lower]
            price_per_quintal = price_row["price_per_quintal"].values[0] if not price_row.empty else "N/A"
            
            st.markdown("<hr style='border-color: #cbd5e1;'>", unsafe_allow_html=True)
            st.markdown(f"<h2>🌾 Custom Agriculture Report</h2>", unsafe_allow_html=True)
            
            # Result Columns
            res_col1, res_col2 = st.columns([1.2, 1], gap="medium")
            
            with res_col1:
                # Crop recommendation Card
                st.markdown(
                    f"""
                    <div class="advisory-card">
                        <div class="card-header-container">
                            <div class="card-icon">🌱</div>
                            <div>
                                <h4 class="card-title">{translations[lang]["rec_crop"]}</h4>
                            </div>
                        </div>
                        <div class="card-body">
                            <h2 style="color: #059669; margin: 0 0 10px 0;">{crop_meta["name"]}</h2>
                            <p style="margin-bottom: 12px;">{crop_meta["desc"]}</p>
                            <span class="badge-value">{translations[lang]["confidence"]}: {confidence_score:.1%}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Fertilizer Advisory Card
                nutrients_user = {"N": N, "P": P, "K": K, "ph": pH}
                nutrients_opt = {"N": optimal["N"], "P": optimal["P"], "K": optimal["K"], "ph": optimal["ph"]}
                fertilizer_advices = generate_fertilizer_advices(nutrients_user, nutrients_opt, lang)
                
                fert_html = ""
                for adv in fertilizer_advices:
                    fert_html += f"<li style='margin-bottom: 8px;'>{adv}</li>"
                
                st.markdown(
                    f"""
                    <div class="advisory-card">
                        <div class="card-header-container">
                            <div class="card-icon">🧪</div>
                            <div>
                                <h4 class="card-title">{translations[lang]["fertilizer_advisory"]}</h4>
                            </div>
                        </div>
                        <div class="card-body">
                            <ul style="padding-left: 20px; margin: 0;">
                                {fert_html}
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            with res_col2:
                # Market Advisory Card
                calc_html = ""
                if price_per_quintal != "N/A":
                    # Income Calculation: assume 1 hectare yield in quintals (1 tonne = 10 quintals)
                    try:
                        # Parse yield range (take average)
                        yield_str = crop_meta["yield"].split()[0]
                        if "-" in yield_str:
                            yield_avg = sum(float(x) for x in yield_str.split("-")) / 2
                        else:
                            yield_avg = float(yield_str)
                            
                        quintals_per_hectare = yield_avg * 10
                        gross_income = quintals_per_hectare * price_per_quintal
                        
                        calc_html = (
                            f'<div style="margin-top: 12px; padding: 12px; background-color: #f8fafc; border-radius: 8px; border: 1px dashed #cbd5e1;">'
                            f'<small style="color: #64748b; display:block;">{translations[lang]["potential_yield"]}</small>'
                            f'<b style="color: #0f172a; font-size: 15px;">{crop_meta["yield"]}</b>'
                            f'<small style="color: #64748b; display:block; margin-top:8px;">{translations[lang]["potential_income"]}</small>'
                            f'<b style="color: #059669; font-size: 18px;">₹{gross_income:,.2f} / hectare</b>'
                            f'</div>'
                        )
                    except Exception:
                        pass
                
                st.markdown(
                    f"""
                    <div class="advisory-card">
                        <div class="card-header-container">
                            <div class="card-icon">💰</div>
                            <div>
                                <h4 class="card-title">{translations[lang]["market_advisory"]}</h4>
                            </div>
                        </div>
                        <div class="card-body">
                            <p style="margin-bottom: 8px;">{translations[lang]["market_desc"]}</p>
                            <h3 style="color: #b45309; margin: 0;">₹{price_per_quintal} / quintal</h3>
                            {calc_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Irrigation Advisory Card
                irrigation_advices = generate_irrigation_advices(rainfall, pred_lower, optimal["rainfall"], lang)
                irr_html = ""
                for adv in irrigation_advices:
                    irr_html += f"<li style='margin-bottom: 8px;'>{adv}</li>"
                    
                st.markdown(
                    f"""
                    <div class="advisory-card">
                        <div class="card-header-container">
                            <div class="card-icon">💧</div>
                            <div>
                                <h4 class="card-title">{translations[lang]["irrigation_advisory"]}</h4>
                            </div>
                        </div>
                        <div class="card-body">
                            <ul style="padding-left: 20px; margin: 0;">
                                {irr_html}
                            </ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # ---------------- Rendering Interactive Charts in Row ----------------
            st.markdown("---")
            st.markdown(f"### 📊 {translations[lang]['optimal_compare']}")
            
            chart_col1, chart_col2 = st.columns([1.2, 1], gap="medium")
            
            with chart_col1:
                # Plotly comparison chart
                try:
                    import plotly.graph_objects as go
                    
                    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                    feature_labels = {
                        'N': 'Nitrogen (N)', 'P': 'Phosphorus (P)', 'K': 'Potassium (K)',
                        'temperature': 'Temperature', 'humidity': 'Humidity',
                        'ph': 'pH Level', 'rainfall': 'Rainfall'
                    } if lang == "English" else {
                        'N': 'नाइट्रोजन (N)', 'P': 'फास्फोरस (P)', 'K': 'पोटेशियम (K)',
                        'temperature': 'तापमान', 'humidity': 'आर्द्रता',
                        'ph': 'pH स्तर', 'rainfall': 'वर्षा'
                    }
                    
                    # Calculate percentage of optimal
                    percentages = []
                    user_vals = []
                    opt_vals = []
                    
                    for f in features:
                        user_val = nutrients_user["ph"] if f == "ph" else (rainfall if f == "rainfall" else (temperature if f == "temperature" else (humidity if f == "humidity" else nutrients_user[f])))
                        opt_val = optimal[f]
                        user_vals.append(user_val)
                        opt_vals.append(opt_val)
                        if opt_val == 0:
                            percentage = 100
                        else:
                            percentage = (user_val / opt_val) * 100
                        percentages.append(percentage)
                        
                    df_chart = pd.DataFrame({
                        'Feature': [feature_labels[f] for f in features],
                        'Your Soil Value': user_vals,
                        'Optimal Value': opt_vals,
                        'Percentage of Optimal (%)': percentages
                    })
                    
                    fig = go.Figure()
                    
                    # Custom color logic: green if close to optimal, amber/red if deficit/excess
                    colors = []
                    for val in percentages:
                        if val < 70 or val > 130:
                            colors.append('#f59e0b')  # Amber alert
                        elif val < 50 or val > 150:
                            colors.append('#ef4444')  # Red alert
                        else:
                            colors.append('#10b981')  # Optimal green
                            
                    fig.add_trace(go.Bar(
                        y=df_chart['Feature'],
                        x=df_chart['Percentage of Optimal (%)'],
                        orientation='h',
                        name='Your Soil % of Optimal',
                        marker=dict(
                            color=colors,
                            line=dict(color='#0f766e', width=1)
                        ),
                        hovertemplate='%{y}: %{x:.1f}% of optimal<extra></extra>'
                    ))
                    
                    # Add a vertical target line at 100%
                    fig.add_shape(
                        type="line",
                        x0=100, y0=-0.5, x1=100, y1=len(features)-0.5,
                        line=dict(color="#047857", width=3, dash="dash"),
                    )
                    
                    fig.update_layout(
                        xaxis=dict(
                            title="Percentage of Optimal (%)",
                            gridcolor="#f1f5f9",
                        ),
                        yaxis=dict(
                            gridcolor="#f1f5f9"
                        ),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=350,
                        margin=dict(l=20, r=20, t=10, b=20),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    # Fallback to standard streamlit bar chart
                    st.warning("Could not load Plotly comparison chart. Showing standard layout.")
                    st.bar_chart(df_chart.set_index('Feature')['Percentage of Optimal (%)'])
            
            with chart_col2:
                # Top 3 recommended crops & confidence
                try:
                    import plotly.express as px
                    
                    sorted_idx = np.argsort(prediction_proba)[::-1]
                    top_crops = [classes[i] for i in sorted_idx[:3]]
                    top_probs = [prediction_proba[i] * 100 for i in sorted_idx[:3]]
                    
                    translated_names = []
                    for c in top_crops:
                        translated_names.append(crop_info[lang].get(c.lower(), {}).get("name", c.capitalize()))
                        
                    df_prob = pd.DataFrame({
                        'Crop': translated_names,
                        'Confidence (%)': top_probs
                    })
                    
                    fig_prob = px.bar(
                        df_prob,
                        x='Confidence (%)',
                        y='Crop',
                        orientation='h',
                        text='Confidence (%)',
                        color='Confidence (%)',
                        color_continuous_scale='Greens',
                    )
                    
                    fig_prob.update_traces(
                        texttemplate='%{text:.1f}%',
                        textposition='outside',
                        marker=dict(line=dict(color='#047857', width=1))
                    )
                    
                    fig_prob.update_layout(
                        xaxis=dict(
                            title="Confidence (%)",
                            range=[0, 115],
                            gridcolor="#f1f5f9",
                        ),
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        height=280,
                        margin=dict(l=20, r=20, t=10, b=20),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                except Exception as e:
                    st.write(df_prob)
        else:
            st.error("Crop recommendation model is currently offline. Please check data files.")

# Under tab 2 (Analytics)
with tab2:
    st.markdown("### 📊 Crop Dataset Conditions Profile")
    if model_loaded:
        st.markdown(
            "Below are the average optimal conditions required for the crops in our dataset. "
            "Use this matrix to understand the crop preferences."
        )
        
        # Display dataset description table
        st.dataframe(
            dataset.groupby('label').mean().round(2),
            use_container_width=True
        )
        
        # Display a scatter plot for distribution
        try:
            import plotly.express as px
            st.markdown("#### 🧬 Parameter Distribution Scatter Map")
            
            scatter_col1, scatter_col2 = st.columns([1, 1])
            with scatter_col1:
                x_axis = st.selectbox("Select X-Axis", ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], index=0)
            with scatter_col2:
                y_axis = st.selectbox("Select Y-Axis", ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'], index=6)
                
            fig_scatter = px.scatter(
                dataset,
                x=x_axis,
                y=y_axis,
                color='label',
                title=f"Crop distribution based on {x_axis} and {y_axis}",
                opacity=0.7,
                color_discrete_sequence=px.colors.qualitative.Dark2
            )
            fig_scatter.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title_text='Crop'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception:
            pass
    else:
        st.error("No dataset available to run analytics.")

# Under tab 3 (Leaf Health & FAQs)
with tab3:
    st.markdown(f"### {translations[lang]['faqs_title']}")
    
    # Render FAQs
    faq_data = [
        {
            "q": "What do Nitrogen, Phosphorus, and Potassium represent?",
            "a": "N, P, and K are the primary macronutrients required by plants. Nitrogen (N) promotes foliage/leaf growth, Phosphorus (P) supports root systems, flower/fruit development, and Potassium (K) regulates water flow and enhances overall disease resistance."
        },
        {
            "q": "What is the importance of soil pH?",
            "a": "Soil pH is a measure of acidity or alkalinity. Most crops prefer slightly acidic to neutral soils (pH 6.0 to 7.0) because nutrients are most accessible to plant roots within this range."
        },
        {
            "q": "How does weather information impact crop selection?",
            "a": "Humidity and temperature affect transpiration and plant metabolic rates, while rainfall determines the availability of natural water. Recommending a crop that matches local weather reduces irrigation costs."
        }
    ]
    
    for item in faq_data:
        with st.expander(f"❓ {item['q']}", expanded=False):
            st.write(item['a'])
            
    st.markdown("---")
    
    # Beautiful External Tool Card
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    padding: 24px; border-radius: 16px; border: 1px solid #334155; 
                    box-shadow: var(--shadow); margin-top: 20px;">
            <h4 style="color: #38bdf8; margin: 0 0 8px 0; font-family: 'Outfit'; font-size: 18px;">
                {translations[lang]['leaf_health_card']}
            </h4>
            <p style="color: #94a3b8; font-size: 14px; margin-bottom: 16px;">
                {translations[lang]['leaf_health_desc']}
            </p>
            <a href="https://ai-powered-leaf-stress-detection-for-sustainable-farming-ngsvn.streamlit.app/" 
               target="_blank" 
               style="display: inline-block; background-color: #0284c7; color: white; 
                      padding: 10px 20px; border-radius: 8px; text-decoration: none; 
                      font-weight: 600; font-size: 14px;">
                🌱 Launch Scan Tool
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------- Inject Chatbase Chatbot (floating bottom-right) ----------------
chatbase_script = """
<script>
(function(){if(!window.chatbase||window.chatbase("getState")!=="initialized"){window.chatbase=(...arguments)=>{if(!window.chatbase.q){window.chatbase.q=[]}window.chatbase.q.push(arguments)};window.chatbase=new Proxy(window.chatbase,{get(target,prop){if(prop==="q"){return target.q}return(...args)=>target(prop,...args)}})}const onLoad=function(){const script=document.createElement("script");script.src="https://www.chatbase.co/embed.min.js";script.id="stGSNcflAgIWun5ntpbM5";script.domain="www.chatbase.co";document.body.appendChild(script)};if(document.readyState==="complete"){onLoad()}else{window.addEventListener("load",onLoad)}})();
</script>
"""
st.markdown(chatbase_script, unsafe_allow_html=True)
