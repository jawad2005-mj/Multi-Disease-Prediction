"""
Multi-Disease Prediction System with Admin Panel
Streamlit Application - Python Only (No External Auth Libraries)

Complete Features:
- TASK 2: Intelligent Auto-Prediction Engine (all models run automatically)
- TASK 3: Complete Patient GUI with PDF reports and visualization
- TASK 4: Professional Admin Panel with authentication & analytics

Run: streamlit run app.py
"""

import os
import sys
import json
import hashlib
import pickle
import warnings
from datetime import datetime, date, timedelta
from io import BytesIO
import base64

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
import bcrypt

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════════

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = None
    st.session_state.user_role = None
    st.session_state.theme = "light"

# Database files
USERS_DB = "users_database.json"
PREDICTIONS_LOG = "predictions_log.json"
ACTIVITY_LOG = "activity_log.json"

# ════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & STYLING
# ════════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="MedPredict | Disease Prediction & Admin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }

.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    padding: 28px 32px; border-radius: 14px; margin-bottom: 24px;
    border: 1px solid #334155;
}
.main-header h1 { color: #e2e8f0; font-size: 1.9rem; font-weight: 700; margin: 0; }
.main-header p  { color: #94a3b8; font-size: 0.88rem; margin: 6px 0 0 0; }

.patient-card {
    background: #1e293b; border: 1px solid #334155;
    border-radius: 12px; padding: 16px 22px; margin-bottom: 20px;
}
.result-positive {
    background: #450a0a; border: 1px solid #f87171;
    border-left: 5px solid #ef4444; border-radius: 10px; padding: 20px 24px;
}
.result-negative {
    background: #052e16; border: 1px solid #4ade80;
    border-left: 5px solid #22c55e; border-radius: 10px; padding: 20px 24px;
}
.result-title { font-size: 1.4rem; font-weight: 700; margin: 0 0 4px 0; }
.result-conf  { font-size: 0.88rem; color: #94a3b8; margin: 0; }

.prob-row { margin-bottom: 10px; }
.prob-meta { display:flex; justify-content:space-between; font-size:0.82rem;
             color:#cbd5e1; margin-bottom:4px; }
.prob-track { background:#1e293b; border-radius:4px; height:8px; overflow:hidden; }
.prob-fill  { height:8px; border-radius:4px; }

.info-box {
    background: #0f172a; border: 1px solid #1e3a5f; border-radius: 10px;
    padding: 14px 18px; font-size: 0.83rem; color: #93c5fd; line-height: 1.6;
}
.scale-note {
    background: #1e1b4b; border: 1px solid #4338ca; border-radius: 8px;
    padding: 12px 16px; font-size: 0.80rem; color: #a5b4fc; margin-bottom: 16px;
}
.disclaimer {
    background: #1c1307; border: 1px solid #78350f; border-radius: 8px;
    padding: 12px 16px; font-size: 0.78rem; color: #fbbf24; margin-top: 16px;
}
div[data-testid="stButton"] > button {
    background: #3b82f6 !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; padding: 0.6rem 1.6rem !important;
}
div[data-testid="stButton"] > button:hover { background: #2563eb !important; }

.metric-box {
    background: #f0f2f6; border-radius: 10px; padding: 20px;
    text-align: center; border: 1px solid #ddd;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #0066CC; }
.metric-label { font-size: 0.9rem; color: #666; margin-top: 8px; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
# DATABASE FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def hash_password(password):
    """Hash password with bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    """Verify password against hash"""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except:
        return False

def load_users_db():
    """Load users database"""
    if os.path.exists(USERS_DB):
        try:
            with open(USERS_DB, 'r') as f:
                return json.load(f)
        except:
            return {"users": []}
    return {"users": []}

def save_users_db(data):
    """Save users database"""
    with open(USERS_DB, 'w') as f:
        json.dump(data, f, indent=2)

def load_predictions_log():
    """Load predictions history"""
    if os.path.exists(PREDICTIONS_LOG):
        try:
            with open(PREDICTIONS_LOG, 'r') as f:
                return json.load(f)
        except:
            return {"predictions": []}
    return {"predictions": []}

def save_predictions_log(data):
    """Save predictions history"""
    with open(PREDICTIONS_LOG, 'w') as f:
        json.dump(data, f, indent=2)

def load_activity_log():
    """Load activity log"""
    if os.path.exists(ACTIVITY_LOG):
        try:
            with open(ACTIVITY_LOG, 'r') as f:
                return json.load(f)
        except:
            return {"activities": []}
    return {"activities": []}

def log_activity(user, action, details=""):
    """Log admin activity"""
    logs = load_activity_log()
    logs["activities"].append({
        "timestamp": datetime.now().isoformat(),
        "user": user,
        "action": action,
        "details": details
    })
    with open(ACTIVITY_LOG, 'w') as f:
        json.dump(logs, f, indent=2)

# ════════════════════════════════════════════════════════════════════════════════
# MODEL LOADING & PREDICTION ENGINE (TASK 2)
# ════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load all disease prediction models"""
    BASE = os.path.dirname(os.path.abspath(__file__))
    
    try:
        scaler = joblib.load(os.path.join(BASE, "scaler.pkl"))
        encoder = joblib.load(os.path.join(BASE, "label_encoder.pkl"))
        
        models = {
            "anemia": joblib.load(os.path.join(BASE, "anemia_model.pkl")),
            "thalassemia": joblib.load(os.path.join(BASE, "thalassemia_model.pkl")),
            "diabetes": joblib.load(os.path.join(BASE, "diabetes_model.pkl")),
            "heart_disease": joblib.load(os.path.join(BASE, "heart_disease_model.pkl")),
            "thrombocytopenia": joblib.load(os.path.join(BASE, "thrombocytopenia_model.pkl")),
            "multiclass": joblib.load(os.path.join(BASE, "multiclass_model.pkl")),
        }
        return scaler, encoder, models, True
    except Exception as e:
        return None, None, None, False

# Load models
SCALER, ENCODER, MODELS, MODELS_LOADED = load_all_models()

if not MODELS_LOADED:
    SCALER_FEATS = []
    RAW_FEATS = []
else:
    SCALER_FEATS = list(SCALER.feature_names_in_)
    RAW_FEATS = [f for f in SCALER_FEATS if f not in ("HGB_RBC_ratio", "Glucose_Chol_ratio", "WBC_PLT_ratio", "NLR")]

# Disease configuration
DISEASE_CONFIG = {
    "anemia": {
        "emoji": "🩸",
        "name": "Anemia",
        "description": "Low red blood cell count leading to reduced oxygen transport",
        "key_features": ["Hemoglobin", "Red Blood Cells", "Hematocrit", "Mean Corpuscular Volume"],
        "medical_info": "Anemia is diagnosed when hemoglobin <12 g/dL (women) or <13.5 g/dL (men). This reduces oxygen-carrying capacity and affects oxygen delivery to tissues.",
    },
    "thalassemia": {
        "emoji": "🧬",
        "name": "Thalassemia",
        "description": "Genetic blood disorder with abnormal hemoglobin synthesis",
        "key_features": ["Hemoglobin", "Red Blood Cells", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin"],
        "medical_info": "Thalassemia shows marked microcytosis (MCV 55-70 fL), elevated RBC count, and severe hypochromia due to defective globin chain synthesis.",
    },
    "diabetes": {
        "emoji": "🍬",
        "name": "Diabetes",
        "description": "Metabolic disorder with elevated blood glucose levels",
        "key_features": ["Glucose", "HbA1c", "Insulin", "BMI", "Cholesterol"],
        "medical_info": "Diabetes diagnosed when fasting glucose ≥126 mg/dL or HbA1c ≥6.5%. Results from impaired glucose homeostasis and pancreatic dysfunction.",
    },
    "heart_disease": {
        "emoji": "❤️",
        "name": "Heart Disease",
        "description": "Cardiac disorder affecting the heart and blood vessels",
        "key_features": ["Systolic Blood Pressure", "Diastolic Blood Pressure", "Troponin", "C-reactive Protein", "LDL Cholesterol"],
        "medical_info": "Heart disease risk increases with elevated BP (>140 mmHg), LDL (>100 mg/dL), elevated troponin (>99th percentile), and CRP (>3.0 mg/L).",
    },
    "thrombocytopenia": {
        "emoji": "🔴",
        "name": "Thrombocytopenia",
        "description": "Abnormally low platelet count with bleeding risk",
        "key_features": ["Platelets", "White Blood Cells", "Hemoglobin", "Hematocrit"],
        "medical_info": "Thrombocytopenia defined as platelet count <150,000/μL. Counts <10,000/μL risk spontaneous hemorrhage.",
    }
}

def compute_engineered_features(raw_vals):
    """Compute engineered features for prediction"""
    v = dict(raw_vals)
    v["HGB_RBC_ratio"] = v.get("Hemoglobin", 0.5) / (v.get("Red Blood Cells", 0.5) + 1e-9)
    v["Glucose_Chol_ratio"] = v.get("Glucose", 0.5) / (v.get("Cholesterol", 0.5) + 1e-9)
    v["WBC_PLT_ratio"] = v.get("White Blood Cells", 0.5) / (v.get("Platelets", 0.5) + 1e-9)
    return v

def predict_disease(raw_vals):
    """
    TASK 2: Intelligent Auto-Prediction Engine
    Runs patient data through ALL disease models automatically
    Returns ranked predictions with confidence scores
    """
    if not MODELS_LOADED:
        return None, "Models not loaded"
    
    try:
        full_vals = compute_engineered_features(raw_vals)
        df = pd.DataFrame([full_vals], columns=SCALER_FEATS)
        X_scaled = pd.DataFrame(SCALER.transform(df), columns=SCALER_FEATS)
        
        # Run through ALL disease models
        predictions = {}
        for disease_key in ["anemia", "thalassemia", "diabetes", "heart_disease", "thrombocytopenia"]:
            if disease_key in MODELS:
                probs = MODELS[disease_key].predict_proba(X_scaled)[0]
                predictions[disease_key] = float(probs[1])
        
        # Rank by confidence
        ranked = sorted([(k, v) for k, v in predictions.items()], key=lambda x: x[1], reverse=True)
        
        return {
            "binary_predictions": predictions,
            "ranked": ranked,
            "top_disease": ranked[0][0] if ranked else None,
            "top_confidence": ranked[0][1] if ranked else 0,
        }, None
        
    except Exception as e:
        return None, str(e)

def generate_medical_reasoning(disease_key, confidence, input_vals):
    """Generate clinical reasoning for prediction"""
    config = DISEASE_CONFIG.get(disease_key, {})
    
    reasoning = f"""
    **Clinical Analysis:**
    
    Predicted Disease: **{config['name'].upper()}**
    Confidence Score: **{confidence*100:.1f}%**
    
    **Key Biomarker Observations:**
    """
    
    key_features = config.get("key_features", [])
    for feat in key_features:
        if feat in input_vals:
            val = input_vals[feat]
            status = "↑ Elevated" if val > 0.7 else ("↓ Low" if val < 0.3 else "→ Normal")
            reasoning += f"\n- {feat}: {val:.2f}/1.0 {status}"
    
    reasoning += f"\n\n**Clinical Context:**\n{config.get('medical_info', '')}"
    
    return reasoning

# ════════════════════════════════════════════════════════════════════════════════
# PDF REPORT GENERATION (TASK 3)
# ════════════════════════════════════════════════════════════════════════════════

def generate_pdf_report(patient_info, input_vals, predictions):
    """Generate professional PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                           rightMargin=0.5*inch, leftMargin=0.5*inch,
                           topMargin=0.75*inch, bottomMargin=0.5*inch)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#0066CC'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    story.append(Paragraph("🩺 MedPredict Disease Prediction Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Patient Information
    story.append(Paragraph("Patient Information", styles['Heading2']))
    patient_data = [
        ["Field", "Value"],
        ["Name", patient_info.get("name", "N/A")],
        ["Age", str(patient_info.get("age", "N/A"))],
        ["Gender", patient_info.get("gender", "N/A")],
        ["Test Date", str(patient_info.get("test_date", date.today()))],
        ["Report Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
    patient_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Prediction Results
    if predictions:
        story.append(Paragraph("Prediction Results", styles['Heading2']))
        
        top_disease = predictions.get("top_disease", "Unknown")
        top_confidence = predictions.get("top_confidence", 0)
        
        config = DISEASE_CONFIG.get(top_disease, {})
        risk_level = "No Risk" if top_confidence < 0.20 else ("Low" if top_confidence < 0.50 else ("Moderate" if top_confidence < 0.75 else "High"))
        
        pred_data = [
            ["Metric", "Result"],
            ["Predicted Disease", f"{config.get('emoji', '')} {config.get('name', top_disease)}"],
            ["Confidence Score", f"{top_confidence*100:.1f}%"],
            ["Risk Level", risk_level],
        ]
        pred_table = Table(pred_data, colWidths=[2*inch, 4*inch])
        pred_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#228B22')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(pred_table)
        story.append(Spacer(1, 0.2*inch))
    
    # Biomarker Values
    story.append(Paragraph("Input Biomarker Values", styles['Heading2']))
    biomarker_data = [["Biomarker", "Value (0-1 scale)"]]
    for feat in RAW_FEATS[:15]:
        if feat in input_vals:
            biomarker_data.append([feat, f"{input_vals[feat]:.3f}"])
    
    biomarker_table = Table(biomarker_data, colWidths=[3*inch, 3*inch])
    biomarker_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066CC')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(biomarker_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.red,
        alignment=TA_CENTER
    )
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This report is for research and educational purposes only. "
        "It is NOT a substitute for professional medical diagnosis. Always consult a qualified healthcare professional.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# ════════════════════════════════════════════════════════════════════════════════
# AUTHENTICATION & USER MANAGEMENT (TASK 4)
# ════════════════════════════════════════════════════════════════════════════════

def login_user(username, password):
    """Authenticate user"""
    users_db = load_users_db()
    for user in users_db.get("users", []):
        if user["username"] == username:
            if verify_password(password, user["password_hash"]):
                st.session_state.logged_in = True
                st.session_state.current_user = username
                st.session_state.user_role = user.get("role", "Viewer")
                log_activity(username, "LOGIN", "Successful login")
                return True, "Login successful!"
            else:
                log_activity(username, "LOGIN_FAILED", "Invalid password")
                return False, "Invalid password"
    
    log_activity("unknown", "LOGIN_FAILED", f"Unknown user: {username}")
    return False, "User not found"

def register_user(username, password, email, role="Viewer"):
    """Register new user"""
    users_db = load_users_db()
    
    for user in users_db.get("users", []):
        if user["username"] == username:
            return False, "Username already exists"
    
    users_db["users"].append({
        "username": username,
        "email": email,
        "password_hash": hash_password(password),
        "role": role,
        "created_at": datetime.now().isoformat(),
        "is_active": True
    })
    
    save_users_db(users_db)
    log_activity("admin", "USER_REGISTERED", f"User {username} registered")
    return True, "Registration successful!"

def get_all_users():
    """Get all users"""
    users_db = load_users_db()
    return users_db.get("users", [])

def delete_user(username):
    """Delete user"""
    users_db = load_users_db()
    users_db["users"] = [u for u in users_db.get("users", []) if u["username"] != username]
    save_users_db(users_db)
    log_activity(st.session_state.current_user, "USER_DELETED", f"User {username} deleted")

def update_user_role(username, new_role):
    """Update user role"""
    users_db = load_users_db()
    for user in users_db.get("users", []):
        if user["username"] == username:
            user["role"] = new_role
    save_users_db(users_db)
    log_activity(st.session_state.current_user, "ROLE_UPDATED", f"{username} → {new_role}")

# ════════════════════════════════════════════════════════════════════════════════
# UI PAGES
# ════════════════════════════════════════════════════════════════════════════════

def page_login():
    """Login/Registration page"""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 50px 0;'>
            <h1>🩺 MedPredict</h1>
            <p style='font-size: 1.2rem; color: #666;'>Multi-Disease Prediction System</p>
            <p style='color: #999;'>ML-Powered Clinical Diagnosis Assistant</p>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("User Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", use_container_width=True, type="primary"):
                success, msg = login_user(username, password)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        
        with tab2:
            st.subheader("New User Registration")
            new_username = st.text_input("Username", key="reg_user")
            new_email = st.text_input("Email", key="reg_email")
            new_password = st.text_input("Password", type="password", key="reg_pass")
            conf_password = st.text_input("Confirm Password", type="password", key="reg_conf")
            
            if st.button("Register", use_container_width=True):
                if not new_username or not new_password or not new_email:
                    st.error("All fields required")
                elif new_password != conf_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters")
                else:
                    success, msg = register_user(new_username, new_password, new_email)
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)

def page_patient_prediction():
    """TASK 3: Patient Prediction with Auto-Analysis"""
    st.title("🩺 Patient Disease Prediction")
    st.markdown("**Automatic analysis across all disease models** — Enter biomarker values for comprehensive disease risk assessment")
    
    col1, col2 = st.columns([1.5, 1], gap="large")
    
    with col1:
        st.subheader("📋 Patient Information")
        p_name = st.text_input("Patient Name", placeholder="e.g., Ali Hassan")
        col_age, col_gender = st.columns(2)
        with col_age:
            p_age = st.number_input("Age", 1, 120, 30)
        with col_gender:
            p_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        p_date = st.date_input("Test Date", value=date.today())
        
        st.divider()
        st.subheader("🧬 Biomarker Values")
        st.caption("Scale: 0.0 (minimum) → 0.5 (mid-range) → 1.0 (maximum)")
        
        input_vals = {}
        cols = st.columns(3)
        
        for i, feat in enumerate(RAW_FEATS):
            with cols[i % 3]:
                input_vals[feat] = st.slider(feat, 0.0, 1.0, 0.5, 0.01)
    
    with col2:
        st.subheader("📊 Quick Reference")
        st.info("""
        **Risk Level Indicators:**
        🟢 **NO RISK:** <20% confidence
        🟢 **LOW:** 20-50%
        🟡 **MODERATE:** 50-75%
        🔴 **HIGH:** >75%
        
        **Diseases Analyzed:**
        • 🩸 Anemia
        • 🧬 Thalassemia
        • 🍬 Diabetes
        • ❤️ Heart Disease
        • 🔴 Thrombocytopenia
        """)
    
    st.divider()
    
    # Predict button
    if st.button("🔍 Run Full Disease Analysis", type="primary", use_container_width=True):
        if not MODELS_LOADED:
            st.error("❌ Models not loaded")
            st.stop()
        
        with st.spinner("Running analysis across all disease models..."):
            predictions, error = predict_disease(input_vals)
        
        if error:
            st.error(f"Error: {error}")
            st.stop()
        
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        res_col, detail_col = st.columns([1.2, 1], gap="large")
        
        with res_col:
            top_disease = predictions["top_disease"]
            top_conf = predictions["top_confidence"]
            config = DISEASE_CONFIG[top_disease]
            
            # Proper risk thresholds
            if top_conf < 0.20:
                risk_color = "🟢"; risk_text = "NO SIGNIFICANT RISK"
                box_border = "#22c55e"; box_bg = "#f0fdf4"; txt_color = "#166534"
            elif top_conf < 0.50:
                risk_color = "🟢"; risk_text = "LOW RISK"
                box_border = "#22c55e"; box_bg = "#f0fdf4"; txt_color = "#166534"
            elif top_conf < 0.75:
                risk_color = "🟡"; risk_text = "MODERATE RISK"
                box_border = "#f59e0b"; box_bg = "#fffbeb"; txt_color = "#92400e"
            else:
                risk_color = "🔴"; risk_text = "HIGH RISK"
                box_border = "#ef4444"; box_bg = "#fef2f2"; txt_color = "#991b1b"
            
            if top_conf < 0.20:
                st.markdown(f"""
                <div style='background:{box_bg}; padding:20px; border-radius:10px; border-left:5px solid {box_border};'>
                    <h3 style='color:{txt_color};'>✅ No Significant Disease Detected</h3>
                    <h4 style='color:{txt_color};'>Highest Match: {top_conf*100:.1f}% ({config['name']})</h4>
                    <p><b>{risk_color} {risk_text}</b></p>
                    <p style='color:#555; margin-top:10px;'>All biomarker values appear within normal range. Please consult a doctor for clinical evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:{box_bg}; padding:20px; border-radius:10px; border-left:5px solid {box_border};'>
                    <h3 style='color:{txt_color};'>{config['emoji']} {config['name']}</h3>
                    <h4 style='color:{txt_color};'>Confidence: {top_conf*100:.1f}%</h4>
                    <p><b>{risk_color} {risk_text}</b></p>
                    <p style='color:#555; margin-top:10px;'>{config['description']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("**All Diseases Ranked:**")
            rank_data = []
            for disease_key, confidence in predictions["ranked"]:
                cfg = DISEASE_CONFIG[disease_key]
                if confidence < 0.20:
                    risk_label = "No Risk"
                elif confidence < 0.50:
                    risk_label = "Low"
                elif confidence < 0.75:
                    risk_label = "Moderate"
                else:
                    risk_label = "High"
                rank_data.append({
                    "Disease": f"{cfg['emoji']} {cfg['name']}",
                    "Confidence": f"{confidence*100:.1f}%",
                    "Risk": risk_label
                })
            
            st.dataframe(pd.DataFrame(rank_data), use_container_width=True, hide_index=True)
        
        with detail_col:
            st.subheader("📋 Key Biomarkers")
            summary_data = []
            for feat in config["key_features"]:
                if feat in input_vals:
                    summary_data.append({"Biomarker": feat, "Value": f"{input_vals[feat]:.2f}"})
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        # Medical reasoning
        st.markdown("---")
        st.subheader("🏥 Medical Reasoning")
        if top_conf < 0.20:
            st.success("✅ All biomarkers appear within normal range. No significant disease pattern detected. Consult a doctor for proper clinical evaluation.")
        else:
            reasoning = generate_medical_reasoning(top_disease, top_conf, input_vals)
            st.markdown(reasoning)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📈 Visualization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            diseases = [DISEASE_CONFIG[d[0]]["name"] for d in predictions["ranked"]]
            confidences = [d[1]*100 for d in predictions["ranked"]]
            colors_list = ["#ef4444" if c > 60 else ("#f59e0b" if c > 30 else "#22c55e") for c in confidences]
            ax.barh(diseases, confidences, color=colors_list, edgecolor="white")
            ax.set_xlabel("Confidence (%)")
            ax.set_title("Disease Prediction Ranking", fontweight="bold")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            key_features = config["key_features"][:5]
            key_values = [input_vals.get(f, 0.5) for f in key_features]
            ax.barh(key_features, key_values, color="#0066CC", edgecolor="white")
            ax.set_xlabel("Normalized Value")
            ax.set_xlim(0, 1)
            ax.set_title(f"Key Biomarkers: {config['name']}", fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
        
        # Save to history
        pred_record = {
            "timestamp": datetime.now().isoformat(),
            "patient_name": p_name,
            "patient_age": p_age,
            "patient_gender": p_gender,
            "predicted_disease": top_disease,
            "confidence": float(top_conf),
            "risk_level": risk_text,
            "biomarkers": {k: float(v) for k, v in input_vals.items()}
        }
        
        predictions_log = load_predictions_log()
        predictions_log["predictions"].append(pred_record)
        save_predictions_log(predictions_log)
        
        # PDF download
        st.markdown("---")
        pdf_buffer = generate_pdf_report(
            {"name": p_name, "age": p_age, "gender": p_gender, "test_date": p_date},
            input_vals,
            predictions
        )
        
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name=f"MedPredict_Report_{p_name}_{date.today()}.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        st.success("✅ Prediction saved to history!")

def page_admin_dashboard():
    """TASK 4: Admin Dashboard"""
    st.title("👨‍💼 Admin Dashboard")
    
    if st.session_state.user_role not in ["Super Admin", "Doctor"]:
        st.error("❌ Admin role required. Access denied.")
        return
    
    admin_tabs = st.tabs(["Overview", "Predictions History", "User Management", "Activity Logs", "Settings"])
    
    with admin_tabs[0]:  # Overview
        st.subheader("📊 System Overview")
        
        predictions_log = load_predictions_log()
        predictions = predictions_log.get("predictions", [])
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(predictions))
        
        with col2:
            unique_patients = len(set([p.get("patient_name", "Unknown") for p in predictions]))
            st.metric("Unique Patients", unique_patients)
        
        with col3:
            users = get_all_users()
            st.metric("Total Users", len(users))
        
        with col4:
            high_risk = len([p for p in predictions if p.get("confidence", 0) > 0.6])
            st.metric("High Risk Cases", high_risk)
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if predictions:
                disease_counts = {}
                for pred in predictions:
                    disease = pred.get("predicted_disease", "Unknown")
                    disease_counts[disease] = disease_counts.get(disease, 0) + 1
                
                fig, ax = plt.subplots(figsize=(8, 5))
                diseases = [DISEASE_CONFIG.get(d, {}).get("name", d) for d in disease_counts.keys()]
                counts = list(disease_counts.values())
                ax.bar(diseases, counts, color="#0066CC", edgecolor="white")
                ax.set_ylabel("Number of Cases")
                ax.set_title("Disease Distribution")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                st.pyplot(fig)
        
        with col2:
            if predictions:
                risk_counts = {"Low": 0, "Moderate": 0, "High": 0}
                for pred in predictions:
                    risk = pred.get("risk_level", "")
                    if "LOW" in risk:
                        risk_counts["Low"] += 1
                    elif "MODERATE" in risk:
                        risk_counts["Moderate"] += 1
                    else:
                        risk_counts["High"] += 1
                
                fig, ax = plt.subplots(figsize=(8, 5))
                colors = ["#22c55e", "#f59e0b", "#ef4444"]
                ax.bar(risk_counts.keys(), risk_counts.values(), color=colors, edgecolor="white")
                ax.set_ylabel("Number of Cases")
                ax.set_title("Risk Level Distribution")
                plt.tight_layout()
                st.pyplot(fig)
    
    with admin_tabs[1]:  # Predictions History
        st.subheader("📋 Predictions History")
        
        predictions_log = load_predictions_log()
        predictions = predictions_log.get("predictions", [])
        
        if predictions:
            col1, col2, col3 = st.columns(3)
            with col1:
                search_patient = st.text_input("Search patient name")
            with col2:
                filter_disease = st.selectbox("Filter disease", ["All"] + list(DISEASE_CONFIG.keys()))
            with col3:
                filter_risk = st.selectbox("Filter risk", ["All", "Low", "Moderate", "High"])
            
            filtered = []
            for pred in predictions:
                if search_patient and search_patient.lower() not in pred.get("patient_name", "").lower():
                    continue
                if filter_disease != "All" and pred.get("predicted_disease") != filter_disease:
                    continue
                if filter_risk != "All":
                    risk = "High" if pred.get("confidence", 0) > 0.6 else ("Moderate" if pred.get("confidence", 0) > 0.3 else "Low")
                    if risk != filter_risk:
                        continue
                filtered.append(pred)
            
            display_data = []
            for pred in filtered:
                display_data.append({
                    "Date": pred.get("timestamp", "")[:10],
                    "Patient": pred.get("patient_name", "N/A"),
                    "Disease": DISEASE_CONFIG.get(pred.get("predicted_disease", "Unknown"), {}).get("name", "Unknown"),
                    "Confidence": f"{pred.get('confidence', 0)*100:.1f}%",
                    "Risk": pred.get("risk_level", "Unknown")
                })
            
            st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
        else:
            st.info("No predictions yet.")
    
    with admin_tabs[2]:  # User Management
        st.subheader("👥 User Management")
        
        if st.session_state.user_role == "Super Admin":
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**All Users**")
                users = get_all_users()
                
                if users:
                    user_data = []
                    for user in users:
                        user_data.append({
                            "Username": user["username"],
                            "Email": user["email"],
                            "Role": user.get("role", "Viewer"),
                            "Active": user.get("is_active", True),
                            "Created": user.get("created_at", "")[:10]
                        })
                    
                    st.dataframe(pd.DataFrame(user_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No users found.")
            
            with col2:
                st.markdown("**Add New User**")
                new_user = st.text_input("Username")
                new_email = st.text_input("Email")
                new_pass = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["Super Admin", "Doctor", "Viewer"])
                
                if st.button("Add User"):
                    if new_user and new_email and new_pass:
                        success, msg = register_user(new_user, new_pass, new_email, new_role)
                        if success:
                            st.success("User added!")
                            st.rerun()
                        else:
                            st.error(msg)
                    else:
                        st.error("All fields required")
        else:
            st.info("Only Super Admin can manage users.")
    
    with admin_tabs[3]:  # Activity Logs
        st.subheader("📋 Activity Logs")
        
        logs = load_activity_log()
        activities = logs.get("activities", [])
        
        if activities:
            display_logs = []
            for activity in activities[-50:]:  # Last 50
                display_logs.append({
                    "Timestamp": activity.get("timestamp", "")[:19],
                    "User": activity.get("user", "N/A"),
                    "Action": activity.get("action", "N/A"),
                    "Details": activity.get("details", "")
                })
            
            st.dataframe(pd.DataFrame(display_logs), use_container_width=True, hide_index=True)
        else:
            st.info("No activity logs yet.")
    
    with admin_tabs[4]:  # Settings
        st.subheader("⚙️ Admin Settings")
        
        if st.session_state.user_role == "Super Admin":
            st.markdown("**System Configuration**")
            
            if st.button("🗑️ Clear Predictions Log"):
                save_predictions_log({"predictions": []})
                st.success("Predictions log cleared!")
                st.rerun()
            
            if st.button("🗑️ Clear Activity Log"):
                with open(ACTIVITY_LOG, 'w') as f:
                    json.dump({"activities": []}, f)
                st.success("Activity log cleared!")
                st.rerun()

def page_settings():
    """Settings page"""
    st.title("⚙️ Settings & Profile")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("👤 Current User")
        st.write(f"**Username:** {st.session_state.current_user}")
        st.write(f"**Role:** {st.session_state.user_role}")
    
    with col2:
        st.subheader("🎨 Preferences")
        theme = st.radio("Theme", ["Light", "Dark"], horizontal=True)
        st.session_state.theme = theme.lower()
    
    st.divider()
    
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.user_role = None
        log_activity(st.session_state.current_user or "unknown", "LOGOUT", "User logged out")
        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ════════════════════════════════════════════════════════════════════════════════

def main():
    """Main application entry point"""
    
    if not st.session_state.logged_in:
        page_login()
    else:
        # Sidebar navigation
        with st.sidebar:
            st.markdown(f"## 🩺 MedPredict")
            st.markdown(f"**User:** {st.session_state.current_user}")
            st.markdown(f"**Role:** {st.session_state.user_role}")
            st.divider()
            
            # Role-based navigation
            if st.session_state.user_role == "Viewer":
                pages = ["Patient Prediction", "Settings"]
            elif st.session_state.user_role == "Doctor":
                pages = ["Patient Prediction", "Admin Dashboard", "Settings"]
            else:  # Super Admin
                pages = ["Patient Prediction", "Admin Dashboard", "Settings"]
            
            selected = st.radio("Navigation", pages)
            
            st.divider()
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.current_user = None
                st.session_state.user_role = None
                st.rerun()
        
        # Main content based on selection
        if selected == "Patient Prediction":
            page_patient_prediction()
        elif selected == "Admin Dashboard":
            page_admin_dashboard()
        elif selected == "Settings":
            page_settings()

if __name__ == "__main__":
    main()

