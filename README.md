# 🩺 MedPredict - Multi-Disease Prediction System

A comprehensive machine learning-powered disease prediction system built with Python, Streamlit, and scikit-learn. Features intelligent auto-prediction across 5 diseases, professional admin panel with authentication, and complete PDF report generation.

## 📋 Project Overview

**All 4 Major Tasks Completed:**

### ✅ TASK 1: Feature Selection References
- Found authentic medical references from PubMed, IEEE, Springer, ScienceDirect, WHO, ADA, AHA
- Created comprehensive documentation for all 40+ feature-disease pairs
- All references in **APA 7th edition format**
- Added to notebook Section 5: Feature Selection References
- **File:** `Medical_References_FeatureSelection.md` (auto-generated)
- **Notebook Updated:** `notebook.ipynb` with full feature justification section

### ✅ TASK 2: Intelligent Auto-Prediction Engine
- **Automatic analysis** of patient data across ALL disease models simultaneously
- Runs through 5 binary disease classifiers automatically
- Returns ranked predictions with confidence scores
- **Medical reasoning generation** explaining why each prediction was made
- No user selection needed - system automatically determines top predicted disease
- **File:** `app.py` - `predict_disease()` function

### ✅ TASK 3: Complete Streamlit GUI
- **Patient Input Form** with all biomarkers (normalized 0-1 scale)
- **Intelligent Results Display:**
  - Top predicted disease with confidence percentage
  - Color-coded risk levels (🟢 Low, 🟡 Moderate, 🔴 High)
  - Ranked list of all diseases by prediction confidence
  - Medical reasoning for each prediction
- **Visualization:**
  - Disease prediction ranking bar chart
  - Key biomarker values visualization
  - Risk level distribution (admin dashboard)
  - Disease distribution (admin dashboard)
- **PDF Report Generation:**
  - Patient information
  - Predicted disease with confidence
  - Input biomarker values
  - Risk level
  - Professional formatting using ReportLab
  - Download button for easy sharing

### ✅ TASK 4: Professional Admin Panel
- **Authentication System:**
  - Secure login page
  - User registration system
  - bcrypt password hashing (not plain text)
  - Session management via Streamlit session_state
  
- **Role-Based Access Control:**
  - **Super Admin:** Full system access, user management, configuration
  - **Doctor:** Patient predictions + admin dashboard + history
  - **Viewer:** Patient predictions only
  
- **Admin Dashboard:**
  - System overview with key metrics
  - Total predictions count
  - Disease-wise breakdown
  - Risk level distribution charts
  - High-risk case tracking
  
- **Predictions History:**
  - Searchable by patient name
  - Filterable by disease
  - Filterable by risk level
  - Timestamped records
  
- **User Management:**
  - Add/edit/delete users (Super Admin only)
  - Role assignment
  - User status tracking
  - Created date tracking
  
- **Activity Logs:**
  - Login/logout tracking
  - User registration events
  - User role changes
  - Administrative actions
  - Timestamps for all activities

- **Professional Sidebar Navigation:**
  - Current user display
  - Current role display
  - Role-based menu options
  - Logout button
  
- **Light/Dark Theme Toggle** (settings page)

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Or manually:**

```bash
pip install streamlit joblib pandas numpy matplotlib seaborn scikit-learn bcrypt reportlab scipy kagglehub
```

### Step 2: Ensure Model Files Exist

The following `.pkl` files must be in the project directory:
- `scaler.pkl`
- `label_encoder.pkl`
- `anemia_model.pkl`
- `thalassemia_model.pkl`
- `diabetes_model.pkl`
- `heart_disease_model.pkl`
- `thrombocytopenia_model.pkl`
- `multiclass_model.pkl`

**Generate these by running the Jupyter notebook:**
```bash
jupyter notebook notebook.ipynb
```

The notebook trains all models and exports them as `.pkl` files.

### Step 3: Run the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

---

## 📖 User Guide

### For Patients/Doctors (Viewer/Doctor Role)

1. **Login:**
   - New users: Click "Register" tab
   - Create account with username, email, password
   - Login with credentials

2. **Run Prediction:**
   - Enter patient information (name, age, gender, date)
   - Adjust biomarker sliders (0.0 = low, 0.5 = normal, 1.0 = high)
   - Click "Run Full Disease Analysis"
   - System automatically analyzes all disease models

3. **View Results:**
   - See top predicted disease with confidence
   - View all diseases ranked by risk
   - Read medical reasoning
   - See visualization charts
   - Download PDF report

### For Administrators (Doctor/Super Admin Role)

1. **Dashboard Overview:**
   - View total predictions
   - See unique patients count
   - Check active users
   - Monitor high-risk cases

2. **Predictions History:**
   - Search by patient name
   - Filter by disease
   - Filter by risk level
   - Download records

3. **User Management (Super Admin only):**
   - View all users
   - Add new users
   - Edit user roles
   - Delete users

4. **Activity Logs:**
   - View all system activities
   - Track user logins
   - Monitor administrative actions
   - Timestamped records

---

## 🔐 Security Features

- **bcrypt Password Hashing:** Passwords securely hashed, never stored in plain text
- **Session Management:** Uses Streamlit session_state for authentication
- **Role-Based Access Control:** Strict permission checks for admin features
- **Activity Logging:** All administrative actions logged with timestamps
- **Secure Logout:** Clears session state on logout

---

## 🏥 Diseases Supported

| Disease | Key Biomarkers | Risk Factors |
|---------|----------------|------------|
| **🩸 Anemia** | Hemoglobin, RBC, Hematocrit, MCV | Low oxygen-carrying capacity |
| **🧬 Thalassemia** | RBC, MCV, MCH, Hemoglobin | Genetic blood disorder |
| **🍬 Diabetes** | Glucose, HbA1c, Insulin, BMI | Metabolic dysfunction |
| **❤️ Heart Disease** | BP, Cholesterol, Troponin, CRP | Cardiovascular risk |
| **🔴 Thrombocytopenia** | Platelets, WBC, Hemoglobin | Low platelet count |

---

## 📊 Medical References

All 40+ feature-disease pairs validated against:
- **Textbooks:** Harrison's, Williams Hematology, Mosby's
- **Clinical Guidelines:** WHO, ADA, AHA
- **Journals:** Circulation, NEJM, Diabetes Care, Blood
- **Studies:** Framingham Heart Study, UKPDS

See `Medical_References_FeatureSelection.md` for complete references in APA 7th edition format.

---

## 📁 Project Structure

```
.
├── app.py                                    # Main Streamlit application
├── notebook.ipynb                            # Model training notebook
├── scaler.pkl                                # Feature scaler
├── label_encoder.pkl                         # Class encoder
├── anemia_model.pkl                          # Anemia binary classifier
├── thalassemia_model.pkl                     # Thalassemia binary classifier
├── diabetes_model.pkl                        # Diabetes binary classifier
├── heart_disease_model.pkl                   # Heart disease binary classifier
├── thrombocytopenia_model.pkl                # Thrombocytopenia binary classifier
├── multiclass_model.pkl                      # Multi-class classifier
├── users_database.json                       # User accounts (auto-generated)
├── predictions_log.json                      # Prediction history (auto-generated)
├── activity_log.json                         # Activity logs (auto-generated)
├── Medical_References_FeatureSelection.md    # Feature references
└── README.md                                 # This file
```

---

## 🔧 Configuration

### Database Files (Auto-Generated)

1. **users_database.json**
   - Stores registered user accounts
   - Contains hashed passwords (bcrypt)
   - Stores user roles and metadata

2. **predictions_log.json**
   - Stores all prediction results
   - Patient info + biomarkers + predictions
   - Timestamped entries

3. **activity_log.json**
   - Logs all system activities
   - Login/logout events
   - Administrative actions

---

## 🧪 Testing the Application

### Test Accounts

1. **Create Super Admin:**
   - Register with username: `admin`
   - Password: `admin123`
   - Role: Super Admin

2. **Create Doctor Account:**
   - Register with username: `doctor1`
   - Password: `doctor123`
   - Role: Doctor

3. **Create Viewer Account:**
   - Register with username: `viewer1`
   - Password: `viewer123`
   - Role: Viewer

### Test Prediction

1. Login as Doctor or Super Admin
2. Click "Patient Prediction"
3. Enter sample values:
   - Patient Name: "Test Patient"
   - Age: 45
   - Gender: Male
   - Biomarkers: Use default (0.5) or adjust as needed
4. Click "Run Full Disease Analysis"
5. View results, medical reasoning, and download PDF

---

## 📝 Feature Details

### Biomarker Input (0-1 Normalized Scale)

The system uses **normalized values (0.0 to 1.0)** matching the training dataset:

- **0.0** = Minimum clinical value (e.g., Hemoglobin = 5 g/dL)
- **0.5** = Mid-range clinical value (e.g., Hemoglobin = 13.5 g/dL)
- **1.0** = Maximum clinical value (e.g., Hemoglobin = 18 g/dL)

**Available Biomarkers (24 total):**
- Glucose, Cholesterol, Hemoglobin, Platelets, WBC, RBC
- Hematocrit, MCV, MCH, MCHC, Insulin, BMI
- Systolic BP, Diastolic BP, Triglycerides, HbA1c
- LDL, HDL, ALT, AST, Heart Rate, Creatinine, Troponin, CRP

### Engineered Features (Computed Automatically)

- **HGB_RBC_ratio** = Hemoglobin / RBC (anemia type indicator)
- **Glucose_Chol_ratio** = Glucose / Cholesterol (metabolic risk)
- **WBC_PLT_ratio** = WBC / Platelets (immune stress)

---

## ⚠️ Disclaimer

**This system is for research and educational purposes ONLY.**

It is **NOT** a substitute for:
- Professional medical diagnosis
- Clinical judgment by healthcare professionals
- Laboratory testing or imaging
- Professional medical advice

Always **consult a qualified healthcare professional** for medical decisions.

---

## 📄 License

Educational Project for Final Year Project (FYP 2026)

---

## 🤝 Support

For issues, questions, or feature requests, please refer to the project documentation or contact your project supervisor.

---

## 🎓 Acknowledgments

- **Dataset:** Kaggle - Multiple Disease Prediction Dataset (ehababoelnaga)
- **Framework:** Streamlit, scikit-learn, pandas
- **References:** WHO, ADA, AHA, peer-reviewed medical journals
- **Models:** Gradient Boosting Classifiers (scikit-learn)

---

**Last Updated:** May 12, 2026  
**Version:** 1.0 - Complete Release  
**Status:** ✅ All Tasks Completed
