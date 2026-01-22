import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(page_title="Smart Loan Approval System", layout="centered")

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------------------
# Title & Description
# ---------------------------
st.title("💳 Smart Loan Approval System")

st.write("""
This system uses **Support Vector Machine (SVM)** models to predict  
whether a loan should be **Approved or Rejected** based on applicant details.
""")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
    return df

df = load_data()

# ---------------------------
# Data Preprocessing
# ---------------------------
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

features = [
    'ApplicantIncome',
    'LoanAmount',
    'Credit_History',
    'Self_Employed',
    'Property_Area'
]

X = df[features]
y = df['Loan_Status'].map({'Y': 1, 'N': 0})

X = pd.get_dummies(X, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("📋 Applicant Details")

app_income = st.sidebar.number_input("Applicant Income", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=100)
credit_history = st.sidebar.selectbox("Credit History", ["Yes", "No"])
employment = st.sidebar.selectbox("Employment Status", ["Yes", "No"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# ---------------------------
# Model Selection
# ---------------------------
st.sidebar.header("⚙️ Model Selection")

kernel_choice = st.sidebar.radio(
    "Choose SVM Kernel",
    ["Linear SVM", "Polynomial SVM", "RBF SVM"]
)

# ---------------------------
# Prediction Button
# ---------------------------
if st.button("🔍 Check Loan Eligibility"):

    # Prepare input
    input_data = {
        'ApplicantIncome': app_income,
        'LoanAmount': loan_amount,
        'Credit_History': 1 if credit_history == "Yes" else 0,
        'Self_Employed_Yes': 1 if employment == "Yes" else 0,
        'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
        'Property_Area_Urban': 1 if property_area == "Urban" else 0
    }

    input_df = pd.DataFrame([input_data])

    # Align columns
    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    input_scaled = scaler.transform(input_df)

    # Choose model
    if kernel_choice == "Linear SVM":
        model = SVC(kernel='linear', probability=True)
    elif kernel_choice == "Polynomial SVM":
        model = SVC(kernel='poly', degree=3, probability=True)
    else:
        model = SVC(kernel='rbf', probability=True)

    model.fit(x_train_scaled, y_train)

    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled)[0][prediction]

    # ---------------------------
    # Output Section
    # ---------------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.write(f"**Kernel Used:** {kernel_choice}")
    st.write(f"**Confidence Score:** {confidence:.2f}")

    # ---------------------------
    # Business Explanation
    # ---------------------------
    st.info(
        "📊 **Decision Explanation:**\n\n"
        "Based on the applicant’s **credit history**, **income pattern**, "
        "and **loan amount**, the model estimates the applicant’s "
        "ability to repay the loan."
    )
