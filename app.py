import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import joblib
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit UI setup
st.set_page_config(page_title='Diabetes Prediction', page_icon=':dna:')
st.markdown(f'<h1 style="text-align: center;">Diabetes Prediction</h1>', unsafe_allow_html=True)

# Load and preprocess data
df = pd.read_csv("./data/diabetes_prediction_dataset.csv")

encoder_path = "models/ordinal_encoder.pkl"
# Create directory for models
os.makedirs("models", exist_ok=True)

# --------- Gender Encoder ---------
gender_encoder_path = "models/gender_encoder.pkl"
if os.path.exists(gender_encoder_path):
    gender_enc = joblib.load(gender_encoder_path)
    df["gender"] = gender_enc.transform(df[["gender"]])
else:
    gender_enc = OrdinalEncoder()
    df["gender"] = gender_enc.fit_transform(df[["gender"]])
    joblib.dump(gender_enc, gender_encoder_path)

# --------- Smoking History Encoder ---------
smoking_encoder_path = "models/smoking_encoder.pkl"
if os.path.exists(smoking_encoder_path):
    smoking_enc = joblib.load(smoking_encoder_path)
    df["smoking_history"] = smoking_enc.transform(df[["smoking_history"]])
else:
    smoking_enc = OrdinalEncoder()
    df["smoking_history"] = smoking_enc.fit_transform(df[["smoking_history"]])
    joblib.dump(smoking_enc, smoking_encoder_path)



x = df.drop("diabetes", axis=1)
y = df["diabetes"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model_defs = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

models = {}
accuracies = {}

# Load or train and save models
for model_name, model in model_defs.items():
    model_file = f"models/{model_name.replace(' ', '_').lower()}.pkl"
    if os.path.exists(model_file):
        models[model_name] = joblib.load(model_file)
    else:
        model.fit(x_train, y_train)
        joblib.dump(model, model_file)
        models[model_name] = model

    y_pred = models[model_name].predict(x_test)
    accuracies[model_name] = metrics.accuracy_score(y_test, y_pred)

# Display model comparison
st.subheader("Model Comparison")
st.write("Accuracy scores for different models:")
accuracy_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
st.table(accuracy_df)

best_model_name = max(accuracies, key=accuracies.get)
best_model_score = accuracies[best_model_name]
st.success(f"💡 Best Model: **{best_model_name}** with Accuracy **{best_model_score:.4f}**")

# Input Form
st.subheader("Make a Prediction")
col1, col2 = st.columns(2, gap='large')

with col1:
    gender = st.selectbox(label='Gender', options=['Male', 'Female', 'Other'])
    age = st.text_input(label='Age')
    hypertension = st.selectbox(label='Hypertension', options=['No', 'Yes'])
    heart_disease = st.selectbox(label='Heart Disease', options=['No', 'Yes'])

with col2:
    smoking_history = st.selectbox(label='Smoking History',
                                   options=['Never', 'Current', 'Former', 'Ever', 'Not Current', 'No Info'])
    bmi = st.text_input(label='BMI')
    hba1c_level = st.text_input(label='HbA1c Level')
    blood_glucose_level = st.text_input(label='Blood Glucose Level')

# Categorical encoding (same as in training)
gender_dict = {'Female': 0.0, 'Male': 1.0, 'Other': 2.0}
hypertension_dict = {'No': 0, 'Yes': 1}
heart_disease_dict = {'No': 0, 'Yes': 1}
smoking_history_dict = {'No Info': 0.0, 'Current': 1.0, 'Ever': 2.0,
                        'Former': 3.0, 'Never': 4.0, 'Not Current': 5.0}

# Model Selection
selected_model = st.selectbox("Select a Model for Prediction", list(models.keys()))

# Prediction
st.write('')
st.write('')
col1, col2 = st.columns([0.438, 0.562])
with col2:
    submit = st.button(label='Submit')
st.write('')

if submit:
    try:
        user_data = np.array([[gender_dict[gender], age, hypertension_dict[hypertension],
                               heart_disease_dict[heart_disease], smoking_history_dict[smoking_history],
                               bmi, hba1c_level, blood_glucose_level]], dtype=float)

        model_path = f"models/{selected_model.replace(' ', '_').lower()}.pkl"
        model = joblib.load(model_path)
        test_result = model.predict(user_data)

        if test_result[0] == 0:
            col1, col2, col3 = st.columns([0.33, 0.30, 0.35])
            with col2:
                st.success('Diabetes Result: Negative')
            st.balloons()
        else:
            col1, col2, col3 = st.columns([0.215, 0.57, 0.215])
            with col2:
                st.error('Diabetes Result: Positive (Please Consult with Doctor)')
    except:
        st.warning('⚠️ Please fill in all fields with valid values.')
