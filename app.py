import streamlit as st
import pickle
import numpy as np
import pandas as pd
from fpdf import FPDF
from io import BytesIO

# Load model + label encoder + symptoms
with open('model/dt_model.pkl', 'rb') as f:
    data = pickle.load(f)
    model = data['model']
    label_encoder = data['label_encoder']
    all_symptoms = data['symptoms']

# Load precautions dataset
precautions_df = pd.read_csv('data/disease_precautions.csv')
precaution_map = {
    row['Disease']: [row['Precaution_1'], row['Precaution_2'], row['Precaution_3'], row['Precaution_4']]
    for _, row in precautions_df.iterrows()
}

# Streamlit page config
st.set_page_config(page_title="AI Health Assistant", page_icon="🩺", layout="wide")

st.title("🩺 AI Health Assistant")
st.write("Select your symptoms to get top 2 predicted diseases and recommended precautions.")

# Sidebar
st.sidebar.header("ℹ️ About")
st.sidebar.write("This app predicts diseases based on selected symptoms using a trained RandomForest model.")
st.sidebar.write("Dataset: Kaggle Symptom-Disease Dataset")

# Keep report data persistent across reruns
if "report_data" not in st.session_state:
    st.session_state.report_data = []
    st.session_state.selected_symptoms = []

# User selects symptoms
selected_symptoms = st.multiselect(
    "Select your symptoms:",
    options=sorted(all_symptoms),
    help="Search and pick multiple symptoms"
)

if st.button("🔍 Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom")
    else:
        # Create input vector (0/1 for each symptom)
        input_vector = np.zeros(len(all_symptoms))
        for sym in selected_symptoms:
            if sym in all_symptoms:
                idx = all_symptoms.index(sym)
                input_vector[idx] = 1

        # Predict probabilities
        proba = model.predict_proba([input_vector])[0]
        top2_idx = proba.argsort()[-2:][::-1]

        st.session_state.report_data = []
        st.session_state.selected_symptoms = selected_symptoms

        st.subheader("Top 2 Predicted Diseases")
        for idx in top2_idx:
            disease = label_encoder.inverse_transform([idx])[0]
            confidence = proba[idx] * 100

            st.write(f"**{disease}**")
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.2f}%")

            if confidence > 5:  # Show precautions only if reasonably confident
                if disease in precaution_map:
                    st.markdown("**Precautions:**")
                    for p in precaution_map[disease]:
                        st.write(f"✅ {p}")
                    st.session_state.report_data.append((disease, confidence, precaution_map[disease]))
                else:
                    st.info("No specific precautions available.")

# Show Download button only if predictions exist
if st.session_state.report_data:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "AI Health Assistant Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Selected Symptoms: {', '.join(st.session_state.selected_symptoms)}")
    pdf.ln(5)

    for disease, conf, precs in st.session_state.report_data:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Disease: {disease} ({conf:.2f}% confidence)", ln=True)
        pdf.set_font("Arial", '', 12)
        for p in precs:
            pdf.multi_cell(0, 8, f"- {p}")
        pdf.ln(5)

    # Get PDF as binary data directly
    pdf_bytes = pdf.output(dest='S').encode('latin-1')

    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_bytes,
        file_name="AIHealth_Report.pdf",
        mime="application/pdf"
    )

else:
    st.info("👈 Select symptoms from the dropdown and click Predict.")