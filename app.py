import streamlit as st
import joblib
import pandas as pd

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Prediksi Risiko Serangan Jantung",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ==================== HEADER ====================
st.markdown("<h1 style='text-align: center;'>Prediksi Risiko Serangan Jantung</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Aplikasi Machine Learning berbasis Random Forest</h3>", unsafe_allow_html=True)
st.info("Silakan isi data pasien dengan benar untuk mendapatkan hasil prediksi yang akurat.")

# ==================== INPUT FORM ====================
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Umur (tahun)", 20, 100, 45)
    previous_heart_disease = st.selectbox("Pernah sakit jantung sebelumnya?", ["Tidak", "Ya"])
    hypertension = st.selectbox("Hipertensi / Darah Tinggi?", ["Tidak", "Ya"])
    diabetes = st.selectbox("Diabetes?", ["Tidak", "Ya"])
    obesity = st.selectbox("Obesitas (IMT ≥ 30)?", ["Tidak", "Ya"])

with col2:
    smoking_status = st.selectbox("Status Merokok", 
                                  ["Tidak pernah", "Pernah (sudah berhenti)", "Masih merokok"])
    cholesterol_level = st.slider("Kadar Kolesterol Total (mg/dL)", 100, 400, 200)
    fasting_blood_sugar = st.slider("Gula Darah Puasa (mg/dL)", 60, 300, 100)
    waist_circumference = st.slider("Lingkar Pinggang (cm)", 60, 150, 90)

st.markdown("---")

# ==================== MAPPING ====================
mapping_binary = {"Tidak": 0, "Ya": 1}
mapping_smoking = {"Tidak pernah": 0, "Pernah (sudah berhenti)": 1, "Masih merokok": 2}

# ==================== PREDICT BUTTON ====================
if st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True):
    # Buat DataFrame sesuai urutan feature waktu training
    input_df = pd.DataFrame({
        'previous_heart_disease': [mapping_binary[previous_heart_disease]],
        'hypertension': [mapping_binary[hypertension]],
        'diabetes': [mapping_binary[diabetes]],
        'obesity': [mapping_binary[obesity]],
        'smoking_status': [mapping_smoking[smoking_status]],
        'age': [age],
        'cholesterol_level': [cholesterol_level],
        'fasting_blood_sugar': [fasting_blood_sugar],
        'waist_circumference': [waist_circumference]
    })

    # Prediksi
    prediksi = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]

    # Hasil
    st.markdown("## Hasil Prediksi")

    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        if prediksi == 1:
            st.error("BERISIKO TINGGI")
            st.write(f"**{prob[1]*100:.1f}%** kemungkinan terkena serangan jantung")
        else:
            st.success("RISIKO RENDAH")
            st.write(f"Hanya **{prob[1]*100:.1f}%** risiko")

    with colB:
        st.metric("Tidak Berisiko", f"{prob[0]*100:.1f}%")
    with colC:
        st.metric("Berisiko", f"{prob[1]*100:.1f}%")

    st.warning("Peringatan: Ini hanya alat bantu. Konsultasikan hasil dengan dokter untuk diagnosis resmi.")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Prediksi Serangan Jantung | Model Random Forest | Data Science 2025</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Andhika - Gilang - Najib</p>", unsafe_allow_html=True)