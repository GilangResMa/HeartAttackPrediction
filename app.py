import streamlit as st
import joblib
import pandas as pd

# ==================== CONFIG ====================
st.set_page_config(
    page_title="Cek Tingkat Risiko Serangan Jantung",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="expanded"  # Input di sidebar biar lebih rapi
)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ==================== HEADER ====================
st.markdown("<h2 style='text-align: center;'>Cek Tingkat Risiko Serangan Jantung</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Aplikasi Machine Learning berbasis Random Forest</h3>", unsafe_allow_html=True)

st.info("<- Isi  di sebelah kiri")

# ==================== SIDEBAR INPUT (Lebih Rapi & Minimalis) ====================
with st.sidebar:
    st.header("📋 Data Pasien")
    
    age = st.slider("Umur (tahun)", 20, 100, 50)
    gender = st.radio("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    
    st.markdown("---")
    previous_heart_disease = st.selectbox("Pernah sakit jantung?", ["Tidak", "Ya"])
    hypertension = st.selectbox("Hipertensi / Darah Tinggi?", ["Tidak", "Ya"])
    diabetes = st.selectbox("Diabetes?", ["Tidak", "Ya"])
    
    st.markdown("---")
    smoking_status = st.selectbox("Merokok?", ["Tidak pernah", "Pernah (sudah berhenti)", "Masih merokok"])
    
    st.markdown("---")
    weight = st.number_input("Berat Badan (kg)", 30, 200, 70)
    height = st.number_input("Tinggi Badan (cm)", 120, 210, 170)
    waist_circ = st.number_input("Lingkar Pinggang (cm)", 60, 150, 90)
    cholesterol_total = st.slider("Kolesterol Total (mg/dL)", 100, 400, 200)
    fasting_blood_sugar = st.slider("Gula Darah Puasa (mg/dL)", 60, 300, 100)

# ==================== FEATURE ENGINEERING (Sama persis, tapi lebih aman) ====================
binary_map = {"Tidak": 0, "Ya": 1}
smoke_map = {"Tidak pernah": 0, "Pernah (sudah berhenti)": 1, "Masih merokok": 2}

# Hitung BMI
bmi = weight / ((height / 100) ** 2)
obesity = 1 if bmi >= 27.5 else 0  # Threshold Asia

# Metabolic syndrome proxy (tanpa trigliserida → pakai cholesterol >200)
metabolic_count = (
    binary_map[diabetes] +
    binary_map[hypertension] +
    obesity +
    (1 if cholesterol_total > 200 else 0)
)
metabolic_syndrome = 1 if metabolic_count >= 3 else 0

# Age group
if age < 30: age_group = 0
elif age < 45: age_group = 1
elif age < 60: age_group = 2
elif age < 80: age_group = 3
else: age_group = 4

# Cholesterol level category
if cholesterol_total < 200: chol_level = 0
elif cholesterol_total < 240: chol_level = 1
else: chol_level = 2

# DataFrame sesuai training (PASTIKAN URUTAN SAMA PERSIS DENGAN SAAT TRAINING!)
input_data = pd.DataFrame({
    "metabolic_syndrome_count": [metabolic_count],
    "previous_heart_disease": [binary_map[previous_heart_disease]],
    "hypertension": [binary_map[hypertension]],
    "metabolic_syndrome": [metabolic_syndrome],
    "diabetes": [binary_map[diabetes]],
    "obesity": [obesity],
    "smoking_status": [smoke_map[smoking_status]],
    "age_group": [age_group],
    "age": [age],
    "cholesterol_level": [chol_level],
    "fasting_blood_sugar": [fasting_blood_sugar],
    "BMI_est": [bmi],
    "waist_circumference": [waist_circ]
})

# ==================== TOMBOL PREDIKSI ====================
if st.button("🔍 Cek Risiko Sekarang", type="primary", use_container_width=True):
    with st.spinner("Sedang menganalisis..."):
        pred = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1] * 100

    st.markdown("## 📊 Hasil Pengecekan")

    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error("⚠️ **RISIKO TINGGI**")
            st.write(f"**{proba:.1f}%** kemungkinan terkena serangan jantung")
        else:
            st.success("✅ **RISIKO RENDAH**")
            st.write(f"Hanya **{proba:.1f}%** risiko")

    with col2:
        st.metric("Tingkat Risiko", f"{proba:.1f}%")

    st.warning("⚕️ **Penting:** Hasil ini hanya alat bantu. Konsultasi ke dokter untuk pemeriksaan lengkap.")

    # Bonus: Tampilkan BMI
    st.info(f"**BMI Anda:** {bmi:.1f} → {'Obesitas' if obesity else 'Normal/Berat Ideal'}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Cek Tingkat Risiko Serangan Jantung | Model Random Forest | Data Science 2025</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 14px;'>Andhika (123230008) - Gilang (123230060) - Najib (123230186)</p>", unsafe_allow_html=True)