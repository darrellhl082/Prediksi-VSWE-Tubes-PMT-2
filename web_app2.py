import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# Konfigurasi Halaman
# ==========================================
st.set_page_config(
    page_title="Antenna VSWR Recommender (Level 2 - 8 Features)",
    page_icon="📡",
    layout="wide",
)

# ==========================================
# Load Model dan Scaler
# ==========================================
@st.cache_resource
def load_ml_components():
    # Load model dan scaler Level 2 (8 fitur)
    try:
        model = joblib.load("model_level2.pkl")
        scaler = joblib.load("scaler_level2.pkl")
        features = joblib.load("features_level2.pkl")
        return model, scaler, features
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}")
        st.info("Pastikan Anda sudah menjalankan script training Level 2.")
        return None, None, None

model, scaler, features = load_ml_components()

# ==========================================
# Judul dan Deskripsi
# ==========================================
st.title("📡 Antenna VSWR Recommender System")
st.markdown("""
**Sistem rekomendasi material dan dimensi antena berbasis Machine Learning (8 Fitur)**  
Model ini memprediksi kualitas VSWR berdasarkan parameter geometri, kondisi fisik, dan material.
- **Good Antenna**: VSWR ≤ 2 (Performa baik / *Matched*)
- **Poor Antenna**: VSWR > 2 (Performa buruk / *Mismatched*)
""")

st.divider()

# ==========================================
# Input User (8 Fitur Level 2)
# ==========================================
if model is not None:
    st.header("📝 Input Parameter Antenna")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📏 Dimensi Antenna")
        length = st.number_input(
            "Length (mm)", 
            min_value=30.0, max_value=60.0, value=45.0, step=0.01,
            help="Panjang patch antena dalam milimeter"
        )
        width = st.number_input(
            "Width (mm)", 
            min_value=20.0, max_value=50.0, value=35.0, step=0.01,
            help="Lebar patch antena dalam milimeter"
        )
        height = st.number_input(
            "Height/Thickness (mm)", 
            min_value=0.5, max_value=3.0, value=1.2, step=0.01,
            help="Ketebalan substrat antena dalam milimeter"
        )
    
    with col2:
        st.subheader("🔧 Kondisi Fisik & Feed")
        bend = st.number_input(
            "Bend (Derajat Tekukan)", 
            min_value=0.0, max_value=2.0, value=0.5, step=0.01,
            help="Tingkat deformasi/tekukan fisik pada struktur antena (0 = lurus, 2 = tekuk ekstrem)"
        )
        feed = st.number_input(
            "Feed Position (mm)", 
            min_value=-30.0, max_value=0.0, value=-15.0, step=0.01,
            help="Posisi titik umpan (feed point) relatif terhadap pusat antena. Sangat kritis untuk impedance matching."
        )
    
    with col3:
        st.subheader("🧪 Material Properties")
        epsilon_r = st.number_input(
            "epsilon_r (Relative Permittivity)", 
            min_value=1.0, max_value=5.0, value=2.5, step=0.01,
            help="Permitivitas relatif substrat material murni (dielektrik konstan). Contoh: FR-4 ≈ 4.4"
        )
        permittivity = st.number_input(
            "Permittivity (Effective)", 
            min_value=1.0, max_value=3.0, value=2.0, step=0.01,
            help="Permitivitas efektif yang sudah memperhitungkan efek fringing fields pada antena."
        )
        conductivity = st.number_input(
            "Conductivity (S/m)", 
            min_value=0.0, max_value=20000.0, value=10000.0, step=100.0,
            help="Konduktivitas material konduktor dalam Siemens per meter. Contoh: Tembaga ≈ 5.8×10⁷ S/m"
        )
    
    # ==========================================
    # Tombol Prediksi
    # ==========================================
    st.divider()
    
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: #1e3799;
            color: white;
            border-color: #1e3799;
        }
        div.stButton > button:first-child:hover {
            background-color: #0c2461;
            color: white;
            border-color: #0c2461;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    if st.button("🔮 Prediksi Performa Antenna", use_container_width=True):
        
        # Siapkan data input sesuai urutan fitur saat training
        input_data = pd.DataFrame([[
            length, width, height,
            bend, feed,
            epsilon_r, permittivity, conductivity
        ]], columns=features)
        
        # Tampilkan input user
        st.subheader("📊 Data Input")
        st.dataframe(input_data.T, column_config={0: "Nilai Parameter"}, hide_index=False)
        
        try:
            # Proses Normalisasi Fitur menggunakan Scaler yang dimuat
            input_scaled = scaler.transform(input_data)
            
            # Lakukan prediksi
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # ==========================================
            # Hasil Prediksi
            # ==========================================
            st.divider()
            st.header("🎯 Hasil Prediksi")
            
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.subheader("Kualitas Antenna")
                if prediction == 1:
                    st.success("✅ **GOOD ANTENNA**")
                    st.info("VSWR ≤ 2 - Antenna memiliki performa yang baik!")
                else:
                    st.error("❌ **POOR ANTENNA**")
                    st.warning("VSWR > 2 - Antenna memiliki performa yang kurang baik.")
            
            with res_col2:
                st.subheader("Probabilitas")
                prob_good = prediction_proba[1] * 100
                prob_poor = prediction_proba[0] * 100
                
                st.metric("Probabilitas Good", f"{prob_good:.2f}%")
                st.metric("Probabilitas Poor", f"{prob_poor:.2f}%")
                
                # Visualisasi probabilitas
                st.progress(prob_good / 100)
                st.caption(f"Keyakinan model: {max(prob_good, prob_poor):.2f}%")
            
            # ==========================================
            # Visualisasi Faktor Paling Berpengaruh
            # ==========================================
            st.divider()
            st.subheader("📊 Faktor Paling Berpengaruh (Feature Importances)")
            
            try:
                importances = model.feature_importances_
                indices = np.argsort(importances)
                sorted_features = [features[i] for i in indices]
                sorted_importances = importances[indices]
                
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_alpha(0.0)
                ax.set_alpha(0.0)
                ax.set_facecolor((0, 0, 0, 0))
                
                # Plotting dengan warna modern
                bars = ax.barh(sorted_features, sorted_importances, color="#2575fc", height=0.6)
                
                # Merapikan garis pembatas
                for spine in ["top", "right", "bottom"]:
                    ax.spines[spine].set_visible(False)
                ax.spines["left"].set_color("#4a5568")
                ax.spines["left"].set_linewidth(1.5)
                
                # Menambahkan label nilai di ujung bar
                max_val = max(sorted_importances)
                for bar in bars:
                    width_bar = bar.get_width()
                    ax.text(
                        width_bar + (max_val * 0.02),
                        bar.get_y() + bar.get_height() / 2,
                        f"{width_bar:.3f}",
                        va="center",
                        ha="left",
                        fontsize=10,
                        fontweight="medium",
                        color="#e2e8f0",
                    )
                
                ax.set_xlabel("Importance Score", fontsize=10, color="#a0aec0", labelpad=10)
                ax.tick_params(axis="x", colors="#a0aec0", labelsize=9)
                ax.tick_params(axis="y", colors="#e2e8f0", labelsize=11, pad=8)
                ax.set_xlim(0, max_val * 1.15)
                
                st.pyplot(fig, clear_figure=True, use_container_width=True)
                
            except AttributeError:
                st.info(
                    "💡 Model aktif tidak menggunakan basis pohon keputusan (tree-based), "
                    "sehingga grafik 'Feature Importance' tidak dapat ditampilkan."
                )
            
            # ==========================================
            # Rekomendasi
            # ==========================================
            st.divider()
            st.header("💡 Rekomendasi")
            
            if prediction == 0:
                st.markdown("""
                **Antenna diprediksi memiliki performa yang kurang baik. Pertimbangkan untuk:**
                - **Cek posisi Feed**: Geser feed position untuk mencapai impedance matching yang lebih baik.
                - **Kurangi Bend**: Tekukan fisik dapat mengubah panjang elektrik efektif antena.
                - **Optimasi dimensi**: Sesuaikan Length dan Width untuk mengkompensasi nilai epsilon_r.
                - **Pilih material**: Pertimbangkan substrat dengan permitivitas dan loss tangent yang lebih sesuai.
                - **Validasi ulang**: Lakukan simulasi EM penuh (CST/HFSS) untuk konfirmasi.
                """)
            else:
                st.markdown("""
                **Antenna diprediksi memiliki performa yang baik!**
                - Kombinasi 8 parameter (dimensi, kondisi fisik, dan material) yang dimasukkan dinilai sudah optimal.
                - Desain dapat dipertimbangkan untuk lanjut ke tahap fabrikasi prototipe.
                - Tetap direkomendasikan melakukan pengujian fisik (*laboratory testing*) untuk validasi final.
                - Dokumentasikan parameter ini untuk referensi desain mendatang.
                """)
            
            # ==========================================
            # Informasi Teknis
            # ==========================================
            with st.expander("📚 Informasi Teknis VSWR & Parameter"):
                st.markdown("""
                **VSWR (Voltage Standing Wave Ratio)** adalah indeks ukuran efisiensi transmisi daya.
                - **VSWR = 1**: *Perfect match* (Seluruh daya terpancar sempurna).
                - **VSWR ≤ 2**: *Good match* (Efisiensi transmisi daya > 90%, standar industri).
                - **VSWR > 2**: *Poor match* (Banyak pantulan, risiko *overheating* komponen).
                
                **Penjelasan 8 Fitur Level 2:**
                1. **Length, Width, Height**: Dimensi fisik antena yang menentukan frekuensi resonansi.
                2. **Bend**: Derajat tekukan yang memengaruhi panjang elektrik efektif.
                3. **Feed**: Posisi titik umpan yang sangat kritis untuk impedance matching.
                4. **epsilon_r**: Permitivitas relatif material substrat murni.
                5. **Permittivity**: Permitivitas efektif yang sudah memperhitungkan *fringing fields*.
                6. **Conductivity**: Konduktivitas material konduktor (memengaruhi rugi-rugi).
                
                **Model Machine Learning:**
                - Algoritma: Random Forest Classifier
                - Training: Stratified 5-Fold Cross-Validation
                - Target: Klasifikasi biner (VSWR ≤ 2 vs VSWR > 2)
                """)
        
        except Exception as e:
            st.error(f"⚠️ Terjadi kesalahan sistem: {str(e)}")
            st.info(
                "Pastikan arsitektur file 'model_level2.pkl' dan 'scaler_level2.pkl' "
                "cocok dengan 8 fitur input."
            )

else:
    st.warning("⚠️ Model belum dimuat. Silakan jalankan script training terlebih dahulu.")

# ==========================================
# Sidebar Informasi
# ==========================================
with st.sidebar:
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Sistem berbasis Machine Learning ini memprediksi kecenderungan performa VSWR 
    struktur antena menggunakan **8 Fitur Kritis (Level 2)**.
    
    **Fitur Input:**
    - **Dimensi**: Length, Width, Height
    - **Kondisi & Feed**: Bend, Feed Position
    - **Material**: epsilon_r, Permittivity, Conductivity
    
    **Alur Kerja Sistem:**
    1. Konfigurasi nilai parameter pada panel utama.
    2. Klik tombol **Prediksi Performa Antenna**.
    3. Sistem melakukan standarisasi z-score lalu mengevaluasi probabilitas via model.
    """)
    
    st.divider()
    
    st.header("📈 Nilai Referensi Umum")
    st.markdown("""
    **Dimensi Standar (mm):**
    - Length: 40.0 - 50.0
    - Width: 30.0 - 40.0
    - Height: 0.8 - 1.6
    
    **Kondisi Fisik & Feed:**
    - Bend: 0.1 - 1.2
    - Feed: -25.0 - -5.0
    
    **Karakteristik Material:**
    - epsilon_r: 1.5 - 4.0
    - Permittivity: 1.5 - 2.5
    - Conductivity: 3000 - 15000 S/m
    """)

# ==========================================
# Footer
# ==========================================
st.divider()
st.caption("Antenna VSWR Recommender System (Level 2 - 8 Features) | Dikembangkan dengan Streamlit | © 2026")