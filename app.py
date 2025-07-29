import streamlit as st
import os
from ultralytics import YOLO
from PIL import Image
import pandas as pd

# Set konfigurasi halaman
st.set_page_config(page_title="KLASIFIKASI SISTEM ISYARAT BAHASA INDONESIA")

# Fungsi untuk memuat model YOLO
def load_model(model_path):
    try:
        model = YOLO(model_path)
        model.fuse()  # Siapkan model untuk inferensi
        return model
    except Exception as e:
        st.error("Error loading model.")
        st.error(f"Exception: {e}")
        st.stop()

def main():
    st.title("KLASIFIKASI SISTEM ISYARAT BAHASA INDONESIA Yolov8")
    st.sidebar.title("Settings")

    # Muat model YOLO
    model_name = "best.pt"
    pizza_model_path = os.path.join(model_name)
    pizza_classify = load_model(pizza_model_path)

    # Upload gambar dan setting confidence threshold
    uploaded_image = st.sidebar.file_uploader("Upload Gambar Terlebih Dahulu", type=["jpg", "jpeg", "png"])
    confidence_threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.05)

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # Membuat tiga kolom: kolom pertama untuk gambar, kolom tengah sebagai spacer, kolom ketiga untuk tabel
        col1, spacer, col2 = st.columns([1, 0.5, 1])
        
        with col1:
            st.image(image, caption='✅ Gambar sudah di upload.', width=300)
        
        with col2:
            with st.spinner('🤹🏽‍♀️ Sedang melakukan klasifikasi...'):
                try:
                    results = pizza_classify.predict(image)
                    any_class_above_threshold = False
                    results_tables = []

                    # Kumpulkan hasil klasifikasi yang memenuhi threshold
                    for result in results:
                        probs = result.probs.data.tolist()
                        names = result.names
                        filtered_results = [(names[i], probs[i]) for i in range(len(names)) if probs[i] >= confidence_threshold]
                        if filtered_results:
                            any_class_above_threshold = True
                            df_results = pd.DataFrame(filtered_results, columns=['Kelas', 'Kepercayaan'])
                            results_tables.append(df_results)

                    if any_class_above_threshold:
                        st.subheader("Hasil Klasifikasi SIBI:")
                        # Tampilkan setiap tabel hasil klasifikasi
                        for table_df in results_tables:
                            st.dataframe(table_df.style.format({'Kepercayaan': '{:.2f}'}))
                    else:
                        st.warning(f"Tidak ada kelas dengan confidence di atas {confidence_threshold}.")
                        
                    st.success('🎉 Klasifikasi berhasil!')
                except Exception as e:
                    st.error('❌ Terjadi kesalahan saat melakukan klasifikasi.')
                    st.error(e)
    else:
        st.markdown('Silahkan upload gambar.')

if __name__ == "__main__":
    main()
