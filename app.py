import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mo_hinh_nhan_dien_khuon_mat.h5")

IMG_SIZE = (224, 224)
 
@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    return model

model = load_model()

# =========================
# Tiền xử lý ảnh
# =========================
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# Giao diện Streamlit
# =========================
st.set_page_config(
    page_title="Nhận diện Con Người (CNN)",
    page_icon="",
    layout="centered"
)

st.title(" Nhận diện ảnh có phải con người hay không")
st.write("Upload ảnh để mô hình CNN dự đoán")

uploaded_file = st.file_uploader(
    "📤 Chọn ảnh",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Ảnh đã tải lên")

    if st.button(" Nhận diện"):
        with st.spinner("Đang phân tích..."):
            img_input = preprocess_image(image)
            prediction = model.predict(img_input)

            
            prob = float(prediction[0][0])

            if prob > 0.5:
                st.success(f" **Con người** (Độ tin cậy: {prob:.2%})")
            elif prob < 0.5:
                st.error(f" **Không phải con người** (Độ tin cậy: {(1-prob):.2%})")
            else:
                st.info("Kết quả không rõ ràng (Độ tin cậy: 50.00%)")

            st.write("###  Chi tiết dự đoán")
            st.write(f"Giá trị output model: {prob:.4f}")
