import streamlit as st
import requests
from PIL import Image
import io

FLASK_URL = "http://127.0.0.1:5000/predict"

def main():
    st.title("Product Type Prediction")
    st.write("Upload an image to predict the product type.")
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes = image_bytes.getvalue()
        if st.button("Predict"):
            st.write("Predicting...")
            response = requests.post(FLASK_URL, files={"image": image_bytes}) 
            if response.status_code == 200:
                prediction = response.json()
                st.write("Prediction:")
                st.json(prediction)
            else:
                st.write("Error:", response.text)

if __name__ == "__main__":
    main()
