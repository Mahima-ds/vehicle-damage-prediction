import streamlit as st
from model_helper import predict

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the image
    st.image(image_path, caption="Uploaded File")  # remove `use_container_width` if your Streamlit version is old

    # Run prediction
    prediction = predict(image_path)
    st.info(f"Predicted Class: {prediction}")
