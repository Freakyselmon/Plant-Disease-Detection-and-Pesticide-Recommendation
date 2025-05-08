# webapp/streamlit_app.py
# This is the Streamlit app for the Crop Disease Detector. 
import streamlit as st
from app.preprocess import preprocess_image
from app.model_utils import predict_disease
from app.recommender import get_recommendation
from app.llama_utils import get_disease_info_llama_local


st.title("Crop Disease Detector")

uploaded_file = st.file_uploader("Upload a crop image")

if uploaded_file:
    image = preprocess_image(uploaded_file)
    disease = predict_disease(image)

    st.success(f"Detected Disease: {disease}")

    recommendation = get_recommendation(disease)

    if isinstance(recommendation, str):
        st.error(recommendation)
    else:
        st.subheader("Pesticide Recommendation")
        st.write("**Pesticide:**", recommendation["pesticide"])
        st.write("**Dosage:**", recommendation["dosage"])
        st.write("**Application Method:**", recommendation["application_method"])
        st.write("**Frequency:**", recommendation["frequency"])
        st.write("**Effective Against:**", recommendation["effective_against"])
        st.image(uploaded_file, caption="Uploaded Image", use_container_width =True)
        st.balloons()
        st.success("Recommendation generated successfully!")

# Predict disease
disease = predict_disease(image)
st.write(f"🦠 Detected Disease: **{disease}**")

# Get LLaMA 3 info
llama_info = get_disease_info_llama_local(disease)
st.subheader("🌿 Disease Information and Cure")
st.info(llama_info)