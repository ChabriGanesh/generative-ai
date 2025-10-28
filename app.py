import streamlit as st
import pandas as pd
import requests
st.title('Healthcare Synthetic Data Generator')
st.sidebar.header('Input Data & Model')
model_type = st.sidebar.selectbox('Model Type', ['CTGAN'])  
uploaded = st.file_uploader("Upload Patient Data (CSV)", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
    if st.button("Generate Synthetic Data"):
        response = requests.post(
            'http://localhost:5000/generate',
            json={"data": df.to_dict(orient='records'), "model_type": model_type}
        )
        synth_data = pd.DataFrame(response.json()['synthetic_data'])
        st.write("Synthetic Data Sample:", synth_data.head())
        st.download_button("Download Synthetic Data", synth_data.to_csv(index=False), "synthetic_data.csv")
