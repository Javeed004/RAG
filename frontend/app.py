import streamlit as st
import requests


st.title("AI Agent")

if st.button("Ping Backend"):
    response = requests.get("http://127.0.0.1:8000/ping")

    if response.status_code == 200:
        data = response.json()
        st.success(data["status"])
    else:
        st.error("Backend request failed")