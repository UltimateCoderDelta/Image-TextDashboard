import streamlit as st
#TASK 1 - Introduction to the dashboard
st.title("Image & Text Classification Dashboard 🗂️")
#Provide a summary of the app
st.divider()
st.markdown("""**Summary:** This dashboard uses state-of-the-art machine learning models from Hugging Face (and custom-built models) for image and text classification.""")
st.markdown("""**PROJECT 001 - AI Classification Dashboard**\n 
**Status:** OPERATIONAL 🕷️""")
st.divider()
st.subheader("Choose Classification Mode")
st.page_link("pages/text_classification.py", icon="✏️")
st.page_link("pages/image_classification.py", icon="📷")
