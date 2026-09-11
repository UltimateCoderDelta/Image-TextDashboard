import streamlit as st
import torch
st.header("Image Classification 📸")
st.write("""
  **Summary**:This section allows the user to classify images
  into various classes.""")
st.divider()

#Adding the various tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Single Image",
    "Classification History",
    "Results & analytics",
    "Load Model"
])

with tab2:
    st.subheader("My Favorite activity is gaming")
    #TODO Continue with building the app
    st.write("gaming football basketball")



