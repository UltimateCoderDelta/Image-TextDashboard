import streamlit as st
from PIL import Image
import numpy as np

home_page = st.Page("pages/Home.py", title="Home", icon="🏠")
image_page = st.Page("pages/image_classification.py", title="Image Classification", icon="📸")
text_page = st.Page("pages/text_classification.py", title="Text Classification", icon="✏️")

pages = st.navigation([home_page, image_page, text_page])

st.set_page_config(
  page_title="Image & Text Classification, Dashboard",
  page_icon="🗂️",
  layout="centered",
  initial_sidebar_state="collapsed"
)
#random text
pages.run()
# #Add Instructions
# st.subheader("Select Classification Mode")
# #Add links to the pages in the side bar
# st.page_link("pages/text_classification.py", icon="✏️")
# st.page_link("pages/image_classification.py", icon="📸")
st.markdown('---')
st.markdown("""
    <footer style="display:flex; flex-direction:column;">
      <span><b><a href="#" style='text-decoration:none; color:#848884;'>About Octagon Technologies</a></b></span>
      <span><a href="#" style='text-decoration:none; color:#848884;'>Classification Tasks</a></span>
      </br>
      This dashboard was made with ❤️ in Amsterdam by the Octagon team.
    </footer>
  """, unsafe_allow_html=True)