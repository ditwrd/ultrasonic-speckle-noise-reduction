# Import Library
from turtle import width

import numpy as np
import streamlit as st
from PIL import Image

from api import logic, utils

# Membuat Judul halaman
st.markdown(
    """ <style> .font {font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """,
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="font" style="text-align: center;">Upload Image here...</p>',
    unsafe_allow_html=True,
)

# Membuat Sidebar
st.sidebar.markdown(
    '<p class="font">My First Photo Converter App</p>', unsafe_allow_html=True
)
with st.sidebar.expander("About the App"):
    st.write(
        """
    Aplikasi ini dibuat untuk mereduksi Speckle Noise pada citra.  \n  \nDibuat oleh tim US GE. Hope you enjoy!
     """
    )

# Memunculkan tombol file upload
file_uploader = st.file_uploader("", type=["jpg", "png", "jpeg"])

if file_uploader is not None:
    image = Image.open(file_uploader)
    # st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
    st.image(image, width=300)


def main():
    st.header("App")


if __name__ == "__main__":
    main()
