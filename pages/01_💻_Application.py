import streamlit as st
from streamlit_image_comparison import image_comparison

from api import logic, utils


def main():
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
    uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = utils.open_image(uploaded_image)
        st.image(image, width=300)
        filtered_image = logic.gaussian_fuzzy_filter("abc", image, 3)
        st.image(filtered_image, width=300)

        noise = image - filtered_image
        st.image(noise, width=300)

        image_comparison(
            image, filtered_image, label1="Original Image", label2="Filtered Image"
        )


if __name__ == "__main__":
    main()
