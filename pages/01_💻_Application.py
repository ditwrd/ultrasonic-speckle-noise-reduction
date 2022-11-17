import streamlit as st
from streamlit_image_comparison import image_comparison

from api import logic, utils


def main():
    # Membuat Sidebar
    with st.sidebar.expander("About the App"):
        st.write(
            """
        Aplikasi ini dibuat untuk mereduksi Speckle Noise pada citra.  \n  \nDibuat oleh tim US GE. Hope you enjoy!
         """
        )

    # Memunculkan tombol file upload
    uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        c1, c2, c3 = st.columns(3)

        image = utils.open_image(uploaded_image)
        c1.subheader("Original Image")
        c1.image(image, width=300)
        filtered_image = logic.gaussian_fuzzy_filter("abc", image, 3)
        c2.subheader("Filtered Image")
        c2.image(filtered_image, width=300)

        noise = image - filtered_image
        c3.subheader("Noise")
        c3.image(noise, width=300)

        image_comparison(
            image, filtered_image, label1="Original Image", label2="Filtered Image"
        )


if __name__ == "__main__":
    main()
