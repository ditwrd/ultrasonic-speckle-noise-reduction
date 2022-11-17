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
        with st.form("Filter Input"):
            ca, cb = st.columns(2)
            padding_mode = ca.selectbox(
                "Padding Mode",
                [
                    "constant",
                    "edge",
                    "linear_ramp",
                    "maximum",
                    "minimum",
                    "mean",
                    "median",
                    "minimum",
                    "reflect",
                    "symmetric",
                    "wrap",
                ],
            )
            kernel_width = cb.number_input(
                "Kernel Width", min_value=3, max_value=101, step=2, value=3
            )

            submit_button = st.form_submit_button("Process Image")

            if submit_button:

                c1, c2, c3 = st.columns(3)

                image = utils.open_image(uploaded_image)
                c1.subheader("Original Image")
                c1.image(image, use_column_width=True)
                filtered_image = logic.gaussian_fuzzy_filter(
                    padding_mode, image, kernel_width
                )
                c2.subheader("Filtered Image")
                c2.image(filtered_image, use_column_width=True)

                noise = image - filtered_image
                c3.subheader("Noise")
                c3.image(noise, use_column_width=True)

                psnr = logic.psnr(image, filtered_image)
                st.markdown(f"PSNR = {psnr} dB")
                image_comparison(
                    image,
                    filtered_image,
                    label1="Original Image",
                    label2="Filtered Image",
                )


if __name__ == "__main__":
    st.set_page_config(page_title="Application", page_icon="💻")
    main()
