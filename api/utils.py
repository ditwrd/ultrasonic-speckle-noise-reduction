import numpy as np
from PIL import Image, ImageOps


def open_image(image_buffer):
    """
        image_buffer : The image buffer from streamlit st.upload_file API

    Returns:
       A 2D numpy array containing the grayscaled image
    """
    rgba_image = Image.open(image_buffer)
    grayscale_image = ImageOps.grayscale(rgba_image)
    grayscale_image_array = np.array(grayscale_image)
    return grayscale_image_array
