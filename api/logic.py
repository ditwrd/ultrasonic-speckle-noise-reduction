import numpy as np
import numpy.typing as npt
from stqdm import stqdm


def _normal_gaussian_fuzzy_membership(
    std: int, a: npt.NDArray[np.int], b: float
) -> npt.NDArray[np.float_]:

    precomputed_sigma: np.int = 2 * np.square(std) + 10e-5

    # Check whether sigma is 0
    # if precomputed_sigma == 0:
    #     raise Exception("Computed sigma/standard deviation is 0")

    divider: np.float = 1 / np.sqrt(precomputed_sigma * np.pi)
    exponential: npt.NDArray[np.float_] = np.exp(-np.square(a - b) / precomputed_sigma)
    delta_x: npt.NDArray[np.float_] = divider * exponential
    return delta_x


def _window_membership(image_window: npt.NDArray[np.int_]) -> npt.ArrayLike:
    i_avg: np.float_ = np.mean(image_window)
    std: np.float_ = np.std(image_window)

    # image_window = np.where(image_window < i_avg, 0, image_window)

    # Calculate fuzzy membership
    membership = _normal_gaussian_fuzzy_membership(std, image_window, i_avg)
    # Get the 2D coordinate of the max value
    row_max, col_max = np.unravel_index(membership.argmax(), membership.shape)
    # Translate the coord from membership to the image window
    image_val_max_member = image_window[row_max, col_max]

    return image_val_max_member


def gaussian_fuzzy_filter(
    padding_mode: str, image_array: npt.NDArray[np.int_], width: int = 3
):
    # Width should be odd
    if width % 2 != 1:
        raise Exception("Window width need to be odd")

    padded_image: npt.NDArray[np.int_] = np.pad(image_array, (width - 1) // 2)
    image_array_processed = np.copy(image_array)
    n_row, n_col = image_array.shape
    # For loop is unoptimizable due to the fact that the process seems to be  sequential
    for i_row in stqdm(range(n_row)):
        for i_col in range(n_col):
            image_array_processed[i_row, i_col] = _window_membership(
                padded_image[i_row : i_row + width, i_col : i_col + width]
            )
    return image_array_processed
