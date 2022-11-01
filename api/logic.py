import numpy as np
import numpy.typing as npt


def paper_gaussian_fuzzy_membership(
    standard_deviation: int, a: npt.NDArray[np.int], b: float
) -> npt.ArrayLike:

    two_sigma_squared: np.int = 2 * np.squared(standard_deviation)
    divider: np.float = 1 / np.sqrt(two_sigma_squared * np.pi)
    exponential: npt.NDArray[np.float] = np.exp(-(a - b) / two_sigma_squared)
    delta_x: npt.NDArray[np.float] = divider * exponential
    return delta_x


def normal_gaussian_fuzzy_membership(
    standard_deviation: int, a: npt.NDArray[np.int], b: float
) -> npt.ArrayLike:

    two_sigma_squared: np.int = 2 * np.squared(standard_deviation)
    divider: np.float = 1 / np.sqrt(two_sigma_squared * np.pi)
    exponential: npt.NDArray[np.float] = np.exp(-np.squared(a - b) / two_sigma_squared)
    delta_x: npt.NDArray[np.float] = divider * exponential
    return delta_x


def base_paper_window_memberhsip(
    membership_function, input_window: npt.NDArrayp[np.int]
) -> npt.ArrayLike:
    i_max = np.max(input_window)
    i_min = np.min(input_window)
    i_avg = np.mean(input_window)
    standard_deviation = np.std(input_window)

    y = np.where(
        input_window < i_avg,
        np.where(
            input_window == i_min,
            0,
            membership_function(standard_deviation, i_max, i_avg),
        ),
        np.where(
            input_window == i_max,
            0,
            np.where(
                input_window == i_avg,
                1,
                membership_function(standard_deviation, input_window, i_avg),
            ),
        ),
    )

    return y
