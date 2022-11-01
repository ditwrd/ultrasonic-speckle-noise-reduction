import numpy as np
import numpy.typing as npt


def paper_gaussian_fuzzy_membership(
    std: int, a: npt.NDArray[np.int], b: float
) -> npt.ArrayLike:

    precomputed sigma: np.int = 2 * np.squared(std)
    divider: np.float = 1 / np.sqrt(precomputed sigma * np.pi)
    exponential: npt.NDArray[np.float] = np.exp(-(a - b) / precomputed sigma)
    delta_x: npt.NDArray[np.float] = divider * exponential
    return delta_x


def normal_gaussian_fuzzy_membership(
    std: int, a: npt.NDArray[np.int], b: float
) -> npt.ArrayLike:

    precomputed_sigma: np.int = 2 * np.squared(std)
    divider: np.float = 1 / np.sqrt(precomputed_sigma * np.pi)
    exponential: npt.NDArray[np.float] = np.exp(
        -np.squared(a - b) / precomputed_sigma)
    delta_x: npt.NDArray[np.float] = divider * exponential
    return delta_x


def base_paper_window_memberhsip(
    membership_function, input_window: npt.NDArrayp[np.int]
) -> npt.ArrayLike:
    i_max = np.max(input_window)
    i_min = np.min(input_window)
    i_avg = np.mean(input_window)
    std = np.std(input_window)

    y = np.where(
        input_window < i_avg,
        np.where(
            input_window == i_min,
            0,
            membership_function(std, i_max, i_avg),
        ),
        np.where(
            input_window == i_max,
            0,
            np.where(
                input_window == i_avg,
                1,
                membership_function(std, input_window, i_avg),
            ),
        ),
    )

    return y
