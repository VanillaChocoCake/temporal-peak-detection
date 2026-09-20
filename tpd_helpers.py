"""Stateless helper functions: BPM reshaping, median filter, WLC sensor.

Small utilities the estimator and pipeline layers call:

- ``reshape`` / ``reshape_aligned``: fold a 1-D BPM sample stream into
  ``(num_batches, batch_size)`` windows, optionally aligned to a
  reference shape for cross-rate longitudinal data.
- ``online_median_filter``: thin wrapper over ``np.median``, kept as a
  single call site so it can be swapped for a numba kernel.
- ``weighted_linear_combination``: the WLC sensor (amplitude vs.
  distance-weighted peak picker).
- ``normalize_to_0_1``: internal normaliser used by the WLC sensor.
"""

import numpy as np

from tune_pipeline.dsp.helpers import find_local_maxima


def reshape(arr: np.ndarray,
            batch_size: int) -> np.ndarray:
    """
    Reshape a one-dimensional array into fixed-size batches.

    Divides the input array into consecutive batches of ``batch_size``
    samples; each row is one analysis window for the subsequent spectral
    processing.

    Parameters:
        arr (np.ndarray): Input one-dimensional array
        batch_size (int): Size of each batch

    Returns:
        np.ndarray: Two-dimensional array of shape (num_batches, batch_size)

    Notes:
        - If the array length is not divisible by batch_size, the tail is
          zero-padded up to a whole number of batches.
    """
    n = arr.size
    remainder = n % batch_size

    if remainder != 0:
        target_size = n + batch_size - remainder
    else:
        target_size = n
    result = np.zeros(target_size)
    result[: n] = arr
    num_batches = target_size // batch_size
    return result.reshape((num_batches, batch_size))


def reshape_aligned(reference_shape: tuple[int, int], f_sampling_ref: float, array: np.ndarray, f_sampling: float) -> np.ndarray:
    """
    Reshape an array so that its time axis lines up with a reference array.

    The reference array was reshaped to ``reference_shape`` at sampling
    rate ``f_sampling_ref``; the column count is rescaled by the
    sampling-rate ratio so that a row of the result spans the same time
    interval as a row of the reference.

    Parameters:
    - reference_shape: (rows, columns) of the reshaped reference array
    - f_sampling_ref: Sampling rate of the reference array, Hz
    - array: Input one-dimensional array to reshape
    - f_sampling: Sampling rate of ``array``, Hz

    Returns:
    - b_reshaped: ``array`` reshaped to (reference_shape[0], n_cols), zero-
      padded or truncated as needed, aligned in time with the reference
    """
    # Number of samples per row at this sampling rate, so that a row spans
    # the same time interval as a row of the reference
    n_cols = int(reference_shape[1] * (f_sampling / f_sampling_ref))

    n_total = reference_shape[0] * n_cols

    if array.shape[0] < n_total:
        b_adjusted = np.zeros(n_total, dtype=array.dtype)
        b_adjusted[:array.shape[0]] = array
    else:
        b_adjusted = array[:n_total]

    b_reshaped = b_adjusted.reshape(reference_shape[0], n_cols)

    return b_reshaped


def normalize_to_0_1(array: np.ndarray) -> np.ndarray:
    """
    Normalize array values to the range [0, 1].

    Linearly scales the input array so that the minimum value becomes 0
    and the maximum value becomes 1. If the array is empty or has constant
    values (min=max), returns an array of zeros with the same shape.

    Parameters:
        array (np.ndarray): Input array to be normalized

    Returns:
        np.ndarray: Normalized array with values in range [0, 1]
    """
    if array.shape[0] == 0 or np.max(array) == np.min(array):
        return np.zeros_like(array)
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def weighted_linear_combination(tune_unit: np.ndarray,
                                psd: np.ndarray,
                                center: float,
                                alpha: float) -> tuple[float, float]:
    """
    Identify the most significant frequency component using weighted criteria.

    Combines amplitude and proximity weighting to identify the most important
    peak in a power spectral density, balancing between the highest amplitude
    peaks and those closest to the expected center frequency.

    Parameters:
        tune_unit (np.ndarray): Array of frequency values (tune units)
        psd (np.ndarray): Power spectral density values corresponding to tune_unit
        center (float): Expected central frequency for proximity weighting
        alpha (float): Weighting factor balancing amplitude vs. distance importance
                      (0 = only distance matters, 1 = only amplitude matters)

    Returns:
        tuple[float, float]: (best_frequency, confidence)
            - best_frequency: The tune value of the most significant peak
            - confidence: Confidence score (0-1) for the identified peak
    """
    index_bool_maxima, _ = find_local_maxima(psd)

    if not np.any(index_bool_maxima):
        return center, float(0.0)

    # Extract the maxima — numba-friendly implementation
    peak_indices = np.where(index_bool_maxima)[0]
    peak_tunes = np.zeros(len(peak_indices), dtype=np.float64)
    peak_psds = np.zeros(len(peak_indices), dtype=np.float64)

    for i, idx in enumerate(peak_indices):
        peak_tunes[i] = tune_unit[idx]
        peak_psds[i] = psd[idx]

    weight_amplitude = normalize_to_0_1(peak_psds)
    distance = normalize_to_0_1(np.abs(peak_tunes - center))
    weight_distance = 1 - distance

    confidence = alpha * weight_amplitude + (1 - alpha) * weight_distance

    best_idx = np.argmax(confidence)

    return float(peak_tunes[best_idx]), float(confidence[best_idx])


def online_median_filter(data: np.ndarray) -> np.float64:
    """
    Calculate the median value of an array.

    A simple wrapper for numpy's median function that is compatible
    with Numba JIT compilation for faster execution.

    Parameters:
        data (np.ndarray): Input array to compute median from

    Returns:
        np.float64: Median value of the input array
    """
    return np.median(data)
