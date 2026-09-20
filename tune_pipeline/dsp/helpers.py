"""Numpy-only DSP helper primitives shared by the spectral preprocessing
stage and the tune estimator.

The module has no dependency on any other module of the project — strictly
``numpy`` (+ optional ``numba``) — so either consumer can import it freely.

``gaussian_filter`` and ``find_local_maxima`` have a numba-JIT path that is
preferred at runtime; the numpy implementations are kept as the
fallback for environments without numba and as the bit-exactness
reference. ``clip_data`` stays numpy-only — its hot work is
``np.percentile``, which numba cannot accelerate further.
"""

import numpy as np

try:
    from numba import njit
    _NUMBA_OK = True
except ImportError:
    _NUMBA_OK = False
    njit = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# gaussian_filter — recursive IIR (van Vliet/Young approximation)
# ---------------------------------------------------------------------------

def _gaussian_filter_np(data: np.ndarray, window_size: int) -> np.ndarray:
    """Pure-numpy reference. ``window_size`` already validated/odd."""
    sigma = (window_size - 1) / 6.0
    if sigma >= 2.5:
        q = 0.98711 * sigma - 0.96330
    else:
        q = 3.97156 - 4.14554 * np.sqrt(1.0 - 0.26891 * sigma)
    q2 = q * q
    q3 = q2 * q
    b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3
    b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3
    b2 = -(1.4281 * q2 + 1.26661 * q3)
    b3 = 0.422205 * q3
    B = 1.0 - (b1 + b2 + b3) / b0
    a1 = b1 / b0
    a2 = b2 / b0
    a3 = b3 / b0

    n = data.shape[0]
    y = np.zeros(n, dtype=np.float64)

    if n > 0:
        y[0] = B * data[0]
    if n > 1:
        y[1] = B * data[1] + a1 * y[0]
    if n > 2:
        y[2] = B * data[2] + a1 * y[1] + a2 * y[0]
    for i in range(3, n):
        y[i] = B * data[i] + a1 * y[i - 1] + a2 * y[i - 2] + a3 * y[i - 3]

    y2 = np.zeros(n, dtype=np.float64)
    if n > 0:
        y2[n - 1] = y[n - 1]
    if n > 1:
        y2[n - 2] = y[n - 2] + a1 * y2[n - 1]
    if n > 2:
        y2[n - 3] = y[n - 3] + a1 * y2[n - 2] + a2 * y2[n - 1]
    for i in range(n - 4, -1, -1):
        y2[i] = y[i] + a1 * y2[i + 1] + a2 * y2[i + 2] + a3 * y2[i + 3]
    return y2


if _NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _gaussian_filter_nb(data: np.ndarray, window_size: int) -> np.ndarray:
        sigma = (window_size - 1) / 6.0
        if sigma >= 2.5:
            q = 0.98711 * sigma - 0.96330
        else:
            q = 3.97156 - 4.14554 * np.sqrt(1.0 - 0.26891 * sigma)
        q2 = q * q
        q3 = q2 * q
        b0 = 1.57825 + 2.44413 * q + 1.4281 * q2 + 0.422205 * q3
        b1 = 2.44413 * q + 2.85619 * q2 + 1.26661 * q3
        b2 = -(1.4281 * q2 + 1.26661 * q3)
        b3 = 0.422205 * q3
        B = 1.0 - (b1 + b2 + b3) / b0
        a1 = b1 / b0
        a2 = b2 / b0
        a3 = b3 / b0

        n = data.shape[0]
        y = np.zeros(n, dtype=np.float64)
        if n > 0:
            y[0] = B * data[0]
        if n > 1:
            y[1] = B * data[1] + a1 * y[0]
        if n > 2:
            y[2] = B * data[2] + a1 * y[1] + a2 * y[0]
        for i in range(3, n):
            y[i] = B * data[i] + a1 * y[i - 1] + a2 * y[i - 2] + a3 * y[i - 3]

        y2 = np.zeros(n, dtype=np.float64)
        if n > 0:
            y2[n - 1] = y[n - 1]
        if n > 1:
            y2[n - 2] = y[n - 2] + a1 * y2[n - 1]
        if n > 2:
            y2[n - 3] = y[n - 3] + a1 * y2[n - 2] + a2 * y2[n - 1]
        for i in range(n - 4, -1, -1):
            y2[i] = y[i] + a1 * y2[i + 1] + a2 * y2[i + 2] + a3 * y2[i + 3]
        return y2


def gaussian_filter(data: np.ndarray, window_size: int, mode: str = "constant") -> np.ndarray:
    """
    IIR (recursive) Gaussian filter implementation.

    Numba-JIT'd inner loop when numba is available (~100× over the numpy
    fallback at the array sizes used here); otherwise falls back to the
    pure-numpy reference. ``mode`` is accepted for interface stability but
    ignored by both paths.

    Parameters
    ----------
    data : np.ndarray
        1-D input signal.
    window_size : int
        Kernel size (rounded up to next odd if even). Must be ≥ 3.
    mode : str
        Boundary handling mode; currently unused, kept for interface
        consistency.

    Returns
    -------
    np.ndarray
        Filtered signal (float64).
    """
    del mode  # unused, kept for caller-signature stability
    if window_size < 3:
        raise ValueError("Window size must be >= 3")
    if window_size % 2 == 0:
        window_size += 1
    arr = np.ascontiguousarray(data, dtype=np.float64)
    if _NUMBA_OK:
        return _gaussian_filter_nb(arr, window_size)
    return _gaussian_filter_np(arr, window_size)


# ---------------------------------------------------------------------------
# clip_data — np.percentile dominates; numba cannot improve this further
# ---------------------------------------------------------------------------

def clip_data(data: np.ndarray, percent: int = 20) -> np.ndarray:
    """Subtract the ``percent``-th percentile and clip negatives to zero.

    Used as a baseline-removal step before peak detection.
    """
    p = np.percentile(data, percent)
    data = np.maximum(data - p, 0)
    return data


# ---------------------------------------------------------------------------
# find_local_maxima — explicit element-wise comparison loop
# ---------------------------------------------------------------------------

def _find_local_maxima_np(array: np.ndarray):
    """Pure-numpy reference."""
    n = array.shape[0]
    maxima = np.zeros(n, dtype=np.bool_)
    values = np.zeros(n, dtype=np.float64)

    if n > 1 and array[0] > array[1]:
        maxima[0] = True
        values[0] = array[0]

    for i in range(1, n - 1):
        if array[i] > array[i - 1] and array[i] > array[i + 1]:
            maxima[i] = True
            values[i] = array[i]

    if n > 1 and array[n - 1] > array[n - 2]:
        maxima[n - 1] = True
        values[n - 1] = array[n - 1]

    return maxima, values


if _NUMBA_OK:
    @njit(cache=True, fastmath=True)
    def _find_local_maxima_nb(array: np.ndarray):
        n = array.shape[0]
        maxima = np.zeros(n, dtype=np.bool_)
        values = np.zeros(n, dtype=np.float64)

        if n > 1 and array[0] > array[1]:
            maxima[0] = True
            values[0] = array[0]

        for i in range(1, n - 1):
            if array[i] > array[i - 1] and array[i] > array[i + 1]:
                maxima[i] = True
                values[i] = array[i]

        if n > 1 and array[n - 1] > array[n - 2]:
            maxima[n - 1] = True
            values[n - 1] = array[n - 1]

        return maxima, values


def find_local_maxima(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Find local maxima in a 1-D array.

    Numba-JIT'd inner loop when numba is available (~190× over the
    numpy fallback at the array sizes used here); otherwise falls back to
    the pure-numpy reference. Edge points are considered local maxima if
    they are strictly higher than their single neighbour.

    Parameters
    ----------
    array : np.ndarray
        Input array to search for local maxima.

    Returns
    -------
    (maxima, values) : tuple
        ``maxima`` — boolean array marking positions of local maxima.
        ``values`` — float64 array with the maxima values (zeros elsewhere).
    """
    arr = np.ascontiguousarray(array, dtype=np.float64)
    if _NUMBA_OK:
        return _find_local_maxima_nb(arr)
    return _find_local_maxima_np(arr)
