"""Per-run stateful sensors: EMA_PSD accumulator + adaptive MSF Kalman filter.

These two classes and their helpers form the sensor layer of the TPD
measurement: one is a streaming PSD smoother, the other the adaptive
multi-sensor fusion (AMSF) Kalman filter that combines the EMA and WLC
readings into the final tune estimate.

The free function ``compute_kalman_step`` is the JIT-friendly core of the
filter; ``_get_max_index`` is a small private helper used by
``EMA_PSD.acquire_q_ema``.
"""

import numpy as np
from collections import deque


def _get_max_index(array: np.ndarray) -> int:
    """
    Find the index of the maximum value in an array.

    A simple wrapper for numpy's argmax function that is compatible
    with Numba JIT compilation for faster execution.

    Parameters:
        array (np.ndarray): Input array to search for maximum value

    Returns:
        int: Index position of the maximum value in the array
    """
    return int(np.argmax(array))


class EMA_PSD:
    def __init__(self,
                 max_len: int,
                 tune_unit: np.ndarray,
                 decay_factor: float = 0.45):
        """
        Initialize an Exponential Moving Average tracker for Power Spectral Density data.

        Maintains a smoothed PSD estimate over time using exponential weighting
        to emphasize recent measurements while reducing noise.

        Parameters:
            max_len (int): Maximum length of the signal data
            tune_unit (np.ndarray): Folded tune axis the PSD bins live on
            decay_factor (float): Weight given to previous estimates (0-1)
                                 Higher values create more stable but slower-changing estimates
        """
        self.max_len = max_len
        self.decay_factor = decay_factor
        self.tune_unit = tune_unit
        self.psd = np.zeros_like(self.tune_unit)
        self.temp_buffer = np.zeros_like(self.tune_unit)

    def append(self, psd: np.ndarray):
        """
        Append new PSD data and update the exponential moving average.

        Takes new spectral measurements and incorporates them into the
        current estimate based on the configured decay factor.

        Parameters:
            psd (np.ndarray): New power spectral density measurements
        """
        np.multiply(self.psd, self.decay_factor, out=self.psd)
        np.multiply(psd, (1 - self.decay_factor), out=self.temp_buffer)  # pre-allocated buffer
        np.add(self.psd, self.temp_buffer, out=self.psd)

    def acquire_q_ema(self) -> (float, np.ndarray, np.ndarray):
        """
        Get the tune corresponding to the maximum of the accumulated PSD.

        Returns:
            tuple: ``(q_ema, tune_unit, psd)`` — the folded tune of the
                   highest PSD bin (``min(q, 1 - q)``), the tune axis, and
                   the current EMA-accumulated PSD.
        """
        max_idx = _get_max_index(self.psd)
        q_ema = self.tune_unit[max_idx]
        return np.minimum(q_ema, 1 - q_ema), self.tune_unit, self.psd


# Core computational kernel of the fusion filter, written so that it stays
# numba-JIT friendly (scalar and ndarray arithmetic only).

def compute_kalman_step(x, P, Q, alpha, min_weight,
                        innov1_array, innov2_array,
                        R1, R2, z1, z2, R_floor):
    """One predict/update cycle of the adaptive multi-sensor fusion (AMSF)
    Kalman filter.

    1. **Innovation-based R adaptation** — ``R_i <- max(α·var(z_i − x_pred)
       + (1−α)·R_i, R_floor)``. The innovation removes the state's own
       drift from the variance estimate, so R reflects the measurement
       noise alone: while the tune ramps, fast motion does not inflate
       R, the Kalman gain stays high and the filter keeps up.

    2. **Fusion weights** — the normalised precisions ``w_i ∝ 1/R_i``,
       clipped by ``min_weight`` so that neither sensor can be rejected
       outright.

    3. **Clipping-aware R_fused** — for independent sensor channels
       ``var(z_fused) = w1²·R1 + w2²·R2``, which holds for any weights,
       including clipped ones.

    4. **Innovation-subtracted Q adaptation** — ``Q = α·max(r² − R_f −
       P_pred, floor) + (1−α)·Q``. The innovation variance is
       ``P_pred + R_f + Q_true``, so subtracting the first two terms
       attributes to the process only what is left over (Sage–Husa
       style).

    5. **Physics-scale R/Q floor** — ``R_floor`` is passed in explicitly
       (the caller sets it to the tune noise floor, e.g. ``(1e-3)²``).
       The Q floor is ``R_floor / 10``, which keeps Q away from zero
       when innovations are briefly small.

    Parameters:
        x, P, Q : state estimate, its error covariance and the process noise
        alpha, min_weight : adaptation rate and lower bound of a sensor weight
        innov1_array, innov2_array : windows of ``z_i − x_pred`` from
            prior frames (NOT raw z_i). The caller maintains them.
        R1, R2 : current measurement-noise estimates of the two sensors
        z1, z2 : the two sensor readings of this frame
        R_floor : floor for R_i, so that numerical ill-conditioning near
            initialisation cannot produce absurd gains.

    Returns:
        tuple: ``(x, P, Q, R1, R2, w1, w2)`` after the step.
    """
    Q_floor = R_floor * 0.1

    # --- (1) Innovation-based R update -------------------------------
    # innov arrays hold the previous frames' (z_i - x_pred) values;
    # their variance reflects only the measurement noise (provided the
    # filter is roughly tracking).
    R1_new = max(alpha * np.var(innov1_array) + (1 - alpha) * R1, R_floor)
    R2_new = max(alpha * np.var(innov2_array) + (1 - alpha) * R2, R_floor)

    # --- (2) Inverse-variance precision weights + min_weight clip ----
    inv_R1 = 1.0 / R1_new
    inv_R2 = 1.0 / R2_new
    w1 = inv_R1 / (inv_R1 + inv_R2)
    w2 = 1.0 - w1
    if w1 < min_weight:
        w1 = min_weight; w2 = 1.0 - min_weight
    elif w1 > 1.0 - min_weight:
        w1 = 1.0 - min_weight; w2 = min_weight

    # --- Kalman predict ----------------------------------------------
    x_pred = x
    P_pred = P + Q

    # --- (3) Clipping-aware R_fused ----------------------------------
    z_fused = w1 * z1 + w2 * z2
    R_fused = w1 * w1 * R1_new + w2 * w2 * R2_new

    # --- Kalman update -----------------------------------------------
    K = P_pred / (P_pred + R_fused)
    residual_fused = z_fused - x_pred
    x_new = x_pred + K * residual_fused
    P_new = (1.0 - K) * P_pred

    # --- (4) Innovation-subtracted Q update --------------------------
    # innov² ≈ P_pred + R_fused + Q_true. Subtract the first two to get
    # an unbiased-in-expectation Q estimate; clamp to Q_floor so an
    # occasional tiny innovation doesn't collapse Q to zero.
    excess = residual_fused * residual_fused - R_fused - P_pred
    Q_new = alpha * max(excess, Q_floor) + (1.0 - alpha) * Q

    return x_new, P_new, Q_new, R1_new, R2_new, w1, w2


class AdaptiveMultiSensorFusionKalmanFilter:
    def __init__(
            self,
            initial_state: float = 0.3,
            initial_estimate_error: float = 1e-4,
            process_noise: float = 1e-6,
            measurement_noise: float = 1e-6,
            alpha: float = 0.6,
            min_weight: float = 0.02,
            window_len: int = 8,
            R_floor: float = 1e-8,
    ):
        """
        Adaptive Kalman filter with dynamic multi-sensor fusion capabilities.

        Implements a Kalman filter that dynamically adjusts measurement and process
        noise parameters while optimally fusing measurements from two independent sensors.

        The default covariances are set on the scale of the tune, which lives
        in [0, 0.5]: a typical measurement σ ≈ 1e-3 gives R ≈ 1e-6, a typical
        drift of 1e-3 per frame gives Q ≈ 1e-6, and the start-up uncertainty
        σ ≈ 1e-2 gives P ≈ 1e-4.

        Parameters:
            initial_state (float): Initial state estimate
            initial_estimate_error (float): Initial state error covariance
            process_noise (float): Initial process noise covariance
            measurement_noise (float): Initial measurement noise for both sensors
            alpha (float): Adaptation rate for noise estimation (0-1)
                          Higher values respond faster to changes in noise characteristics
            min_weight (float): Minimum weight threshold for any sensor
                               Prevents complete rejection of either sensor
            window_len (int): Length of window for noise estimation
            R_floor (float): Lower bound on adapted R_i. Default ``1e-8`` =
                (1e-4)² tune units, i.e. a measurement noise well below one
                tune-grid bin is treated as negligible.
        """
        # Store ctor args so reset() can restore them later
        self._initial_state = initial_state
        self._initial_estimate_error = initial_estimate_error
        self._initial_process_noise = process_noise
        self._initial_measurement_noise = measurement_noise
        self._window_len = window_len

        self.x = initial_state
        self.P = initial_estimate_error
        self.Q = process_noise
        self.R1 = measurement_noise
        self.R2 = measurement_noise
        self.w1 = 0.5
        self.w2 = 0.5
        self.alpha = alpha
        self.min_weight = min_weight
        # Sliding noise-estimation windows, one per sensor, holding the
        # innovations z_i − x_pred (see predict_update).
        self.window1 = deque(maxlen=window_len)
        self.window2 = deque(maxlen=window_len)
        self.R_floor = R_floor

    def reset(self, new_initial_state: float = None):
        """Reset all internal state to the initial configuration.

        Called when starting a new measurement run, so that the previous
        run's Kalman convergence cannot bias the new one.

        Parameters:
            new_initial_state (float, optional): If given, overrides the ctor's
                initial_state. Useful when switching between qx and qy measurements.
        """
        self.x = new_initial_state if new_initial_state is not None else self._initial_state
        self.P = self._initial_estimate_error
        self.Q = self._initial_process_noise
        self.R1 = self._initial_measurement_noise
        self.R2 = self._initial_measurement_noise
        self.w1 = 0.5
        self.w2 = 0.5
        self.window1.clear()
        self.window2.clear()

    def predict_update(self,
                       z1: np.float64, z2: np.float64):
        """
        Execute a complete Kalman filter cycle with sensor fusion.

        Performs prediction and update steps using measurements from two sensors,
        dynamically adjusting their weights based on estimated noise characteristics.

        Parameters:
            z1 (np.float64): Measurement from sensor 1
            z2 (np.float64): Measurement from sensor 2

        Returns:
            float: Updated state estimate after fusion and filtering
        """
        # The windows store the innovations z_i − x_pred, so their variance
        # reflects the measurement noise alone, not the motion of the state.
        x_pred = self.x  # capture pre-update prediction for this frame
        self.window1.append(float(z1) - x_pred)
        self.window2.append(float(z2) - x_pred)
        innov1_array = np.ascontiguousarray(self.window1, dtype=np.float64)
        innov2_array = np.ascontiguousarray(self.window2, dtype=np.float64)
        self.x, self.P, self.Q, self.R1, self.R2, self.w1, self.w2 = compute_kalman_step(
            float(self.x), float(self.P), float(self.Q),
            float(self.alpha), float(self.min_weight),
            innov1_array, innov2_array,
            float(self.R1), float(self.R2), float(z1), float(z2),
            float(self.R_floor),
        )
        return self.x

    def detector_weights(self):
        """
        Get the current weights assigned to each sensor.

        Returns the normalized weights used in the fusion process,
        indicating the relative reliability of each sensor.

        Returns:
            tuple[float, float]: (w1, w2) - Weights for sensor 1 and sensor 2,
                                where w1 + w2 = 1
        """
        return self.w1, self.w2
