"""Temporal peak-detection (TPD) estimator: FrontendFrame → MeasurementResult.

The enhanced peak-detection algorithm published in JINST 20 (2025) P06058. Two sensors read
the tune from the post-processed tune-axis PSD: the peak of its exponential
moving average (EMA), and a weighted linear combination (WLC) of amplitude
and proximity scores over the candidate peaks of the current frame. Each
reading passes an online median filter, and the two are fused by an adaptive
multi-sensor fusion (AMSF) Kalman filter.

The implementation is split into three concerns:

- ``EstimatorConfig``   — frozen immutable parameters (algorithm knobs,
  detector passband, grid size).
- ``EstimatorState``    — mutable per-run state: the streaming EMA_PSD
  accumulator, rolling EMA/WLC queues for the online median filter, the
  adaptive multi-sensor Kalman fusion filter, and ``frev_prev`` (carried
  into the next spectral-preprocessing call's harmonic-foldback fallback).
- ``HistoryState``      — append-only deques of the five per-frame
  outputs, kept separate so callers can reset or persist the history
  without touching live state.

``TPDEstimator.process(frame)`` is a pure function of
(frame, state, history, cfg) and returns a ``MeasurementResult``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np
from scipy.ndimage import gaussian_filter1d

from tune_pipeline.frames import FrontendFrame, MeasurementResult, QualityFlag
from tune_pipeline.conventions import LOW_CUT_Q

# The sensor and helper layers are imported directly, so the dependency
# stays one-way: estimator -> tpd_sensors / tpd_helpers.
from tpd_sensors import EMA_PSD, AdaptiveMultiSensorFusionKalmanFilter
from tpd_helpers import weighted_linear_combination, online_median_filter


@dataclass(frozen=True)
class EstimatorConfig:
    """Immutable knobs for the TPD estimator.

    The caller derives these from its own application-level configuration
    objects, so this class stays independent of them.
    """

    sideband_width: float
    alpha: float
    max_len: int
    decay_factor: float
    initial_q: float

    # Detector passband, carried so a caller can pass its detector
    # configuration through unchanged; the estimator does not gate on it.
    detector_fc: float
    detector_bandwidth: float

    tune_grid_size: int = 1024

    # Post-processing of the mapped PSD.
    # Single source of truth (``conventions.LOW_CUT_Q``): the estimator's own
    # low-q suppression has to equal the ``valid_mask`` cutoff built by the
    # spectral preprocessing stage, otherwise sidebands near the low end of
    # the operating range [0.05, 0.49] are clipped.
    q_suppress_below: float = LOW_CUT_Q
    mapping_window_size: int = 25

    # EMA/WLC online-median window length (sensor smoothing)
    median_window: int = 5


@dataclass
class EstimatorState:
    """Mutable per-run state owned by the estimator.

    Created via ``EstimatorState.new(cfg, tune_unit)`` so the queues /
    filters are built with the correct maxlen / initial state.
    """

    ema_psd: object                              # EMA_PSD instance
    ema_queue: Deque[float]
    wlc_queue: Deque[float]
    msf: object                                  # AdaptiveMultiSensorFusionKalmanFilter
    frev_prev: float

    @classmethod
    def new(cls, cfg: EstimatorConfig, tune_unit: np.ndarray) -> "EstimatorState":
        return cls(
            ema_psd=EMA_PSD(
                max_len=cfg.max_len,
                decay_factor=cfg.decay_factor,
                tune_unit=tune_unit,
            ),
            ema_queue=deque(maxlen=cfg.median_window),
            wlc_queue=deque(maxlen=cfg.median_window),
            msf=AdaptiveMultiSensorFusionKalmanFilter(initial_state=cfg.initial_q),
            frev_prev=4e6,
        )

    def clear(self, cfg: EstimatorConfig) -> None:
        """Reset live state to a fresh-run baseline.

        Clears the state owned here: the EMA_PSD accumulator, the rolling
        queues, and the Kalman filter's converged state. Histories are
        reset separately through ``HistoryState.clear``.
        """
        self.ema_queue.clear()
        self.wlc_queue.clear()
        self.ema_psd.psd = np.zeros_like(self.ema_psd.psd)
        self.ema_psd.temp_buffer = np.zeros_like(self.ema_psd.temp_buffer)
        # Note: do NOT reset self.ema_psd.tune_unit — it is the immutable q_grid;
        # zeroing it would break subsequent mapping lookups.
        self.msf.reset(new_initial_state=cfg.initial_q)
        # frev_prev returns to the pre-run default so the next
        # spectral-preprocessing call's harmonic-foldback fallback does not
        # inherit a stale value.
        self.frev_prev = 4e6


@dataclass
class HistoryState:
    """Append-only per-run history of the five per-frame outputs."""

    q_ema: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    q_wlc: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    q_pd: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    q_predicted: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    failed_to_detect: Deque[bool] = field(default_factory=lambda: deque(maxlen=2048))

    def clear(self) -> None:
        self.q_ema.clear()
        self.q_wlc.clear()
        self.q_pd.clear()
        self.q_predicted.clear()
        self.failed_to_detect.clear()


class TPDEstimator:
    """FrontendFrame → MeasurementResult via the TPD fusion pipeline.

    One frame passes through: EMA sensor → WLC sensor → adaptive
    multi-sensor Kalman fusion. The direct peak-detection readout is
    computed alongside as a diagnostic channel and is not fed back into
    the filter.
    """

    def __init__(
        self,
        cfg: EstimatorConfig,
        tune_unit: np.ndarray,
        state: Optional[EstimatorState] = None,
        history: Optional[HistoryState] = None,
    ):
        self.cfg = cfg
        self.tune_unit = tune_unit
        self.state = state if state is not None else EstimatorState.new(cfg, tune_unit)
        self.history = history if history is not None else HistoryState()


    def postprocess(self, mapped_psd: np.ndarray,
                    valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Low-q suppression + Gaussian smoothing + dynamic-range clip.

        The spectral preprocessing stage hands back only the PSD mapped
        onto the tune axis, so an estimator that needs this kind of
        cleanup runs it here and the preprocessing stage stays
        algorithm-agnostic.

        ``valid_mask`` (per-bin validity as declared upstream) is OR'd
        into the suppression set when provided. ``process()`` passes
        ``frame.valid_mask``, so the Gaussian smoothing cannot leak
        signal back into bins the preprocessing stage declared invalid.
        With the default ``None`` only ``q_suppress_below`` is applied,
        for callers that hand in a raw PSD directly.

        Returns a new float64 array — input is not mutated.
        """
        g = np.asarray(self.tune_unit, dtype=np.float64)
        psd = np.asarray(mapped_psd, dtype=np.float64).copy()

        q_suppress_below = self.cfg.q_suppress_below
        mapping_window_size = self.cfg.mapping_window_size

        # ---- 1) Low-q suppression — combine the estimator-config
        #    threshold with the upstream valid_mask. Either source can
        #    declare a bin invalid; the union is suppressed.
        suppress_mask = np.zeros_like(g, dtype=np.bool_)
        if q_suppress_below is not None and q_suppress_below > g[0]:
            suppress_mask = g < q_suppress_below
        if valid_mask is not None:
            suppress_mask = suppress_mask | (~np.asarray(valid_mask, dtype=bool))
        psd[suppress_mask] = 0

        # ---- 2) Gaussian smoothing on q_grid ----
        if mapping_window_size:
            sigma = (mapping_window_size - 1) / 4.0
            psd = gaussian_filter1d(psd, sigma, mode="nearest")

        # ---- 3) Re-zero suppression region (smoothing may have bled into it) ----
        psd[suppress_mask] = 0

        # ---- 4) Dynamic-range clip (subtract vmax/3, floor at 0) ----
        vmax = float(np.max(psd)) if np.any(psd > 0) else 0.0
        if vmax > 0:
            psd = np.maximum(0.0, psd - vmax / 3)

        return psd


    def process(self, frame: FrontendFrame) -> MeasurementResult:
        """Run one frame of the TPD fusion pipeline.

        Returns a ``MeasurementResult``; side-effects: state queues and
        the Kalman filter advance, five history deques grow by one.

        If the preprocessing stage flagged the frame as ``FEW_PEAKS``
        (the contract for "data is unusable — don't rely on the mapped
        PSD") the pipeline must NOT advance EMA / WLC / Kalman state — the
        mapped PSD is all-zero by construction and would feed meaningless
        measurements into the fusion filter. The frame is recorded as a
        failed measurement; live state is frozen until the next usable
        frame.
        """
        if int(frame.quality_flags) & int(QualityFlag.FEW_PEAKS):
            # Freeze live state. Mirror the prior q on the histories so
            # downstream consumers (plots, error stats) see a defined
            # value per frame; flag as failed.
            last_q = (
                self.history.q_predicted[-1]
                if len(self.history.q_predicted) > 0
                else self.cfg.initial_q
            )
            self.history.q_ema.append(last_q)
            self.history.q_wlc.append(last_q)
            self.history.q_predicted.append(last_q)
            self.history.q_pd.append(last_q)
            self.history.failed_to_detect.append(True)
            return MeasurementResult(q=last_q, failed=True, source="tpd")

        psd_mapped = self.postprocess(frame.mapped_psd, frame.valid_mask)
        self.state.ema_psd.append(psd_mapped)

        # Sensor 1: EMA-based reference tune
        q_ema, tune_unit, psd = self.state.ema_psd.acquire_q_ema()
        # q_ref is the centre of the WLC proximity weighting: the previous
        # fused estimate, or a mid-range prior on the very first frame.
        q_ref = self.history.q_predicted[-1] if len(self.history.q_predicted) > 0 else 0.33
        self.state.ema_queue.append(q_ema)
        q_ema = online_median_filter(np.asarray(self.state.ema_queue, dtype=np.float64))
        self.history.q_ema.append(q_ema)

        # Sensor 2: weighted linear combination
        q_wlc, _ = weighted_linear_combination(
            np.minimum(tune_unit, 1 - tune_unit),
            psd_mapped,
            q_ref,
            self.cfg.alpha,
        )
        self.state.wlc_queue.append(q_wlc)
        q_wlc = online_median_filter(np.asarray(self.state.wlc_queue, dtype=np.float64))
        self.history.q_wlc.append(q_wlc)

        # Multi-sensor Kalman fusion
        q_pred = self.state.msf.predict_update(q_ema, q_wlc)
        self.history.q_predicted.append(q_pred)

        # Direct peak-detection readout (diagnostic channel)
        q_pd = self.tune_unit[np.argmax(psd_mapped)]
        self.history.q_pd.append(q_pd)

        self.history.failed_to_detect.append(False)

        return MeasurementResult(q=q_pred, failed=False, source="tpd")
