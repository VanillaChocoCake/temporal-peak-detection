# temporal-peak-detection

Core implementation of the enhanced peak-detection algorithm described in

> P. Sun, M. Zhang, R. Yuan, D. Li, J. Dong and Y. Shi, *High-accuracy Schottky diagnostics for low-SNR betatron tune measurement in ramping synchrotrons*, JINST **20** (2025) P06058, [doi:10.1088/1748-0221/20/06/P06058](https://doi.org/10.1088/1748-0221/20/06/P06058)

(section "Enhanced Peak-Detection Algorithm"). JINST **21** (2026) P08005 refers to it as the temporal peak-detection (T-PD) algorithm.

## Method

The temporal peak-detection (TPD) estimator reads the betatron tune from Schottky spectra folded onto the tune axis q ∈ [0, 0.5). For every frame:

1. **Postprocess.** Low-tune suppression, Gaussian smoothing and a dynamic-range clip of the mapped PSD.
2. **EMA sensor.** The post-processed PSD is accumulated into an exponential moving average, and the tune is read at its maximum.
3. **WLC sensor.** The local maxima of the current frame are scored by a weighted linear combination of amplitude and proximity to the previous estimate, and the best candidate gives the second reading.
4. **Fusion.** Each reading passes an online median filter, and an adaptive multi-sensor fusion (AMSF) Kalman filter combines the two. The measurement noise of each sensor, the fusion weights and the process noise are adapted at run time.

## Files

| File | Content |
|---|---|
| `tpd_estimator.py` | `TPDEstimator` (`process(frame) -> MeasurementResult`), its configuration, state and history |
| `tpd_sensors.py` | `EMA_PSD` accumulator and `AdaptiveMultiSensorFusionKalmanFilter` with its update kernel |
| `tpd_helpers.py` | WLC sensor, online median filter, reshaping helpers |
| `tune_pipeline/dsp/helpers.py` | Local-maxima search, recursive Gaussian filter, baseline clipping |
| `tune_pipeline/frames.py` | `FrontendFrame` and `MeasurementResult` data contracts |
| `tune_pipeline/conventions.py` | Low-tune cutoff shared with the preprocessing stage |

The modules import each other by file name, so the repository root has to be on `sys.path`.

## Scope

This repository contains the estimator only. The spectral preprocessing that produces a `FrontendFrame` (revolution-frequency estimation and mapping of the PSD onto the tune grid; paper sections "Data Preprocessing" and "Spectral Processing"), the simulation code and the datasets are not included.

## Dependencies

NumPy, SciPy; Numba is optional.

## Citation

```bibtex
@article{sun2025highaccuracy,
  title     = {High-accuracy Schottky diagnostics for low-SNR betatron tune measurement in ramping synchrotrons},
  author    = {Sun, Peihan and Zhang, Manzhou and Yuan, Renxian and Li, Deming and Dong, Jian and Shi, Ying},
  journal   = {Journal of Instrumentation},
  volume    = {20},
  number    = {06},
  pages     = {P06058},
  year      = {2025},
  publisher = {IOP Publishing}
}
```

## License

MIT, see [LICENSE](LICENSE).
