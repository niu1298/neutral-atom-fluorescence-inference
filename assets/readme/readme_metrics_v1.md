### V1 held-out loss-sweep inference

**Held-out count baselines.** Model choice used validation shots; the test metrics below were scored once.

| dataset | train/validation/test shots | selected emission model | test mean NLL | model-implied overlap | d-prime | posterior entropy |
|---|---|---|---:|---:|---:|---:|
| switch-off hold | 6/2/2 per condition; 66/22/22 total | `shrinkage_site_offsets_k5` | 8.081 | 2.6% | 3.71 | 0.079 |
| bright wait | 6/2/2 per condition; 60/20/20 total | `shrinkage_site_offsets_k5` | 7.908 | 4.1% | 3.30 | 0.080 |

These count-model quantities are predictive and model-implied; they are **not** empirical fidelity, FPR, FNR, or labelled physical loss.

**Operational switch-off-hold model.** Validation selected shared slope.

- **`lambda_switch_off`:** 0.0449 s^-1 (0.0302–0.0642 s^-1, 95% cluster CI).
- **`tau_switch_off`:** 22.29 s (15.57–33.15 s, 95% cluster CI).

| interval | fixed inter-readout survival q_j |
|---|---:|
| 1→2 | 0.768 (0.743–0.794, 95% cluster CI) |
| 2→3 | 0.859 (0.841–0.878, 95% cluster CI) |
| 3→4 | 0.854 (0.837–0.875, 95% cluster CI) |
| 4→5 | 0.889 (0.869–0.912, 95% cluster CI) |

This is an operational decay under the switch-off command. DDS settings remain configured and cooling was not optimized, so it is not an intrinsic dark lifetime.

**Effective bright-wait model.** Validation selected no-floor exponential.

- **`lambda_bright_effective`:** 0.6084 s^-1 (0.5456–0.6661 s^-1, 95% cluster CI).
- **`tau_bright_effective`:** 1.64 s (1.50–1.83 s, 95% cluster CI).
- **Image-1 to image-2 control:** validation selected flat retention; no post-wait retention trend was resolved.

**Cross-dataset comparison.** The sequences remain separate; the comparison propagates complete-shot bootstrap uncertainty.

- **Bright/switch-off rate ratio:** 13.56 (9.22–20.64, 95% cluster CI).
- **Bright-model prediction over 50 ms:** 3.00% (2.69–3.28%, 95% cluster CI) loss.
- **Later-interval apparent fixed loss:** 13.25% (11.60–14.70%, 95% cluster CI).
- **Observed minus predicted:** 10.26% (8.62–11.79%, 95% cluster CI).

The simple constant-rate bright-wait model does not explain the full inter-readout loss. This does not prove a fixed per-pulse cost.

**Latent-state gate.** Not accepted: complete-shot clustered uncertainty for latent transition parameters was not supplied.
