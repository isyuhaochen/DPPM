# Release verification

The clean package was checked on **the complete PrefEval test split**, using Qwen3-4B-Instruct-2507 and the included DPPM seed-42 checkpoint. This is an inference reproduction with trained weights, not a new five-seed training experiment.

After the source-tree run passed, the wheel was built and installed into a separate directory. The same full split was evaluated through that installed package, without the research project's Python modules on its import path. Both runs reproduced the archived predictions and logits exactly.

| Evaluation | Correct / total | Overall | 10 turns | 70 turns | 300 turns |
|---|---:|---:|---:|---:|---:|
| Archived seed 42 | 1,422 / 1,620 | 87.78% | 85.74% | 88.89% | 88.70% |
| Clean-package rerun | 1,422 / 1,620 | 87.78% | 85.74% | 88.89% | 88.70% |

All **1,620 question IDs, labels, and predicted answers agree**. Every legal-option logit is exactly equal; the maximum absolute difference is **0.0**. File hashes differ because the clean output includes extra dataset/category metadata. Machine-readable comparison: [prediction_comparison.json](prediction_comparison.json).

The paper's **86.79%** PrefEval result averages five independently trained seeds. A single seed should match its own archived run, not that mean. No thresholds, labels, or checkpoints were changed to obtain agreement.

## Scope of verification

- Prepared PrefEval again with this package: **7,380 training / 1,620 test questions**, with the same held-out topics and option permutation.
- Checked all test questions' text, options, labels, and recomputed segmentation before importing **578 frozen compiled segments**. Cached tensor copies were verified by SHA-256.
- Ran the memory aggregation, LoRA decoding/installation, and language-model inference anew for every test question. No archived predictions were used for answering.
- Independently re-encoded a **41-token** preference segment and a **4,016-token** history segment from raw messages; both matched their cached tensors exactly. See [encoder_check.json](encoder_check.json). All 578 segments were not re-encoded in this acceptance run.
- Checked one answer-CE backward pass: memory gradients are nonzero; the frozen backend has **zero parameter gradients**. This verifies the gradient route, not convergence of a fresh training run.
- Checked exact DPPM forward equivalence to the research implementation on a random latent sequence using the exported weights.
- Nine CPU unit tests cover evidence invariance, delta preservation, order sensitivity, first-session identity, gradients, parameter counts, checkpoint identity, and data validity.

Optional baseline wrappers, the two other backbones, and PersonaMem-v2 are included as interfaces/configurations/checkpoints, but were not all rerun end to end for this release.

## Reference environment

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.6.0 |
| Transformers | 4.51.3 |
| PEFT | 0.15.2 |
| Accelerate | 1.6.0 |
| FlashAttention | 2.7.4.post1 |
| bitsandbytes | 0.46.1 |
| GPU | NVIDIA A800 80 GB |

Frozen model forward passes use BF16; memory aggregation/decoder tensors use FP32, following the archived run. TF32 matrix multiplication is disabled for DPPM. Bitwise identity on other GPU types, kernels, or dependency versions is not guaranteed.

The compiler SHA-256 is `6438b46c828dd3b5f88f21add0f7f5cacc7994d47bf15eda266786a506044591`. The portable memory checkpoint SHA-256 is `7470fed8396bf238323b2b11c071b64f777b5c7336daa3a61342d60d534251af`.

## Comparing another rerun

```bash
python scripts/compare_predictions.py \
  --reference runs/reference/predictions.jsonl \
  --actual runs/new/predictions.jsonl --output runs/comparison.json
```

The utility requires identical question coverage and reports label/prediction mismatches plus logit differences. It exits unsuccessfully if labels or predictions disagree. The report stores checksums and numbers, without embedding source paths.
