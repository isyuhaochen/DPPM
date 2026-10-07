# Memory checkpoints

Each `<backbone>_<dataset>_seed42.pt` contains the DPPM memory trained for five epochs using answer CE. These files are approximately 2 MB each and contain 527,364 trainable parameters, not the frozen language model or compiler.

The tensor-only dictionary has `format_version`, `method`, `backbone`, `seed`, and `state_dict`. Load with `torch.load(path, weights_only=True)` or the evaluation CLI. [manifest.json](manifest.json) records hashes. Evaluation rejects a mismatched method or backbone; select the matching dataset checkpoint explicitly.

Only seed 42 is distributed. Recreate the five-seed experiment with the training CLI; do not interpret one checkpoint's score as the five-seed mean. Qwen4B/PrefEval seed 42 has archived accuracy 87.78%; see [release verification](../docs/validation.md).
