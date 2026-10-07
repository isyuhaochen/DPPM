# Baseline setup

All commands run at the repository root and use the normalized datasets from `dppm prepare`. Each method writes to its own output directory. Baseline adapters expose `score(row)`, returning one score per displayed option. Generation is used internally for summaries/extraction; benchmark answers are scored at the option-letter token.

The full release rerun covers Qwen4B DPPM on PrefEval. The wrappers below have not all received a new complete benchmark run in this release. They should not be interpreted as independently verified reproduction claims for every paper row.

## References, RAG, and rolling summaries

Install `pip install -e '.[text]'`. Use a backbone config and choose `no_context`, `gold_state`, `full_context`, `rag`, or `rolling_summary`. These methods require neither a compiler cache nor a memory checkpoint.

RAG embeds two-message units with BGE-M3 and retrieves the top 10 using the question, excluding answer options. Encoder inputs use the retriever's 8,192-token limit. Rolling Summary updates one summary per visible session with a 1,024-token output budget. Full Context uses all visible history and raises an error if the answer prompt exceeds the backbone's native capacity. These are reference evaluation implementations, not optimized serving engines.

## Mem0 and LightMem

Use Python 3.10 or 3.11. The YAML files already point to these checkout locations:

```bash
pip install -e '.[text,stores]'
git clone https://github.com/mem0ai/mem0.git third_party/mem0
git -C third_party/mem0 checkout 42cf18c4e6adb448e981aa1c7b55c1602b0cb670
git clone https://github.com/zjunlp/LightMem.git third_party/LightMem
git -C third_party/LightMem checkout 8449d574df6bae1bdf3314a1564da65e2f37e046

dppm evaluate --config configs/qwen4b.yaml --data data/prefeval \
  --method mem0 --output runs/qwen4b/prefeval/mem0
dppm evaluate --config configs/qwen4b.yaml --data data/prefeval \
  --method lightmem --output runs/qwen4b/prefeval/lightmem
```

Upstream source packages are loaded through `external.<method>.source`; installing their large default extras is unnecessary. The wrappers keep official extraction/indexing/update logic and supply local frozen-model generation plus BGE-M3 embeddings. They do not call hosted LLM APIs. Upstream packages may have their own telemetry settings; for Mem0 use `MEM0_TELEMETRY=false` if desired.

Mem0 extracts/updates facts with up to 2,048 generated tokens and retrieves 10 facts. LightMem uses LLMLingua-2 precompression, topic segmentation, user-only extraction, embedding retrieval, and offline updates. Its required ordering timestamps are synthetic indices, explicitly described as nonfactual in the extraction prompt. Models are downloadable via public IDs in the config. Store state is isolated by history hash and saved under the evaluation output.

Provider failures are surfaced even when upstream code catches exceptions. Incomplete JSON and malformed memory records stop evaluation instead of being counted as successful empty extraction. This is stricter than historical runs that tolerated some empty extractions, so exact historical baseline parity is not asserted. A partially written store requires a fresh output directory to avoid duplicate insertion. Provider factories are process-global: run one store method per process.

## Direct D2L and PLUME

Install the core Doc-to-LoRA environment described in the README. Direct D2L compiles the visible history into chunks, concatenates the resulting low-rank factors, and includes the compiler's shared bias once. It performs no learned recurrent aggregation.

```bash
git clone https://github.com/xiaobingshi-LLM/PLUME.git third_party/PLUME
git -C third_party/PLUME checkout 9f8880a4264be19358bd6d65619c5a47edc1f924
dppm evaluate --config configs/qwen4b.yaml --data data/prefeval \
  --method direct_d2l --output runs/qwen4b/prefeval/direct_d2l
dppm evaluate --config configs/qwen4b.yaml --data data/prefeval \
  --method plume --output runs/qwen4b/prefeval/plume
```

PLUME calls its official router, global adapter update, default hyperparameters, and full-vocabulary Jensen–Shannon fusion rule. Its old context is the prefix before the final visible user turn; its full context is the entire visible history. Compiled chunks use equal padded widths and microbatch size one. Only after full-vocabulary fusion are legal option scores selected. A common D2L loader avoids dependence on PLUME's modified upstream loader signature. PLUME uses native chat templates for both context and answering, as in the archived Qwen experiment.

## Released δ-mem and Metis models

These pretrained models use a **separate Transformers 5.4 environment**. Do not install the `native` and `d2l` extras in the same environment.

```bash
python -m venv .venv-native
source .venv-native/bin/activate
pip install torch==2.6.0
pip install -e '.[native]'
git clone https://github.com/declare-lab/delta-Mem.git third_party/delta-Mem
git -C third_party/delta-Mem checkout bb6e99bb6b3a94ddae7f9b4743bfd8ae37e98675
git clone https://github.com/MemTensor/Metis.git third_party/Metis
git -C third_party/Metis checkout 22f7aabc8d3e9c39fcea676b11a10007bc8b3748

dppm evaluate --config configs/delta_mem.yaml --data data/prefeval \
  --method delta_mem --output runs/delta_mem/prefeval
dppm evaluate --config configs/metis4b.yaml --data data/prefeval \
  --method metis --output runs/metis4b/prefeval
```

Use `configs/metis9b.yaml` or `configs/metis27b.yaml` for the larger releases. Public model IDs are `declare-lab/delta-mem_qwen3_4b-instruct` and `IAAR-Shanghai/Metis-{4B,9B,27B}`. Override `external.<method>.checkpoint` to use local weights. δ-mem's base is Qwen3-4B-Instruct-2507. Metis uses its native backbone, so its 4B/9B/27B results are not same-backbone architectural controls.

Visible history is written chronologically in 8,192-token blocks. Memory is reset between different histories and read without updating during answers. These are **pretrained inference-only** adapters, with no task-specific training. Metis loading rejects missing, unexpected, or mismatched weights. The configs use one visible GPU; larger models require sufficient memory, and this release does not implement model parallelism.

## Learned memories and controls

Use the core environment. `dppm train --method METHOD` trains only the memory parameters with answer CE. Supported controls:

| Method | Meaning |
|---|---|
| `rpmem` | Coordinate recurrent old/new gate, 524,800 parameters |
| `evidence` | Coordinate softmax evidence pooling with scale restoration |
| `delta` | Standalone associative Delta path: 8 key dimensions, no output calibration |
| `delta_matched` | Delta branch as used inside DPPM: 4 key dimensions, calibrated output |
| `rpmem_matched` | Active rank-4 gate correction; 527,364 parameters |
| `dppm_fixed` | Equal fixed fusion instead of a learned mixture |
| `dppm_no_final_calibration` | Remove fused-output calibration only |
| `dppm_no_calibration` | Remove both branch and fused-output calibration |

All other DPPM settings stay the same. Train controls separately; evaluation checks checkpoint method identity. `delta` is not the public δ-mem model. Component ablations and matched-parameter controls are available even though they are not every row of the main table.

## Adding another method

Provide an importable class with `__init__(config, dataset, output)` and `score(row)`. The latter must return a tensor of finite legal-option scores. Use `--method your_package:YourMethod`; sharding, completeness checks, and metric calculation remain shared. This plugin runs as local Python code.
