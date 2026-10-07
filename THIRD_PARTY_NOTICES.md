# Third-party notices

The compiler chat templates under `src/dppm/templates/` are copied from [SakanaAI/doc-to-lora](https://github.com/SakanaAI/doc-to-lora), revision `baa85db4d5df9b29d618af432d6ebf28b3ad5a29`, copyright 2026 Sakana AI. Its MIT license is reproduced in [LICENSES/doc-to-lora-MIT.txt](LICENSES/doc-to-lora-MIT.txt).

The frozen compiler is supplied separately by Doc-to-LoRA. The optional baseline integrations call the upstream implementations of PLUME, Mem0, LightMem, δ-mem, and Metis; their code and model weights are not redistributed here. See [baseline documentation](docs/baselines.md) for sources and revisions.

Base models and datasets are downloaded separately under their respective terms. Included checkpoint files contain only DPPM memory tensors trained for this work, with no frozen backbone or compiler weights.
