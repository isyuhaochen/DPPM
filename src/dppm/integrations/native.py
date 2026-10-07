"""Official released Delta-Mem and Metis with complete chronological writes."""

from pathlib import Path

import torch

from ..backend import prompt_ids
from ..config import import_source
from ..data import hash_text


class NativeMemory:
    def __init__(self, method, config, dataset):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.data = dataset
        self.device = config.get("device", "cuda:0")
        settings = config.get("external", {}).get(method, {})
        checkpoint = settings.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"Set external.{method}.checkpoint in YAML")
        if method == "metis":
            import_source(settings.get("source"), "metis")
            from metis.modeling_metis import MetisForCausalLM
            from metis.memory_utils import encode_and_commit_memory

            self.model, loading = MetisForCausalLM.from_pretrained(
                checkpoint,
                dtype=torch.bfloat16,
                device_map=self.device,
                key_mapping={r"\.language_model": ""},
                output_loading_info=True,
            )
            if any(
                loading.get(k)
                for k in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]
            ):
                raise RuntimeError(f"Incomplete native checkpoint loading: {loading}")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
            self.reset = self.model.reset
            self.write = lambda x: encode_and_commit_memory(
                self.model, x, attention_mask=torch.ones_like(x)
            )
        else:
            import_source(settings.get("source"), "deltamem")
            if not Path(checkpoint).is_dir():
                from huggingface_hub import snapshot_download

                checkpoint = snapshot_download(checkpoint)
            from deltamem.core import (
                HFDeltaMemConfig,
                attach_delta_mem,
                load_delta_mem_adapter,
                reset_delta_mem_states,
                set_delta_mem_write_enabled,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                config["base_model"], dtype=torch.bfloat16, attn_implementation="sdpa"
            ).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
            cfg = HFDeltaMemConfig.from_pretrained(checkpoint)
            if cfg.memory_write_granularity != "token":
                raise ValueError("This adapter expects the released token-write Delta-Mem recipe")
            attach_delta_mem(self.model, cfg)
            load_delta_mem_adapter(self.model, checkpoint)
            self.model.to(self.device)
            self.reset = lambda: reset_delta_mem_states(self.model)

            def write(x):
                set_delta_mem_write_enabled(self.model, True)
                try:
                    self.model(
                        input_ids=x,
                        attention_mask=torch.ones_like(x),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                finally:
                    set_delta_mem_write_enabled(self.model, False)

            self.write = write
            set_delta_mem_write_enabled(self.model, False)
        self.model.eval().requires_grad_(False)
        self.previous = None

    def score(self, row):
        history = self.data.history(row)
        key = hash_text(history)
        if key != self.previous:
            self.reset()
            ids = self.tokenizer.apply_chat_template(
                [dict(role="user", content=history)],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            for start in range(0, len(ids), 8192):
                self.write(torch.tensor([ids[start : start + 8192]], device=self.device))
            self.previous = key
        ids = prompt_ids(self.tokenizer, row)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        model_config = getattr(self.model.config, "backbone_configs", self.model.config)
        model_config = getattr(model_config, "text_config", model_config)
        if len(ids) > model_config.max_position_embeddings:
            raise ValueError("Native answer input exceeds context capacity")
        x = torch.tensor([ids], device=self.device)
        labels = [
            self.tokenizer.encode(chr(65 + i), add_special_tokens=False)
            for i in range(len(row["options"]))
        ]
        if any(len(v) != 1 for v in labels):
            raise ValueError("Native tokenizer must encode each option letter as one token")
        logits = self.model(
            input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1
        ).logits[0, -1]
        return logits[[v[0] for v in labels]]
