"""PLUME's official router/update/JS rule, adapted to legal-option scoring."""

import math

import torch

from ..backend import D2LBackend, question, tokenizer
from ..config import import_source
from ..data import serialize


class Plume:
    def __init__(self, config, dataset):
        import_source(config.get("d2l_source"), "ctx_to_lora")
        import_source(config.get("external", {}).get("plume", {}).get("source"), "plume")
        from plume.config import PlumeConfig

        # Use the common loader: upstream PLUME's loader requires a modified
        # Doc-to-LoRA signature. Router, update and fusion remain upstream code.
        self.backend = D2LBackend(config, encoder=True)
        self.backend.net.enable_iterative_mode(False)
        self.ctx_tokenizer = tokenizer({**config, "answer_template": "native"})
        self.backend.tokenizer = self.ctx_tokenizer
        self.options, self.data = PlumeConfig(), dataset
        self.previous = None

    def compile(self, text):
        from plume.d2l import tokenize_context
        from ctx_to_lora.modeling.lora_merger import combine_lora

        b = self.backend
        ids = tokenize_context(text, self.ctx_tokenizer)
        count = max(1, math.ceil(len(ids) / 8192))
        width = max(1, math.ceil(len(ids) / count))
        chunks = [ids[i : i + width] for i in range(0, len(ids), width)] or [[]]
        parts = []
        for chunk in chunks:
            x = torch.full(
                (1, width), self.ctx_tokenizer.pad_token_id or 0, dtype=torch.long, device=b.device
            )
            mask = torch.zeros_like(x)
            if chunk:
                x[0, : len(chunk)] = torch.tensor(chunk, device=b.device)
                mask[0, : len(chunk)] = 1
            raw, _ = b.net.generate_weights(x, mask)
            parts.append(
                {m: {k: v.cpu() for k, v in factors.items()} for m, factors in raw.items()}
            )
        raw = {
            m: {k: torch.cat([p[m][k] for p in parts]).to(b.device) for k in ["A", "B"]}
            for m in parts[0]
        }
        return combine_lora(
            raw,
            torch.tensor([len(chunks)], device=b.device),
            lora_bias=b.hyper.get_head_bias() if b.hyper.config.use_bias else None,
        )

    def score(self, row):
        from plume.adapters import global_update
        from plume.memory import segment_memory, activate_memory
        from plume.decoding import js_divergence

        key = tuple(row["sessions"])
        b, cfg = self.backend, self.options
        if key != self.previous:
            sessions = self.data.sessions(row)
            positions = [
                (i, j)
                for i, session in enumerate(sessions)
                for j, m in enumerate(session)
                if m["role"] == "user"
            ]
            old = []
            if positions:
                i, j = positions[-1]
                old = sessions[:i] + ([sessions[i][:j]] if j else [])

            def render(ss):
                if row["dataset"] == "persona":
                    return "\n".join(serialize(s) for s in ss)
                return "\n\n".join(f"Session {i + 1}:\n{serialize(s)}" for i, s in enumerate(ss))

            full = self.data.history(row)
            old_adapter, full_adapter = self.compile(render(old)), self.compile(full)
            self.global_adapter = global_update(full_adapter, old_adapter, cfg.alpha, cfg.beta)
            self.units = segment_memory(full, self.ctx_tokenizer, cfg.max_memory_unit_tokens)
            self.previous = key
        text = question(row)
        activated = activate_memory(
            text, self.units, k1=cfg.lexical_k1, recency_margin=cfg.recency_margin, mode="lexical"
        )
        local = self.compile(activated.text)
        ids = b.tokenizer.apply_chat_template(
            [dict(role="user", content=text)],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=False,
        ).to(b.device)
        if ids.shape[1] > b.model.config.max_position_embeddings:
            raise ValueError("PLUME answer input exceeds context capacity")
        values = []
        for adapter in [self.global_adapter, local]:
            b.install(adapter)
            values.append(
                b.model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    use_cache=False,
                    logits_to_keep=1,
                )
                .logits[0, -1]
                .float()
                .log_softmax(-1)
            )
        global_logp, local_logp = values
        js = js_divergence(local_logp, global_logp)
        weight = cfg.lambda_max * js / (js + cfg.tau)
        labels = [
            b.tokenizer.encode(chr(65 + i), add_special_tokens=False)
            for i in range(len(row["options"]))
        ]
        if any(len(v) != 1 for v in labels):
            raise ValueError("Expected single-token option labels")
        return (global_logp + weight * local_logp)[[v[0] for v in labels]]
