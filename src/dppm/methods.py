"""Common score(row) interface for all main-table method families."""

import json
from pathlib import Path

import torch

from .backend import D2LBackend, context_ids, prompt_ids, tokenizer
from .cache import LatentCache
from .config import import_source
from .data import serialize
from .memory import TRAINABLE_METHODS

METHODS = (
    *TRAINABLE_METHODS,
    "no_context",
    "gold_state",
    "full_context",
    "rag",
    "rolling_summary",
    "mem0",
    "lightmem",
    "direct_d2l",
    "plume",
    "delta_mem",
    "metis",
)


class ParametricMethod:
    def __init__(self, method, config, dataset, cache_dir, checkpoint):
        from .runner import load_weights

        if not cache_dir or not checkpoint:
            raise ValueError("Trained memories require --cache and --checkpoint")
        self.backend = D2LBackend(config)
        self.cache = LatentCache(cache_dir, self.backend.fingerprint())
        self.memory = load_weights(checkpoint, method, config, self.backend.device).eval()
        self.previous = None
        self.adapter = None
        torch.backends.cuda.matmul.allow_tf32 = method in {"rpmem", "rpmem_matched"}
        torch.backends.cudnn.allow_tf32 = True

    def score(self, row):
        key = tuple(self.cache.rows[row["id"]]["segments"])
        if key != self.previous:
            q = self.cache.load(row, self.backend.device)
            self.adapter = self.backend.decode(self.memory(q))
            self.previous = key
        logits = self.backend.logits(prompt_ids(self.backend.tokenizer, row), self.adapter)
        return logits[self.backend.labels[: len(row["options"])]]


class Retriever:
    def __init__(self, config):
        from transformers import AutoModel, AutoTokenizer

        self.device = config.get("device", "cuda:0")
        name = config.get("retriever", "BAAI/bge-m3")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = (
            AutoModel.from_pretrained(name, torch_dtype=torch.float16).to(self.device).eval()
        )

    @torch.no_grad()
    def embed(self, texts):
        parts = []
        for i in range(0, len(texts), 8):
            x = self.tokenizer(
                texts[i : i + 8],
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            ).to(self.device)
            parts.append(
                torch.nn.functional.normalize(
                    self.model(**x).last_hidden_state[:, 0].float(), dim=-1
                ).cpu()
            )
        return torch.cat(parts)


class TextMethod:
    def __init__(self, method, config, dataset, output):
        from transformers import AutoModelForCausalLM

        self.method, self.config, self.data = method, config, dataset
        self.device = config.get("device", "cuda:0")
        self.tokenizer = tokenizer(config)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                config["base_model"],
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation=config.get("attention", "flash_attention_2"),
            )
            .eval()
            .requires_grad_(False)
        )
        self.retriever = Retriever(config) if method in {"rag", "mem0", "lightmem"} else None
        self.cache = {}
        self.output = Path(output)
        self.store = None
        if method in {"mem0", "lightmem"}:
            from .integrations.stores import Store

            self.store = Store(method, self, config, self.output / "stores")

    def generate(self, messages, max_tokens=1024, response_format=None):
        ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        if len(ids) + max_tokens > self.model.config.max_position_embeddings:
            raise ValueError("Provider input exceeds context capacity; refusing truncation")
        x = torch.tensor([ids], device=self.device)
        extra = {}
        if response_format:
            import xgrammar as xgr
            from xgrammar.contrib.hf import LogitsProcessor

            info = xgr.TokenizerInfo.from_huggingface(
                self.tokenizer,
                vocab_size=self.model.config.vocab_size,
                stop_token_ids=self.tokenizer.eos_token_id,
            )
            grammar = xgr.GrammarCompiler(info).compile_json_schema(
                {"type": "object", "additionalProperties": True},
                strict_mode=False,
                any_whitespace=True,
                max_whitespace_cnt=None,
            )
            extra["logits_processor"] = [LogitsProcessor(grammar)]
        y = self.model.generate(
            input_ids=x,
            attention_mask=torch.ones_like(x),
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            **extra,
        )
        return self.tokenizer.decode(y[0, len(ids) :], skip_special_tokens=True), {
            "prompt_tokens": len(ids),
            "completion_tokens": y.shape[1] - len(ids),
            "total_tokens": y.shape[1],
        }

    def memory(self, row):
        method = self.method
        if method == "no_context":
            return ""
        if method == "gold_state":
            if "gold" not in row:
                raise ValueError("Gold State requires supplied annotations")
            return json.dumps(row["gold"], ensure_ascii=False, indent=2)
        if method == "full_context":
            return self.data.history(row)
        if self.store:
            return self.store.retrieve(self.data.sessions(row), row["query"])
        key = tuple(row["sessions"])
        if method == "rag":
            if key not in self.cache:
                messages = [m for s in self.data.sessions(row) for m in s]
                units = [serialize(messages[i : i + 2]) for i in range(0, len(messages), 2)]
                self.cache[key] = (units, self.retriever.embed(units))
            units, vectors = self.cache[key]
            order = torch.argsort(
                vectors @ self.retriever.embed([row["query"]])[0], descending=True, stable=True
            )[:10].tolist()
            return "\n\n".join(units[i] for i in order)
        if key not in self.cache:
            value = ""
            for session in self.data.sessions(row):
                text = (
                    "Update the user memory summary using the new session. Preserve concrete "
                    "preferences, entities, changes, and requests to forget. Distinguish the user "
                    "from other people. Keep the summary within 1024 tokens. Return only the "
                    f"updated summary.\n\nPrevious summary:\n{value}\n\nNew session:\n{serialize(session)}"
                )
                value, _ = self.generate([dict(role="user", content=text)], 1024)
            self.cache[key] = value
        return self.cache[key]

    def score(self, row):
        ids = prompt_ids(self.tokenizer, row, self.memory(row))
        if len(ids) > self.model.config.max_position_embeddings:
            raise ValueError("Answer input exceeds native context; truncation is disabled")
        labels = [
            self.tokenizer.encode(chr(65 + i), add_special_tokens=False)
            for i in range(len(row["options"]))
        ]
        if any(len(x) != 1 for x in labels):
            raise ValueError("Expected single-token option labels")
        x = torch.tensor([ids], device=self.device)
        logits = self.model(
            input_ids=x, attention_mask=torch.ones_like(x), use_cache=False, logits_to_keep=1
        ).logits[0, -1]
        return logits[[x[0] for x in labels]]


class DirectD2L:
    def __init__(self, config, dataset):
        self.backend, self.data = D2LBackend(config, encoder=True), dataset
        self.previous = None

    def score(self, row):
        from ctx_to_lora.data.processing import split_too_long_ctx
        from ctx_to_lora.data.definitions import CTX_AFFIXES

        b = self.backend
        key = tuple(row["sessions"])
        if key != self.previous:
            ids = context_ids(b.compiler_tokenizer, self.data.history(row))
            official = b.config["model_id"]
            affix = CTX_AFFIXES[official]
            budget = min(
                8192,
                b.model.config.max_position_embeddings
                - len(affix["prefix"])
                - len(affix["suffix"]),
            )
            chunks = split_too_long_ctx(
                dict(ctx_ids=ids),
                official,
                num_chunk_probs=None,
                max_chunk_len=budget,
                min_chunk_len=-1,
                max_num_split=None,
                is_train=False,
            )["ctx_ids"]
            pieces = [b.decode(b.encode(chunk).to(b.device), add_bias=False) for chunk in chunks]
            raw = {
                m: {k: torch.cat([p[m][k] for p in pieces], dim=0) for k in ["A", "B"]}
                for m in pieces[0]
            }
            self.adapter, self.previous = b.assemble(raw), key
        return b.logits(prompt_ids(b.tokenizer, row), self.adapter)[b.labels[: len(row["options"])]]


def create_method(method, config, dataset, cache_dir=None, checkpoint=None, output="runs"):
    if method in TRAINABLE_METHODS:
        return ParametricMethod(method, config, dataset, cache_dir, checkpoint)
    if method in {
        "no_context",
        "gold_state",
        "full_context",
        "rag",
        "rolling_summary",
        "mem0",
        "lightmem",
    }:
        return TextMethod(method, config, dataset, output)
    if method == "direct_d2l":
        return DirectD2L(config, dataset)
    if method in {"metis", "delta_mem"}:
        from .integrations.native import NativeMemory

        return NativeMemory(method, config, dataset)
    if method == "plume":
        from .integrations.plume import Plume

        return Plume(config, dataset)
    if ":" in method:
        module, name = method.split(":", 1)
        return getattr(import_source(None, module), name)(
            config=config, dataset=dataset, output=output
        )
    raise ValueError(f"Unknown method {method!r}; available: {METHODS}")
