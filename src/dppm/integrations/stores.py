"""Official Mem0/LightMem stores with local frozen-model providers."""

import copy
import datetime
import json
import re
from functools import wraps
from pathlib import Path

from ..config import import_source
from ..data import hash_text

_runtime = None  # Upstream provider factories are process-global: one store evaluator per process.


def checked_provider(fn):
    """Surface provider failures even when an upstream worker catches them."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            _runtime.provider_errors.append(f"{type(exc).__name__}: {exc}")
            raise

    return wrapped


class LocalEmbedder:
    def __init__(self, config=None):
        self.config = config

    @checked_provider
    def embed(self, text, **kwargs):
        return _runtime.retriever.embed([text])[0].tolist()

    @checked_provider
    def embed_batch(self, texts, **kwargs):
        return _runtime.retriever.embed(texts).tolist()

    def get_stats(self):
        return dict(total_calls=0, total_tokens=0)


class Mem0LLM:
    def __init__(self, config):
        self.config = config

    @checked_provider
    def generate_response(self, messages, response_format=None, tools=None, **kwargs):
        if tools:
            raise ValueError("Tool calls are not part of the local memory provider")
        text, _ = _runtime.generate(messages, self.config.max_tokens, response_format)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError("Memory extraction returned incomplete JSON") from None
        if not isinstance(payload, dict):
            raise ValueError("Memory extraction must return a JSON object")
        if payload.get("memory"):
            values = payload["memory"]
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ValueError("Malformed extracted memory records")
        return text


class Store:
    def __init__(self, method, runtime, config, root):
        global _runtime
        _runtime = runtime
        runtime.provider_errors = []
        self.method, self.runtime, self.config = method, runtime, config
        self.root = Path(root) / method
        self.root.mkdir(parents=True, exist_ok=True)
        self.previous, self.instance = None, None
        settings = config.get("external", {}).get(method, {})
        if method == "mem0":
            import_source(settings.get("source"), "mem0")
            from mem0.utils.factory import LlmFactory, EmbedderFactory
            from mem0.configs.llms.base import BaseLlmConfig

            LlmFactory.provider_to_class["langchain"] = (
                "dppm.integrations.stores.Mem0LLM",
                BaseLlmConfig,
            )
            EmbedderFactory.provider_to_class["huggingface"] = (
                "dppm.integrations.stores.LocalEmbedder"
            )
        else:
            import_source(settings.get("source"), "lightmem")
            from lightmem.factory.memory_manager.openai import OpenaiManager
            from lightmem.factory.memory_manager.factory import MemoryManagerFactory
            from lightmem.factory.text_embedder.factory import TextEmbedderFactory

            class LocalManager(OpenaiManager):
                def __init__(self, config):
                    self.config, self.tokenizer, self.client = (
                        config,
                        runtime.tokenizer,
                        runtime.model,
                    )

                @checked_provider
                def generate_response(self, messages, response_format=None, tools=None):
                    if tools:
                        raise ValueError("Unexpected tool calls")
                    messages = copy.deepcopy(messages)
                    messages[0]["content"] = (
                        "Timestamps dated 2000-01-01 are synthetic ordering indices only, "
                        "not real dates of events. Do not treat them as factual evidence.\n"
                        + messages[0]["content"]
                    )
                    return runtime.generate(messages, self.config.max_tokens, response_format)

            globals()["LightManager"] = LocalManager
            MemoryManagerFactory._MODEL_MAPPING["openai"] = "dppm.integrations.stores.LightManager"
            TextEmbedderFactory._MODEL_MAPPING["huggingface"] = (
                "dppm.integrations.stores.LocalEmbedder"
            )

    def initialize(self, path):
        model = self.config.get("retriever", "BAAI/bge-m3")
        vectors = dict(
            collection_name="memory",
            embedding_model_dims=1024,
            path=str(path / "vectors"),
            on_disk=True,
        )
        if self.method == "mem0":
            from mem0 import Memory

            return Memory.from_config(
                dict(
                    llm=dict(
                        provider="langchain",
                        config=dict(model="local-frozen-backbone", max_tokens=2048, temperature=0),
                    ),
                    embedder=dict(
                        provider="huggingface", config=dict(model=model, embedding_dims=1024)
                    ),
                    vector_store=dict(provider="qdrant", config=vectors),
                    history_db_path=str(path / "history.sqlite"),
                )
            )
        import lightmem.memory.lightmem as upstream

        upstream.GLOBAL_TOPIC_IDX = 0
        upstream.GLOBAL_LAST_SUMMARY_TIME = None
        return upstream.LightMemory.from_config(
            dict(
                pre_compress=True,
                pre_compressor=dict(
                    model_name="llmlingua-2",
                    configs=dict(
                        llmlingua_config=dict(
                            model_name=self.config.get(
                                "compressor",
                                "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                            ),
                            device_map=self.runtime.device,
                            use_llmlingua2=True,
                        )
                    ),
                ),
                topic_segment=True,
                precomp_topic_shared=True,
                topic_segmenter=dict(model_name="llmlingua-2"),
                messages_use="user_only",
                metadata_generate=True,
                text_summary=True,
                memory_manager=dict(
                    model_name="openai",
                    configs=dict(
                        model="local-frozen-backbone",
                        max_tokens=2048,
                        temperature=0,
                        do_sample=False,
                    ),
                ),
                extract_threshold=0.1,
                index_strategy="embedding",
                text_embedder=dict(
                    model_name="huggingface", configs=dict(model=model, embedding_dims=1024)
                ),
                logging=dict(level="WARNING", console_level="WARNING", file_enabled=False),
                retrieve_strategy="embedding",
                embedding_retriever=dict(model_name="qdrant", configs=vectors),
                update="offline",
            )
        )

    def retrieve(self, sessions, query):
        def check_errors():
            if self.runtime.provider_errors:
                raise RuntimeError("Memory provider failed: " + self.runtime.provider_errors[0])

        check_errors()
        key = hash_text(json.dumps(sessions, ensure_ascii=False, sort_keys=True))
        path = self.root / key
        path.mkdir(exist_ok=True)
        if key != self.previous:
            if self.instance is not None:
                client = (
                    self.instance.vector_store.client
                    if self.method == "mem0"
                    else self.instance.embedding_retriever.client
                )
                client.close()
            if (path / "started.json").exists() and not (path / "complete.json").exists():
                raise RuntimeError(
                    "Partial store detected; use a new output directory to avoid duplicate writes"
                )
            self.instance, self.previous = self.initialize(path), key
        mem = self.instance
        if not (path / "complete.json").exists():
            (path / "started.json").write_text(json.dumps(dict(history_sha256=key)))
            for i, session in enumerate(sessions):
                if self.method == "mem0":
                    mem.add(copy.deepcopy(session), user_id="visible-history", infer=True)
                else:
                    for j in range(0, len(session), 2):
                        chunk = copy.deepcopy(session[j : j + 2])
                        stamp = (
                            datetime.datetime(2000, 1, 1) + datetime.timedelta(minutes=i)
                        ).strftime("%Y-%m-%d %H:%M:%S")
                        for message in chunk:
                            message["time_stamp"] = stamp
                        last = i == len(sessions) - 1 and j + 2 >= len(session)
                        mem.add_memory(messages=chunk, force_segment=last, force_extract=last)
                check_errors()
            if self.method == "lightmem":
                mem.construct_update_queue_all_entries(max_workers=1)
                mem.offline_update_all_entries(score_threshold=0.8, max_workers=1)
            check_errors()
            (path / "complete.json").write_text(json.dumps(dict(history_sha256=key)))
        if self.method == "mem0":
            result = "\n".join(
                r["memory"]
                for r in mem.search(query, filters={"user_id": "visible-history"}, top_k=10)[
                    "results"
                ]
            )
        else:
            result = "\n".join(
                re.sub(r"^2000-01-01\S*\s+\w+\s+", "", s) for s in mem.retrieve(query, limit=10)
            )
        check_errors()
        return result
