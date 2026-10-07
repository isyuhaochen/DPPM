"""Content-addressed frozen latents; never share caches between compiler releases."""

import json
from functools import lru_cache
from pathlib import Path

import torch

from .backend import context_ids, tokenizer
from .data import hash_text, write_jsonl


def segment(messages, tok, event_limit=4000, max_tokens=4096):
    events = []
    for message in messages:
        text = f"{message['role']}: {message.get('content', '')}"
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) > event_limit:
            events.extend(
                tok.decode(ids[i : i + event_limit], skip_special_tokens=False)
                for i in range(0, len(ids), event_limit)
            )
        else:
            events.append(text)
    chunks, current = [], []
    for event in events:
        if current and len(context_ids(tok, "\n".join(current + [event]))) > max_tokens:
            chunks.append(current)
            current = current[-1:]
            if len(context_ids(tok, "\n".join(current + [event]))) > max_tokens:
                current = []
        current.append(event)
    if current:
        chunks.append(current)
    result = []
    for chunk in chunks:
        ids = context_ids(tok, "\n".join(chunk))
        if len(ids) > max_tokens:
            raise ValueError("A segment exceeds the compiler budget")
        result.append((hash_text(json.dumps(ids, separators=(",", ":"))), ids))
    return result


class LatentCache:
    def __init__(self, directory, fingerprint=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if fingerprint and self.manifest["fingerprint"] != fingerprint:
            raise ValueError("Cache belongs to a different compiler or tokenizer")
        self.rows = {
            r["id"]: r for r in (json.loads(x) for x in (self.directory / "examples.jsonl").open())
        }

    @lru_cache(maxsize=128)
    def _latent(self, sid):
        return torch.load(
            self.directory / "latents" / (sid + ".pt"), map_location="cpu", weights_only=True
        )

    def load(self, row, device):
        stored = self.rows[row["id"]]
        if stored["sessions"] != row["sessions"]:
            raise ValueError("Cached history differs from evaluation input")
        return torch.stack([self._latent(sid) for sid in stored["segments"]]).to(device)


def build_cache(config, dataset, output, backend, split="all"):
    output = Path(output)
    (output / "latents").mkdir(parents=True, exist_ok=True)
    tok = tokenizer(config, compiler=True) if config["backbone"] != "qwen4b" else tokenizer(config)
    rows = dataset.examples if split == "all" else dataset.split(split)
    manifest = output / "manifest.json"
    if not manifest.exists() and any((output / "latents").iterdir()):
        raise ValueError("Unverified pre-existing latent cache")
    if (
        manifest.exists()
        and json.loads(manifest.read_text())["fingerprint"] != backend.fingerprint()
    ):
        raise ValueError("Existing cache uses another compiler")
    manifest.write_text(json.dumps(dict(fingerprint=backend.fingerprint(), complete=False)))
    session_segments, compiled, exported = {}, set(), []
    for index, row in enumerate(rows):
        sids = []
        for sid in row["sessions"]:
            if sid not in session_segments:
                pieces = segment(dataset.store[sid], tok, config.get("event_tokens", 4000))
                session_segments[sid] = [key for key, _ in pieces]
                for key, ids in pieces:
                    path = output / "latents" / (key + ".pt")
                    if key not in compiled:
                        # Existing tensors may only be resumed within this fingerprint.
                        if not path.exists():
                            temp = path.with_suffix(".tmp")
                            torch.save(backend.encode(ids), temp)
                            temp.replace(path)
                        compiled.add(key)
            sids.extend(session_segments[sid])
        if not sids:
            raise ValueError("A parametric-memory example needs at least one history segment")
        exported.append({**row, "segments": sids})
        if index % 100 == 0:
            print(f"Cached {index + 1}/{len(rows)} examples", flush=True)
    write_jsonl(output / "examples.jsonl", exported)
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                fingerprint=backend.fingerprint(),
                examples=len(exported),
                segments=len(compiled),
                training=False,
                sessions_sha256=hash_text(json.dumps(dataset.store, sort_keys=True)),
            ),
            indent=2,
        )
    )
