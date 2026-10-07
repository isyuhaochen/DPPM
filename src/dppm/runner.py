"""Shared training/evaluation contracts and portable, resumable outputs."""

import collections
import json
import math
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from .backend import D2LBackend, prompt_ids
from .cache import LatentCache
from .config import digest
from .data import hash_text, read_jsonl
from .memory import TRAINABLE_METHODS, make_memory


def atomic(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def save_weights(path, memory, method, config, seed):
    torch.save(
        dict(
            format_version=1,
            method=method,
            backbone=config["backbone"],
            seed=seed,
            state_dict={k: v.detach().cpu() for k, v in memory.state_dict().items()},
        ),
        path,
    )


def load_weights(path, method, config, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1 or checkpoint["method"] != method:
        raise ValueError(
            "Checkpoint method/format mismatch; use `dppm export-checkpoint` for legacy files"
        )
    if checkpoint["backbone"] != config["backbone"]:
        raise ValueError("Checkpoint was trained with a different compiler/backbone")
    memory = make_memory(method).to(device)
    memory.load_state_dict(checkpoint["state_dict"], strict=True)
    return memory


def train(config, dataset, cache_dir, method, output, seed=42, resume=False):
    if method not in TRAINABLE_METHODS:
        raise ValueError("This method is inference-only")
    rows = dataset.split("train")
    if not rows:
        raise ValueError("No training split")
    if {r["user"] for r in rows} & {r["user"] for r in dataset.split("test")}:
        raise ValueError("Train/test users overlap")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and not resume:
        raise FileExistsError("Run exists; use --resume or another output directory")
    backend = D2LBackend(config)
    cache = LatentCache(cache_dir, backend.fingerprint())
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = method in {"rpmem", "rpmem_matched"}
    torch.backends.cudnn.allow_tf32 = True
    memory = make_memory(method, backend.dim).to(backend.device)
    settings = {
        "epochs": 5,
        "lr": 0.001,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        **config.get("training", {}),
    }
    optimizer = torch.optim.AdamW(
        memory.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"]
    )
    identity = dict(
        method=method,
        backbone=config["backbone"],
        seed=seed,
        settings=settings,
        train_ids_sha256=hash_text(json.dumps(sorted(r["id"] for r in rows))),
        data_sha256=digest(dataset.directory / "examples.jsonl"),
        sessions_sha256=digest(dataset.directory / "sessions.jsonl"),
        compiler=backend.fingerprint(),
        parameters=sum(p.numel() for p in memory.parameters()),
    )
    if (output / "manifest.json").exists():
        if json.loads((output / "manifest.json").read_text()) != identity:
            raise ValueError("Refusing to resume a different training recipe")
    else:
        atomic(output / "manifest.json", identity)
    start_epoch = offset = steps = 0
    if resume:
        last = torch.load(output / "last.pt", map_location=backend.device, weights_only=True)
        memory.load_state_dict(last["memory"])
        optimizer.load_state_dict(last["optimizer"])
        start_epoch, offset, steps = last["epoch"], last["offset"], last["steps"]
        torch.set_rng_state(last["cpu_rng"].cpu())
        torch.cuda.set_rng_state(last["gpu_rng"].cpu())

    def checkpoint(epoch, position):
        temp = output / "last.tmp"
        torch.save(
            dict(
                memory=memory.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                offset=position,
                steps=steps,
                cpu_rng=torch.get_rng_state(),
                gpu_rng=torch.cuda.get_rng_state(),
            ),
            temp,
        )
        temp.replace(output / "last.pt")

    with (output / "training.jsonl").open("a", buffering=1) as log:
        for epoch in range(start_epoch, settings["epochs"]):
            rng = random.Random(seed + epoch)
            if rows[0]["dataset"] == "persona":
                by_user = collections.defaultdict(list)
                for row in rows:
                    by_user[row["user"]].append(row)
                users = sorted(by_user)
                rng.shuffle(users)
                order = []
                for user in users:
                    part = list(by_user[user])
                    rng.shuffle(part)
                    order.extend(part)
            else:
                order = list(rows)
                rng.shuffle(order)
            memory.train()
            for position, row in enumerate(order):
                if epoch == start_epoch and position < offset:
                    continue
                optimizer.zero_grad(set_to_none=True)
                q = cache.load(row, backend.device)
                adapter = backend.decode(memory(q))
                logits = backend.logits(prompt_ids(backend.tokenizer, row), adapter)
                legal = logits[backend.labels[: len(row["options"])]]
                loss = F.cross_entropy(
                    legal[None], torch.tensor([row["label"]], device=backend.device)
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(memory.parameters(), settings["grad_clip"])
                if not torch.isfinite(norm) or any(
                    p.grad is not None for p in backend.net.parameters()
                ):
                    raise RuntimeError("Invalid gradients or frozen-parameter gradient leak")
                optimizer.step()
                steps += 1
                if steps % 50 == 0:
                    event = dict(epoch=epoch + 1, step=steps, loss=float(loss.detach()))
                    log.write(json.dumps(event) + "\n")
                    print(event, flush=True)
                if steps % 250 == 0:
                    checkpoint(epoch, position + 1)
                del q, adapter, logits, legal, loss
            checkpoint(epoch + 1, 0)
            offset = 0
    save_weights(output / "memory.pt", memory, method, config, seed)
    atomic(
        output / "complete.json",
        dict(
            epochs=settings["epochs"],
            steps=steps,
            checkpoint_sha256=digest(output / "memory.pt"),
            frozen_parameter_gradients=0,
        ),
    )


def metrics(rows):
    def accuracy(part):
        return 100 * sum(r["correct"] for r in part) / len(part) if part else None

    values = {"Overall": accuracy(rows), "count": len(rows)}
    if rows and rows[0]["dataset"] == "persona":
        values.update(
            Self=accuracy([r for r in rows if r.get("who") == "self"]),
            Current=accuracy([r for r in rows if r.get("updated") == "False"]),
        )
        values["types"] = {
            t: accuracy([r for r in rows if r.get("pref_type") == t])
            for t in sorted({r["pref_type"] for r in rows if "pref_type" in r})
        }
    elif rows and rows[0]["dataset"] == "prefeval":
        values["intervals"] = {
            str(t): accuracy([r for r in rows if r.get("turns") == t]) for t in [10, 70, 300]
        }
        values["topics"] = {
            t: accuracy([r for r in rows if r.get("topic") == t])
            for t in sorted({r["topic"] for r in rows if "topic" in r})
        }
    return values


def evaluate(
    config,
    dataset,
    method,
    output,
    cache_dir=None,
    checkpoint=None,
    split="test",
    shard=0,
    shards=1,
):
    from .methods import create_method

    if not 0 <= shard < shards:
        raise ValueError("Invalid shard index")
    selected = dataset.split(split)[shard::shards]
    if not selected:
        raise ValueError("Empty evaluation selection")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        method=method,
        backbone=config["backbone"],
        split=split,
        shard=shard,
        shards=shards,
        count=len(selected),
        data_sha256=digest(dataset.directory / "examples.jsonl"),
        config_sha256=hash_text(json.dumps(config, sort_keys=True)),
        sessions_sha256=digest(dataset.directory / "sessions.jsonl"),
        checkpoint_sha256=digest(checkpoint) if checkpoint else None,
        cache_manifest_sha256=digest(Path(cache_dir) / "manifest.json") if cache_dir else None,
        compiler_sha256=digest(config["compiler_checkpoint"])
        if config.get("compiler_checkpoint") and Path(config["compiler_checkpoint"]).is_file()
        else None,
    )
    manifest = output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Output directory belongs to another evaluation")
    atomic(manifest, identity)
    predictor = create_method(method, config, dataset, cache_dir, checkpoint, output)
    path = output / "predictions.jsonl"
    done = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
    if not set(done) <= {r["id"] for r in selected}:
        raise ValueError("Saved predictions do not match this shard")
    with path.open("a", buffering=1) as stream, torch.inference_mode():
        for row in selected:
            if row["id"] in done:
                continue
            values = predictor.score(row).float().cpu().tolist()
            if len(values) != len(row["options"]) or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid legal-option scores")
            pred = max(range(len(values)), key=values.__getitem__)
            record = {
                k: row[k]
                for k in [
                    "id",
                    "dataset",
                    "user",
                    "label",
                    "who",
                    "updated",
                    "pref_type",
                    "topic",
                    "form",
                    "turns",
                ]
                if k in row
            }
            record.update(
                prediction=pred, correct=pred == row["label"], logits=values, truncated=False
            )
            stream.write(json.dumps(record) + "\n")
            done[row["id"]] = record
            if len(done) % 100 == 0:
                print(f"Evaluated {len(done)}/{len(selected)}", flush=True)
    result = metrics(list(done.values()))
    atomic(output / "metrics.json", result)
    atomic(
        output / "complete.json",
        dict(count=len(done), full_split=shards == 1, predictions_sha256=digest(path)),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result
