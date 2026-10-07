"""Small, explicit commands for data preparation, memory learning and evaluation."""

import argparse
import json
import shutil
from pathlib import Path

from .config import load_config
from .data import Dataset, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(prog="dppm")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="List built-in methods")
    p = commands.add_parser("prepare", help="Normalize an official dataset checkout")
    p.add_argument("--dataset", choices=["persona", "prefeval"], required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    for name in ["cache", "train", "evaluate"]:
        p = commands.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
        if name == "cache":
            p.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
        else:
            p.add_argument("--method", default="dppm")
            p.add_argument("--cache")
        if name == "train":
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--resume", action="store_true")
        if name == "evaluate":
            p.add_argument("--checkpoint")
            p.add_argument("--split", default="test")
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--shards", type=int, default=1)
    p = commands.add_parser("export-checkpoint", help="Package a tensor-only legacy memory state")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", default="dppm")
    p.add_argument("--backbone", choices=["qwen4b", "gemma2b", "mistral7b"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser(
        "import-cache", help="Verify tokenization before importing frozen legacy latents"
    )
    for name in ["config", "data", "examples", "latents", "output"]:
        p.add_argument("--" + name, required=True)
    p.add_argument("--split", default="test")
    p = commands.add_parser("merge", help="Audit and merge disjoint evaluation shards")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--outputs", nargs="+", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "methods":
        from .methods import METHODS

        print("\n".join(METHODS))
    elif args.command == "prepare":
        from .data import prepare

        print(prepare(args.dataset, args.source, args.output))
    elif args.command == "export-checkpoint":
        import torch
        from .memory import make_memory
        from .runner import save_weights

        state = torch.load(args.input, map_location="cpu", weights_only=True)
        memory = make_memory(args.method)
        memory.load_state_dict(state.get("memory", state.get("state_dict", state)), strict=True)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save_weights(args.output, memory, args.method, {"backbone": args.backbone}, args.seed)
    elif args.command == "import-cache":
        from .backend import compiler_fingerprint, tokenizer
        from .cache import segment
        from .config import digest

        config, dataset = load_config(args.config), Dataset(args.data)
        tok = tokenizer(config, compiler=config["backbone"] != "qwen4b")
        legacy = {r["id"]: r for r in read_jsonl(args.examples)}
        output = Path(args.output)
        (output / "latents").mkdir(parents=True, exist_ok=True)
        mapped, chunks, exported = {}, set(), []
        for row in dataset.split(args.split):
            old = legacy[row["id"]]
            if any(old[k] != row[k] for k in ["query", "options", "label"]):
                raise ValueError(f"Legacy example differs: {row['id']}")
            ids = []
            for sid in row["sessions"]:
                if sid not in mapped:
                    mapped[sid] = [
                        key
                        for key, _ in segment(
                            dataset.store[sid], tok, config.get("event_tokens", 4000)
                        )
                    ]
                ids.extend(mapped[sid])
            if ids != old["segments"]:
                raise ValueError(f"Segmentation changed: {row['id']}")
            for sid in ids:
                if sid in chunks:
                    continue
                src, dst = Path(args.latents) / (sid + ".pt"), output / "latents" / (sid + ".pt")
                if not dst.exists():
                    shutil.copyfile(src, dst)
                if digest(src) != digest(dst):
                    raise ValueError("Imported latent checksum differs")
                chunks.add(sid)
            exported.append({**row, "segments": ids})
        write_jsonl(output / "examples.jsonl", exported)
        (output / "manifest.json").write_text(
            json.dumps(
                dict(
                    fingerprint=compiler_fingerprint(config),
                    examples=len(exported),
                    segments=len(chunks),
                    imported=True,
                    tokenization_and_labels_verified=True,
                ),
                indent=2,
            )
        )
        print(f"Verified {len(exported)} examples and {len(chunks)} compiled segments")
    elif args.command == "merge":
        from .config import digest
        from .runner import atomic, metrics

        target = {r["id"]: r for r in Dataset(args.data).split(args.split)}
        records, identities = {}, []
        for directory in map(Path, args.outputs):
            mark = json.loads((directory / "complete.json").read_text())
            if digest(directory / "predictions.jsonl") != mark["predictions_sha256"]:
                raise ValueError("Prediction checksum mismatch")
            identities.append(json.loads((directory / "manifest.json").read_text()))
            for row in read_jsonl(directory / "predictions.jsonl"):
                if (
                    row["id"] in records
                    or row["id"] not in target
                    or row["label"] != target[row["id"]]["label"]
                ):
                    raise ValueError("Duplicate, unknown or mislabeled prediction")
                records[row["id"]] = row
        if set(records) != set(target):
            raise ValueError("Incomplete shard coverage")
        for key in [
            "method",
            "backbone",
            "split",
            "data_sha256",
            "sessions_sha256",
            "config_sha256",
            "checkpoint_sha256",
            "cache_manifest_sha256",
            "compiler_sha256",
            "shards",
        ]:
            if len({r[key] for r in identities}) != 1:
                raise ValueError("Mixed evaluation settings")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        write_jsonl(output / "predictions.jsonl", [records[k] for k in target])
        atomic(output / "metrics.json", metrics(list(records.values())))
        atomic(
            output / "complete.json",
            dict(
                count=len(records),
                full_split=True,
                predictions_sha256=digest(output / "predictions.jsonl"),
            ),
        )
    else:
        config, dataset = load_config(args.config), Dataset(args.data)
        if args.command == "cache":
            from .backend import D2LBackend
            from .cache import build_cache

            build_cache(config, dataset, args.output, D2LBackend(config, encoder=True), args.split)
        elif args.command == "train":
            from .runner import train

            train(config, dataset, args.cache, args.method, args.output, args.seed, args.resume)
        else:
            from .runner import evaluate

            evaluate(
                config,
                dataset,
                args.method,
                args.output,
                args.cache,
                args.checkpoint,
                args.split,
                args.shard,
                args.shards,
            )


if __name__ == "__main__":
    main()
