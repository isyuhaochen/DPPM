"""Normalized benchmark input: examples reference deduplicated raw sessions."""

import ast
import csv
import hashlib
import json
import random
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def hash_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def serialize(messages):
    return "\n".join(f"{m['role']}: {m.get('content', '')}" for m in messages)


def parse(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return ast.literal_eval(value)


def permute(options, key):
    order = list(range(len(options)))
    random.Random(int(hash_text("42:" + key)[:16], 16)).shuffle(order)
    return [options[i] for i in order], order.index(0)


class Dataset:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.examples = read_jsonl(self.directory / "examples.jsonl")
        self.store = {r["id"]: r["messages"] for r in read_jsonl(self.directory / "sessions.jsonl")}
        if len({r["id"] for r in self.examples}) != len(self.examples):
            raise ValueError("Example IDs must be unique")
        for row in self.examples:
            if not 2 <= len(row["options"]) <= 8 or not 0 <= row["label"] < len(row["options"]):
                raise ValueError(f"Invalid options/label: {row['id']}")
            if any(s not in self.store for s in row["sessions"]):
                raise ValueError(f"Missing history for {row['id']}")

    def split(self, name):
        return [r for r in self.examples if r["split"] == name]

    def sessions(self, row):
        return [self.store[s] for s in row["sessions"]]

    def history(self, row):
        sessions = self.sessions(row)
        if row["dataset"] == "persona":
            return serialize(sessions[0])
        return "\n\n".join(f"Session {i + 1}:\n{serialize(s)}" for i, s in enumerate(sessions))


def prepare(dataset, source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "examples.jsonl").exists():
        raise FileExistsError(
            "Use an empty output directory; preparation never overwrites datasets"
        )
    rows, sessions = [], {}

    def add_session(messages):
        messages = [{"role": m["role"], "content": m.get("content", "")} for m in messages]
        sid = hash_text(serialize(messages))
        sessions.setdefault(sid, messages)
        return sid

    if dataset == "persona":
        raw = {
            s: list(csv.DictReader((source / f"benchmark/text/{s}.csv").open()))
            for s in ["train", "val", "benchmark"]
        }
        heldout = {r["persona_id"] for r in raw["benchmark"]}
        for split, records in raw.items():
            for index, r in enumerate(records):
                if split != "benchmark" and r["persona_id"] in heldout:
                    continue
                wrong = parse(r["incorrect_answers"]) or []
                wrong = [x.strip() for x in wrong if isinstance(x, str) and x.strip()]
                answer = r["correct_answer"].strip()
                if not wrong or answer in wrong:
                    continue
                iid = f"persona/{split}/{index}"
                options, label = permute([answer] + wrong, iid)
                path = (source / r["chat_history_32k_link"]).resolve()
                if not path.is_relative_to(source.resolve()):
                    raise ValueError("History path escapes dataset directory")
                messages = json.loads(path.read_text())["chat_history"]
                query = parse(r["user_query"])
                query = query["content"] if isinstance(query, dict) else query
                rows.append(
                    dict(
                        id=iid,
                        dataset=dataset,
                        split="test" if split == "benchmark" else split,
                        user=r["persona_id"],
                        query=query,
                        options=options,
                        label=label,
                        sessions=[add_session(messages)],
                        gold={k: r[k] for k in ["preference", "who", "updated", "pref_type"]},
                        **{k: r[k] for k in ["who", "updated", "pref_type", "sensitive_info"]},
                    )
                )
        expected = {"train": 18527, "val": 2059, "test": 5000}
    elif dataset == "prefeval":
        if (source / "benchmark_dataset").exists():
            source /= "benchmark_dataset"
        heldout = {"travel_transportation", "shop_technology", "education_resources", "shop_motors"}
        inter = json.loads((source / "filtered_inter_turns.json").read_text())
        intervals = {}
        for turns in [10, 70, 300]:
            remain, ids = turns * 2, []
            for session in inter:
                messages = session["conversation"][:remain]
                if messages:
                    ids.append(add_session(messages))
                    remain -= len(messages)
                if remain == 0:
                    break
            if remain:
                raise ValueError("Incomplete PrefEval intervening conversations")
            intervals[turns] = ids
        for path in sorted((source / "implicit_preference/choice-based").glob("*.json")):
            topic = path.stem
            tasks = json.loads((source / "mcq_options" / path.name).read_text())
            choices = json.loads(path.read_text())
            personas = json.loads(
                (source / "implicit_preference/persona-driven" / path.name).read_text()
            )
            for index, r in enumerate(tasks):
                c, p = choices[index]["conversation"], personas[index]["conversation"]
                forms = {
                    "explicit": [dict(role="user", content=r["preference"])],
                    "choice": [
                        dict(role=role, content=c[key])
                        for role, key in [
                            ("user", "query"),
                            ("assistant", "assistant_options"),
                            ("user", "user_selection"),
                            ("assistant", "assistant_acknowledgment"),
                        ]
                    ],
                    "persona": [
                        dict(role=role, content=turn[role])
                        for turn in p.values()
                        for role in ["user", "assistant"]
                    ],
                }
                for form, messages in forms.items():
                    sid = add_session(messages)
                    for turns in [10, 70, 300]:
                        iid = f"prefeval/{topic}/{index}/{form}/{turns}"
                        options, label = permute(r["classification_task_options"], iid)
                        rows.append(
                            dict(
                                id=iid,
                                dataset=dataset,
                                split="test" if topic in heldout else "train",
                                user=f"{topic}/{index}",
                                topic=topic,
                                form=form,
                                turns=turns,
                                query=r["question"],
                                options=options,
                                label=label,
                                sessions=[sid] + intervals[turns],
                                gold={"preference": r["preference"]},
                            )
                        )
        expected = {"train": 7380, "test": 1620}
    else:
        raise ValueError("Choose persona or prefeval")
    counts = {s: sum(r["split"] == s for r in rows) for s in expected}
    if counts != expected:
        raise ValueError(f"Unexpected release/splits: {counts}; expected {expected}")
    if {r["user"] for r in rows if r["split"] == "train"} & {
        r["user"] for r in rows if r["split"] == "test"
    }:
        raise ValueError("Train/test user overlap")
    write_jsonl(output / "examples.jsonl", rows)
    write_jsonl(output / "sessions.jsonl", [dict(id=k, messages=v) for k, v in sessions.items()])
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                dataset=dataset,
                counts=counts,
                examples_sha256=hash_text((output / "examples.jsonl").read_text()),
                train_test_user_overlap=False,
            ),
            indent=2,
        )
    )
    return counts
