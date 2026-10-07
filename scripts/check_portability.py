"""Check distributable text for workstation paths, private remotes and embedded secrets."""

from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
patterns = {
    "absolute workstation path": r"/(?:home|root|mnt|scratch|csy[^/]*)/[A-Za-z0-9_]",
    "private remote": r"(?:git" + r"@|ssh:" + r"//|git\.overleaf\.com)",
    "embedded access token": r"(?:hf_[A-Za-z0-9]{25,}|ghp_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9]{24,})",
}
excluded = {
    ".git",
    "__pycache__",
    ".venv",
    "third_party",
    "data",
    "models",
    "artifacts",
    "runs",
    "build",
    "dist",
}
failures = []
for path in root.rglob("*"):
    if not path.is_file() or any(
        p in excluded or p.endswith(".egg-info") for p in path.relative_to(root).parts
    ):
        continue
    if path.suffix not in {
        ".py",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".toml",
        ".cff",
        ".jinja",
        ".txt",
    }:
        continue
    for label, pattern in patterns.items():
        if re.search(pattern, path.read_text()):
            failures.append(f"{path.relative_to(root)}: {label}")
if failures:
    raise SystemExit("\n".join(failures))
print("Portability scan passed")
