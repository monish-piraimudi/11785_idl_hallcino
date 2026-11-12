"""Create a small source-only archive for copying to a GPU/Linux host."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = (
    ".gitignore",
    ".github",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "hallucinot",
    "scripts/run_500.sh",
    "scripts/generate_dummy_safetensors.py",
    "scripts/package_scp.py",
    "tests",
    "data",
    "docs",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest-tmp", ".venv", "venv", "runs"}


def package(output: Path) -> tuple[Path, str]:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for entry in INCLUDE:
        source = ROOT / entry
        if not source.exists():
            raise FileNotFoundError(f"required package path is missing: {source}")
        candidates = [source] if source.is_file() else sorted(source.rglob("*"))
        for candidate in candidates:
            if not candidate.is_file() or any(part in EXCLUDED_PARTS for part in candidate.parts):
                continue
            if candidate.suffix == ".pyc" or candidate.name.endswith(".zip"):
                continue
            files.append(candidate)

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            archive.write(source, Path(ROOT.name) / source.relative_to(ROOT))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return output, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "hallucinot-scp.zip"))
    args = parser.parse_args()
    path, digest = package(Path(args.output))
    print(f"Created {path} ({path.stat().st_size:,} bytes)")
    print(f"SHA-256 {digest}")
    print("The archive contains source, docs, tests, and fixtures; no model weights, run data, or API keys.")


if __name__ == "__main__":
    main()
