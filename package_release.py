#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
package_release.py
==================
Rebuilds the Word (.docx) documents, generates FILE_LIST_GitHub_Zenodo.txt
with exact SHA-256 checksums and file sizes, and packages the complete
GitHub/Zenodo v2.0.0 release archive.
"""

import hashlib
import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent
ZIP_OUT = ROOT.parent / "maintained-coupling-kuramoto-v2.0.0.zip"


def sha256_of_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    # 1. Rebuild all Word (.docx) documents
    subprocess.run([sys.executable, str(ROOT / "build_manuscript_docx.py")], check=True)

    # 2. Collect all repository files (excluding __pycache__ and .pyc)
    files = sorted(
        [
            p
            for p in ROOT.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not p.name.endswith(".pyc")
            and p.name != "FILE_LIST_GitHub_Zenodo.txt"
        ],
        key=lambda p: str(p.relative_to(ROOT)).lower(),
    )

    lines = [
        "========================================================================================",
        "GITHUB & ZENODO REPRODUCIBILITY PACKAGE MANIFEST (v2.0.0)",
        "Repository: https://github.com/lgonzalezpaz/maintained-coupling-kuramoto",
        "Zenodo DOI: 10.5281/zenodo.22928314",
        "Title: Memory as maintained coupling: a Kuramoto model formalization of synchronization",
        "       dynamics and developmental pruning in neural systems",
        "========================================================================================",
        "",
        f"{'Relative Path':<48} | {'Size (Bytes)':>14} | {'SHA-256 Checksum'}",
        "-" * 136,
    ]

    for fpath in files:
        rel = fpath.relative_to(ROOT).as_posix()
        size = fpath.stat().st_size
        digest = sha256_of_file(fpath)
        lines.append(f"{rel:<48} | {size:>14,d} | {digest}")

    lines.append("-" * 136)
    lines.append(f"Total verified files: {len(files)}")
    manifest_path = ROOT / "FILE_LIST_GitHub_Zenodo.txt"
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path} ({len(files)} files indexed)")

    # Include manifest in zip
    files.append(manifest_path)

    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fpath in files:
            arcname = f"maintained-coupling-kuramoto/{fpath.relative_to(ROOT).as_posix()}"
            zf.write(fpath, arcname=arcname)

    print(f"Created release archive: {ZIP_OUT} ({ZIP_OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
