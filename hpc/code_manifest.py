"""
Fingerprint list of the code (md5 of every .py / .sh / .sbatch, line endings normalised to LF), so that "the code on my PC" and
"the code on UBELIX" can be compared with one short string, and so every result can name the exact code that made it.

  python code_manifest.py --root <Code folder> [--hash | --out CODE_MANIFEST.md5]
Trees covered: ImageSegmentation/{scripts,hpc}, ShapeAnalysis/{scripts,hpc} (not _archive, not __pycache__).
Same format as `md5sum` writes on the cluster (hash, two spaces, ./relative/path), sorted by path.
"""
import argparse
import hashlib
from pathlib import Path

TREES = ("ImageSegmentation/scripts", "ImageSegmentation/hpc", "ShapeAnalysis/scripts", "ShapeAnalysis/hpc")
SUFFIXES = {".py", ".sh", ".sbatch"}
SKIP = {"_archive", "__pycache__"}


def manifest_lines(root):
    rows = []
    for tree in TREES:
        for p in (root / tree).rglob("*"):
            rel = p.relative_to(root)
            if p.is_file() and p.suffix in SUFFIXES and not SKIP & set(rel.parts):
                digest = hashlib.md5(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                rows.append((rel.as_posix(), f"{digest}  ./{rel.as_posix()}"))
    return [line for _, line in sorted(rows)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--hash", action="store_true", help="print only the 8-character fingerprint of the whole list")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = "\n".join(manifest_lines(args.root)) + "\n"
    if args.out:
        args.out.write_text(text, newline="\n")
    if args.hash:
        print(hashlib.md5(text.encode()).hexdigest()[:8])
    elif not args.out:
        print(text, end="")


if __name__ == "__main__":
    main()
