"""En cas d'échec, publie la fin du journal de build en annotations (lisibles via l'API des check-runs)."""
import re
import sys
from pathlib import Path


def esc(s: str) -> str:
    return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


lines = []
for p in sys.argv[1:]:
    f = Path(p)
    if f.exists():
        lines += f.read_text(errors="replace").splitlines()
if not lines:
    print("::error title=log::journal vide")
    sys.exit(0)

errs = [l for l in lines if re.search(r"(error|Error|ERROR|FAILURE|Exception|What went wrong)", l)]
if errs:
    print(f"::error title=errors::{esc(chr(10).join(errs[-60:])[-3800:])}")
tail = "\n".join(lines[-160:])
chunks = [tail[i:i + 3800] for i in range(0, len(tail), 3800)][-6:]
for i, c in enumerate(chunks):
    print(f"::error title=tail-{i}::{esc(c)}")
