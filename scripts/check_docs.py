"""Check relative Markdown links, local images, JSON and source provenance."""
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
errors = []
links = 0
for path in ROOT.rglob("*.md"):
    if any(p in {".git", "artifacts", "build", "dist"} or p.endswith(".egg-info") for p in path.relative_to(ROOT).parts):
        continue
    for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
        target = target.split(' "')[0].strip("<>")
        if "://" in target or target.startswith(("#", "mailto:")):
            continue
        target = unquote(target.split("#")[0])
        if target and not (path.parent / target).exists():
            errors.append(f"{path.relative_to(ROOT)}: missing {target}")
        links += 1
manifest = json.loads((ROOT / "docs/SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for row in manifest["files"]:
    path = ROOT / row["release"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row["release_sha256"]:
        errors.append("source manifest mismatch: " + row["release"])
for path in (ROOT / "worldledger/protocols").rglob("*.json"):
    json.loads(path.read_text(encoding="utf-8"))
print(json.dumps({"relative_links_checked": links, "source_files_checked": len(manifest["files"]),
                  "errors": errors, "passed": not errors}, indent=2))
raise SystemExit(bool(errors))
