"""Conservative local release scan. Reports locations and rule names, never values."""
from pathlib import Path
import argparse
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SKIP = {".git", "__pycache__", ".pytest_cache", ".venv", "venv", "build", "dist", "artifacts", "results"}
PATTERNS = {
    "provider_token": re.compile(rb"\b(?:sk-(?:proj-|or-v1-)?|gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_-]{20,}"),
    "aws_access_id": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "private_email": re.compile(rb"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "personal_windows_path": re.compile(rb"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s\"']+", re.I),
    "personal_unix_path": re.compile(rb"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "url_credentials": re.compile(rb"https?://[^\s/:]+:[^\s/@]+@", re.I),
    "private_network_url": re.compile(rb"https?://(?:127\.0\.0\.1|localhost|10\.[0-9.]+|192\.168\.[0-9.]+|172\.(?:1[6-9]|2[0-9]|3[01])\.[0-9.]+)(?::|/|\b)", re.I),
    "embedded_secret_assignment": re.compile(rb"(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*[\"'][A-Za-z0-9_./+\-=]{12,}[\"']", re.I),
}


def candidate_files(root):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP or part.startswith((".venv", ".package_cache")) or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.is_file() or path.is_symlink():
            yield path


def scan_bytes(name, data):
    return [{"file": name, "line": data[:m.start()].count(b"\n") + 1, "rule": rule}
            for rule, pattern in PATTERNS.items() for m in pattern.finditer(data)]


def scan_tree(root):
    findings = []
    count = 0
    for path in candidate_files(root):
        rel = path.relative_to(root).as_posix()
        count += 1
        if path.is_symlink():
            findings.append({"file": rel, "rule": "symlink"})
            continue
        name = path.name.lower()
        if (name == ".env" or (name.startswith(".env.") and name != ".env.example")
                or (name.startswith("api") and name.endswith(".txt"))
                or name.endswith((".pem", ".key", ".p12", ".pfx"))
                or "credentials" in name):
            findings.append({"file": rel, "rule": "private_filename"})
        findings.extend(scan_bytes(rel, path.read_bytes()))
    return count, findings


def scan_index(root):
    result = subprocess.run(["git", "-C", str(root), "ls-files", "--stage", "-z"],
                            capture_output=True, check=True)
    findings = []
    count = 0
    for record in result.stdout.split(b"\0"):
        if not record:
            continue
        meta, name = record.split(b"\t", 1)
        mode, sha, stage = meta.split()
        rel = name.decode("utf-8")
        if mode != b"100644" and mode != b"100755":
            findings.append({"file": rel, "rule": "nonregular_git_entry"})
        data = subprocess.run(["git", "-C", str(root), "cat-file", "blob", sha.decode()],
                              capture_output=True, check=True).stdout
        findings.extend(scan_bytes(rel, data))
        count += 1
    return count, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="also scan the exact Git index blobs")
    args = parser.parse_args()
    count, findings = scan_tree(ROOT)
    staged_count = 0
    if args.staged:
        staged_count, staged_findings = scan_index(ROOT)
        findings.extend(staged_findings)
    print(json.dumps({"files_scanned": count, "staged_blobs_scanned": staged_count,
                      "findings": findings, "passed": not findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
