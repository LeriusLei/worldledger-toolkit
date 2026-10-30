"""Use actual input-file hashes, then detect a synthetic protocol mutation."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_audit import ResearchAuditHarness
from research_audit.harness import digest


def main():
    with TemporaryDirectory(prefix="worldledger-demo-") as directory:
        root = Path(directory)
        paths = {}
        manifest = {"task_id": "synthetic-audit-demo"}
        for field in ResearchAuditHarness.IDENTITY[1:]:
            path = root / (field + ".txt")
            path.write_text("synthetic input for " + field, encoding="utf-8")
            paths[field] = path
            manifest[field] = digest(path.read_bytes())
        run = ResearchAuditHarness(root / "run", manifest)
        assert run.inspect_files(paths)
        run.reserve("known", "1.50")
        run.persist_raw("observation", b'{"score":0.5}', "application/json")
        run.derived_metric("observation", "score", 0.5)
        run.execution("observation", "VALID")
        run.settle("known", "0.40")
        run.reserve("unanswered", "2.00")
        run.settle("unanswered", None)
        paths["protocol_sha256"].write_text("changed synthetic protocol", encoding="utf-8")
        mutation_detected = not run.inspect_files(paths)
        result = {"synthetic_control": True, "currency": "illustrative-units",
                  "file_mutation_detected": mutation_detected, **run.replay()}
        assert result["known_cost"] == "0.40"
        assert result["unknown_reserved_cost"] == "2.00"
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    main()
