"""Exercise failure and restart boundaries of the independently released ledger."""
import json

import pytest

from research_audit import AuditError, ResearchAuditHarness
from research_audit.harness import digest


def inputs(root):
    manifest = {"task_id": "synthetic-test"}
    paths = {}
    for field in ResearchAuditHarness.IDENTITY[1:]:
        path = root / field
        path.write_bytes(field.encode())
        paths[field] = path
        manifest[field] = digest(path.read_bytes())
    return manifest, paths


def test_unknown_reservations_survive_reopening_and_can_settle(tmp_path):
    manifest, _ = inputs(tmp_path)
    run = ResearchAuditHarness(tmp_path / "run", manifest)
    run.reserve("known", "1.00")
    run.settle("known", "0.25")
    run.reserve("unknown", "2.50")
    run.settle("unknown", None)
    run.reserve("unanswered", "1.50")
    restarted = ResearchAuditHarness(tmp_path / "run", manifest)
    summary = restarted.replay()
    assert summary["unknown_request_ids"] == ["unanswered", "unknown"]
    assert summary["total_conservative_cost"] == "4.25"
    restarted.settle("unanswered", "0.10")
    assert restarted.replay()["total_conservative_cost"] == "2.85"
    with pytest.raises(AuditError, match="duplicate_settlement"):
        restarted.settle("unknown", "0.10")


def test_raw_first_replay_detects_actual_file_tampering(tmp_path):
    manifest, _ = inputs(tmp_path)
    run = ResearchAuditHarness(tmp_path / "run", manifest)
    with pytest.raises(AuditError, match="raw_artifact_required_first"):
        run.derived_metric("sample", "score", 1.0)
    run.persist_raw("sample", b'{"score":0.5}', "application/json")
    run.derived_metric("sample", "score", 0.5)
    run.execution("sample", "VALID")
    assert run.replay()["raw_artifact_count"] == 1
    (tmp_path / "run/raw/sample").write_bytes(b"modified")
    with pytest.raises(AuditError, match="raw_artifact_corrupt_or_missing"):
        ResearchAuditHarness(tmp_path / "run", manifest)


def test_protocol_files_and_manifest_mutation_detected(tmp_path):
    manifest, paths = inputs(tmp_path)
    run = ResearchAuditHarness(tmp_path / "run", manifest)
    assert run.inspect_files(paths)
    paths["metric_sha256"].write_bytes(b"changed")
    assert not run.inspect_files(paths)
    changed = dict(manifest, threshold_sha256="0" * 64)
    with pytest.raises(AuditError, match="immutable_manifest_changed"):
        ResearchAuditHarness(tmp_path / "run", changed)
    rows = [json.loads(line) for line in run.ledger.read_text().splitlines()]
    assert rows[-1]["payload"]["changed_fields"] == ["metric_sha256"]


def test_event_mutation_detected_and_invalid_execution_retained(tmp_path):
    manifest, _ = inputs(tmp_path)
    run = ResearchAuditHarness(tmp_path / "run", manifest)
    run.execution(None, "INVALID", "synthetic_failure")
    assert run.replay()["event_count"] == 1
    event = json.loads(run.ledger.read_text())
    event["payload"]["status"] = "VALID"
    run.ledger.write_text(json.dumps(event) + "\n")
    with pytest.raises(AuditError, match="ledger_chain_invalid"):
        run.replay()


def test_duplicate_overspend_and_unsafe_artifact_rejected(tmp_path):
    manifest, _ = inputs(tmp_path)
    run = ResearchAuditHarness(tmp_path / "run", manifest)
    run.reserve("request", "1.00")
    with pytest.raises(AuditError, match="duplicate_request"):
        run.reserve("request", "1.00")
    with pytest.raises(AuditError, match="cost_exceeded_reservation"):
        run.settle("request", "1.01")
    with pytest.raises(AuditError, match="unsafe_artifact_id"):
        run.persist_raw("../outside", b"x", "text/plain")
    with pytest.raises(AuditError, match="invalid_cost"):
        run.reserve("nonfinite", "NaN")
    assert run.replay()["unknown_reserved_cost"] == "1.00"
