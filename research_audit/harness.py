"""Append-only execution records. Scientific evidence is interpreted elsewhere.

This package contains no dataset, metric, hypothesis, or stopping-rule logic.
It is deliberately single-writer; a future concurrent runner must add file locks.
"""
from __future__ import annotations

import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path


class AuditError(ValueError):
    pass


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def amount(value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise AuditError("invalid_cost") from exc
    if not result.is_finite() or result < 0:
        raise AuditError("invalid_cost")
    return result


class ResearchAuditHarness:
    """Record raw data before derived observations and preserve uncertain spend."""

    IDENTITY = ("task_id", "protocol_sha256", "metric_sha256", "dataset_sha256",
                "evaluator_sha256", "threshold_sha256")

    def __init__(self, run_dir: str | Path, manifest: dict):
        self.root = Path(run_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir(exist_ok=True)
        if not isinstance(manifest, dict) or not all(manifest.get(k) for k in self.IDENTITY):
            raise AuditError("manifest_missing_immutable_identity")
        self.manifest = dict(manifest)
        self.manifest_hash = digest(canonical(manifest))
        self.manifest_path = self.root / "manifest.json"
        if self.manifest_path.exists():
            if digest(self.manifest_path.read_bytes()) != self.manifest_hash:
                raise AuditError("immutable_manifest_changed")
        else:
            with self.manifest_path.open("xb") as handle:
                handle.write(canonical(manifest))
                handle.flush()
                os.fsync(handle.fileno())
        self.ledger = self.root / "events.jsonl"
        if self.ledger.exists():
            self.replay()

    def _events(self) -> list[dict]:
        if not self.ledger.exists():
            return []
        events = []
        prior = "0" * 64
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            seen = row.pop("event_hash", None)
            if row.get("prev_hash") != prior or digest(canonical(row)) != seen:
                raise AuditError("ledger_chain_invalid")
            prior = seen
            row["event_hash"] = seen
            events.append(row)
        return events

    def _append(self, kind: str, payload: dict) -> dict:
        events = self._events()
        row = {"sequence": len(events), "prev_hash": events[-1]["event_hash"] if events else "0" * 64,
               "manifest_sha256": self.manifest_hash, "kind": kind, "payload": payload}
        row["event_hash"] = digest(canonical(row))
        with self.ledger.open("ab") as handle:
            handle.write(canonical(row) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        return row

    def inspect_identity(self, observed: dict) -> bool:
        changed = sorted(k for k in self.IDENTITY if observed.get(k) != self.manifest[k])
        if changed:
            self._append("protocol_mutation", {"changed_fields": changed})
            return False
        self._append("identity_check", {"clean": True})
        return True

    def inspect_files(self, paths: dict[str, str | Path]) -> bool:
        """Rehash the actual frozen inputs; caller must supply all identity files."""
        fields = set(self.IDENTITY) - {"task_id"}
        if set(paths) != fields:
            raise AuditError("identity_files_incomplete")
        observed = {"task_id": self.manifest["task_id"]}
        for field, filename in paths.items():
            path = Path(filename)
            observed[field] = digest(path.read_bytes()) if path.is_file() else None
        return self.inspect_identity(observed)

    def reserve(self, request_id: str, maximum_cost: object) -> None:
        events = self._events()
        if any(e["kind"] == "cost_reserved" and e["payload"]["request_id"] == request_id for e in events):
            raise AuditError("duplicate_request_reservation")
        self._append("cost_reserved", {"request_id": request_id, "maximum_cost": str(amount(maximum_cost))})

    def settle(self, request_id: str, actual_cost: object | None) -> None:
        events = self._events()
        reservations = [e["payload"] for e in events if e["kind"] == "cost_reserved"
                        and e["payload"]["request_id"] == request_id]
        if len(reservations) != 1:
            raise AuditError("missing_reservation")
        if any(e["kind"] == "cost_settled" and e["payload"]["request_id"] == request_id for e in events):
            raise AuditError("duplicate_settlement")
        observed = None if actual_cost is None else amount(actual_cost)
        if observed is not None and observed > amount(reservations[0]["maximum_cost"]):
            raise AuditError("cost_exceeded_reservation")
        self._append("cost_settled", {"request_id": request_id,
                                      "actual_cost": None if observed is None else str(observed)})

    def persist_raw(self, artifact_id: str, raw: bytes, media_type: str) -> str:
        if not artifact_id or not all(c.isalnum() or c in "-_" for c in artifact_id):
            raise AuditError("unsafe_artifact_id")
        path = self.raw_dir / artifact_id
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        sha = digest(raw)
        self._append("raw_artifact", {"artifact_id": artifact_id, "sha256": sha,
                                      "bytes": len(raw), "media_type": media_type})
        return sha

    def derived_metric(self, artifact_id: str, name: str, value: object,
                       uncertainty: object | None = None) -> None:
        rows = [e["payload"] for e in self._events() if e["kind"] == "raw_artifact"]
        if not any(row["artifact_id"] == artifact_id for row in rows):
            raise AuditError("raw_artifact_required_first")
        self._append("derived_metric", {"artifact_id": artifact_id, "name": name,
                                        "value": value, "uncertainty": uncertainty})

    def execution(self, artifact_id: str | None, status: str, reason: str = "") -> None:
        if status not in {"VALID", "INVALID"}:
            raise AuditError("unknown_execution_status")
        if status == "VALID" and (artifact_id is None or not any(
                e["kind"] == "raw_artifact" and e["payload"]["artifact_id"] == artifact_id
                for e in self._events())):
            raise AuditError("valid_execution_requires_raw_artifact")
        self._append("execution", {"artifact_id": artifact_id, "status": status, "reason": reason})

    def action(self, name: str, detail: dict | None = None) -> None:
        self._append("agent_action", {"name": name, "detail": detail or {}})

    def replay(self) -> dict:
        if digest(self.manifest_path.read_bytes()) != self.manifest_hash:
            raise AuditError("immutable_manifest_changed")
        events = self._events()
        reserved: dict[str, Decimal] = {}
        settled: dict[str, Decimal | None] = {}
        raw_count = 0
        for event in events:
            if event["manifest_sha256"] != self.manifest_hash:
                raise AuditError("ledger_manifest_mismatch")
            p = event["payload"]
            if event["kind"] == "cost_reserved":
                reserved[p["request_id"]] = amount(p["maximum_cost"])
            elif event["kind"] == "cost_settled":
                settled[p["request_id"]] = None if p["actual_cost"] is None else amount(p["actual_cost"])
            elif event["kind"] == "raw_artifact":
                path = self.raw_dir / p["artifact_id"]
                if not path.is_file() or digest(path.read_bytes()) != p["sha256"]:
                    raise AuditError("raw_artifact_corrupt_or_missing")
                raw_count += 1
        unknown = sorted(k for k in reserved if k not in settled or settled[k] is None)
        known = sum((v for v in settled.values() if v is not None), Decimal(0))
        held = sum((reserved[k] for k in unknown), Decimal(0))
        return {"event_count": len(events), "raw_artifact_count": raw_count,
                "known_cost": str(known), "unknown_reserved_cost": str(held),
                "unknown_request_ids": unknown, "total_conservative_cost": str(known + held)}
