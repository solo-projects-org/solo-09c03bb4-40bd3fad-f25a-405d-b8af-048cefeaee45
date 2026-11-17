"""Domain logic: revision rules, interval settlement, rebuild transactions.

Key guarantees implemented here:

* Submissions are versioned by (source_id, revision).  Revisions must be
  strictly increasing; re-posting the exact same payload is idempotent;
  posting different content under an existing (source_id, revision) is
  rejected with 409.
* As soon as a late reading/correction is *accepted*, every current interval
  of every affected device is marked ``stale`` inside the same transaction.
* The rebuild that follows runs in its own transaction per device.  A failure
  (or a process kill, or a restart) leaves the stale rows in place — old
  results never come back as current, and devices not involved keep serving.
* Undeterminable intervals are stored with ``energy_kwh = NULL``; zero is
  only ever written for a genuinely measured zero difference.

Correction-impact tracking (``GET /correction-impact``):

* Every accepted submission is recorded in ``submissions`` with the devices
  it touched, giving a monotonic acceptance order.
* Same-device corrections additionally get a ``correction_events`` row whose
  baseline snapshot freezes the intervals actually servable immediately
  before acceptance (an empty baseline means comparison is impossible).
* The event becomes ``published`` only when a rebuild succeeds with no
  intervening submission for the device; the published snapshot is frozen so
  later rebuilds cannot rewrite the answer.  Failed rebuilds leave it
  ``pending`` (retry may still deliver); an intervening acceptance marks it
  ``superseded`` and later results are never attributed to that correction.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3

from .errors import (
    ConflictError,
    ImpactAmbiguous,
    ImpactNotFound,
    ImpactNotEligible,
    RebuildFailure,
)
from .models import HandoverIn, ReadingIn, to_utc_iso, utc_now_iso


def _hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Correction-impact bookkeeping
# ---------------------------------------------------------------------------

# Fields compared to decide whether an interval changed.  Provenance on both
# ends and the handover reference count as changes even when energy is equal.
_PROVENANCE_FIELDS = (
    "start_source_id", "start_revision", "end_source_id", "end_revision",
    "handover_source_id", "handover_revision",
)
_COMPARED_FIELDS = ("energy_kwh", "undeterminable", "reason", *_PROVENANCE_FIELDS)


def _interval_view(row: sqlite3.Row) -> dict:
    """Serializable view of an intervals row used for frozen snapshots."""
    return {
        "device_id": row["device_id"],
        "start_ts": row["start_ts"],
        "end_ts": row["end_ts"],
        "start_meter_no": row["start_meter_no"],
        "end_meter_no": row["end_meter_no"],
        "energy_kwh": row["energy_kwh"],
        "undeterminable": row["energy_kwh"] is None,
        "reason": row["reason"],
        "start_source_id": row["start_source_id"],
        "start_revision": row["start_revision"],
        "end_source_id": row["end_source_id"],
        "end_revision": row["end_revision"],
        "handover_source_id": row["handover_source_id"],
        "handover_revision": row["handover_revision"],
    }


def _current_interval_snapshot(conn: sqlite3.Connection, device_id: str) -> list[dict]:
    """What the device actually has available for supply right now."""
    rows = conn.execute(
        "SELECT * FROM intervals WHERE device_id = ? AND status = 'current' "
        "ORDER BY start_ts, id",
        (device_id,),
    ).fetchall()
    return [_interval_view(r) for r in rows]


def _supersede_pending_events(
    conn: sqlite3.Connection, device_ids: list[str]
) -> None:
    """Another submission touched these devices: their pending correction
    events can no longer claim the next rebuild's results."""
    if not device_ids:
        return
    placeholders = ",".join("?" for _ in device_ids)
    conn.execute(
        f"UPDATE correction_events SET state = 'superseded', updated_at = ? "
        f"WHERE state = 'pending' AND device_id IN ({placeholders})",
        [utc_now_iso(), *device_ids],
    )


def _record_submission(
    conn: sqlite3.Connection,
    kind: str,
    source_id: str,
    revision: int,
    device_id: str,
    affected: list[str],
) -> int:
    cur = conn.execute(
        "INSERT INTO submissions (kind, source_id, revision, device_id, "
        "affected_devices) VALUES (?,?,?,?,?)",
        (kind, source_id, revision, device_id, json.dumps(affected)),
    )
    return cur.lastrowid


def _has_intervening_submission(
    conn: sqlite3.Connection, device_id: str, submission_id: int
) -> bool:
    """Any later accepted submission (of any kind) that touched this device?"""
    row = conn.execute(
        "SELECT 1 AS hit FROM submissions s, json_each(s.affected_devices) je "
        "WHERE je.value = ? AND s.id > ? LIMIT 1",
        (device_id, submission_id),
    ).fetchone()
    return row is not None


def _finalize_pending_events(
    conn: sqlite3.Connection, device_id: str, run_id: int, new_rows: list[dict]
) -> None:
    """Resolve pending correction events of a device after a successful rebuild.

    An event is *published* only when no other submission for the device was
    accepted in between; otherwise it stays superseded and the rebuild's
    output is never attributed to that correction.
    """
    pending = conn.execute(
        "SELECT * FROM correction_events WHERE device_id = ? AND state = 'pending'",
        (device_id,),
    ).fetchall()
    for event in pending:
        if _has_intervening_submission(conn, device_id, event["submission_id"]):
            conn.execute(
                "UPDATE correction_events SET state = 'superseded', updated_at = ? "
                "WHERE id = ?",
                (utc_now_iso(), event["id"]),
            )
        else:
            conn.execute(
                "UPDATE correction_events SET state = 'published', "
                "published_run_id = ?, published_snapshot = ?, failure_reason = NULL, "
                "updated_at = ? WHERE id = ?",
                (run_id, json.dumps(new_rows), utc_now_iso(), event["id"]),
            )


# ---------------------------------------------------------------------------
# Revisioned submission handling
# ---------------------------------------------------------------------------

def _check_revision(
    conn: sqlite3.Connection,
    table: str,
    source_id: str,
    revision: int,
    content_hash: str,
) -> bool:
    """Validate the revision chain.

    Revisions must be consecutive: the first submission is revision 1 and
    every later submission must use exactly ``latest + 1``.  Re-posting the
    exact same payload under an existing revision is idempotent.  A content
    mismatch on an existing (source_id, revision) or any rollback/skipping is
    rejected with ConflictError (HTTP 409).
    """
    row = conn.execute(
        f"SELECT revision, content_hash FROM {table} WHERE source_id = ? "
        f"ORDER BY revision DESC LIMIT 1",
        (source_id,),
    ).fetchone()
    if row is not None:
        latest = row["revision"]
        if revision < latest:
            raise ConflictError(
                f"revision rollback rejected: latest revision of source "
                f"'{source_id}' is {latest}, got {revision}"
            )
        if revision == latest:
            existing = conn.execute(
                f"SELECT content_hash FROM {table} WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            ).fetchone()
            if existing["content_hash"] != content_hash:
                raise ConflictError(
                    f"content conflict: source '{source_id}' revision {revision} "
                    f"already exists with different content"
                )
            return True  # identical re-submission
        if revision != latest + 1:
            raise ConflictError(
                f"non-consecutive revision rejected: latest revision of source "
                f"'{source_id}' is {latest}, expected {latest + 1}, got {revision}"
            )
    elif revision != 1:
        raise ConflictError(
            f"first revision of source '{source_id}' must be 1, got {revision}"
        )
    return False


def _stale_current_intervals(conn: sqlite3.Connection, device_ids: list[str]) -> int:
    """Immediately revoke current-supply eligibility on affected devices."""
    if not device_ids:
        return 0
    placeholders = ",".join("?" for _ in device_ids)
    cur = conn.execute(
        f"UPDATE intervals SET status = 'stale', "
        f"superseded_at = ? WHERE status = 'current' AND device_id IN ({placeholders})",
        [utc_now_iso(), *device_ids],
    )
    return cur.rowcount


def submit_reading(conn: sqlite3.Connection, payload: ReadingIn) -> dict:
    ts_iso = to_utc_iso(payload.ts)
    content = {
        "device_id": payload.device_id,
        "ts": ts_iso,
        "meter_no": payload.meter_no,
        "cumulative_kwh": payload.cumulative_kwh,
    }
    digest = _hash(content)
    affected: list[str] = []

    conn.execute("BEGIN IMMEDIATE")
    try:
        idempotent = _check_revision(conn, "readings", payload.source_id, payload.revision, digest)
        if not idempotent:
            # The new reading belongs to its own device.  A correction may also
            # move the reading to a different device, so stale whatever device
            # the previous revision pointed at.
            prev = conn.execute(
                "SELECT device_id FROM readings WHERE source_id = ? AND revision < ? "
                "ORDER BY revision DESC LIMIT 1",
                (payload.source_id, payload.revision),
            ).fetchone()
            prev_device = prev["device_id"] if prev is not None else None
            devices = {payload.device_id}
            if prev_device is not None:
                devices.add(prev_device)
            affected = sorted(devices)

            # A new acceptance first invalidates any pending correction of the
            # same device: later rebuild results must never be attributed to it.
            _supersede_pending_events(conn, affected)
            # Freeze what was actually servable BEFORE this submission revokes
            # current eligibility (empty list => no baseline, comparison off).
            baseline = _current_interval_snapshot(conn, payload.device_id)
            submission_id = _record_submission(
                conn, "reading", payload.source_id, payload.revision,
                payload.device_id, affected,
            )
            conn.execute(
                "INSERT INTO readings (source_id, revision, device_id, ts, meter_no, "
                "cumulative, content_hash) VALUES (?,?,?,?,?,?,?)",
                (
                    payload.source_id, payload.revision, payload.device_id, ts_iso,
                    payload.meter_no, payload.cumulative_kwh, digest,
                ),
            )
            _stale_current_intervals(conn, affected)
            # Impact events exist only for revisions whose previous revision
            # still belongs to the same device; cross-device moves are not
            # handled by the correction-impact API.
            if prev_device == payload.device_id:
                conn.execute(
                    "INSERT INTO correction_events (kind, source_id, revision, "
                    "device_id, submission_id, baseline_available, baseline_snapshot, "
                    "state) VALUES ('reading',?,?,?,?,?,?,'pending')",
                    (
                        payload.source_id, payload.revision, payload.device_id,
                        submission_id, 1 if baseline else 0,
                        json.dumps(baseline) if baseline else None,
                    ),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "reading",
        "source_id": payload.source_id,
        "revision": payload.revision,
        "idempotent": idempotent,
        "affected_devices": affected,
    }


def submit_handover(conn: sqlite3.Connection, payload: HandoverIn) -> dict:
    ts_iso = to_utc_iso(payload.ts)
    content = {
        "device_id": payload.device_id,
        "ts": ts_iso,
        "old_meter_no": payload.old_meter_no,
        "old_final_kwh": payload.old_final_kwh,
        "new_meter_no": payload.new_meter_no,
        "new_initial_kwh": payload.new_initial_kwh,
    }
    digest = _hash(content)
    affected: list[str] = []

    conn.execute("BEGIN IMMEDIATE")
    try:
        idempotent = _check_revision(conn, "handovers", payload.source_id, payload.revision, digest)
        if not idempotent:
            prev = conn.execute(
                "SELECT device_id, old_meter_no, new_meter_no FROM handovers "
                "WHERE source_id = ? AND revision < ? ORDER BY revision DESC LIMIT 1",
                (payload.source_id, payload.revision),
            ).fetchone()
            prev_device = prev["device_id"] if prev is not None else None
            devices = {payload.device_id}
            if prev_device is not None:
                devices.add(prev_device)
            affected = sorted(devices)

            _supersede_pending_events(conn, affected)
            baseline = _current_interval_snapshot(conn, payload.device_id)
            submission_id = _record_submission(
                conn, "handover", payload.source_id, payload.revision,
                payload.device_id, affected,
            )
            conn.execute(
                "INSERT INTO handovers (source_id, revision, device_id, ts, old_meter_no, "
                "old_final, new_meter_no, new_initial, content_hash) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    payload.source_id, payload.revision, payload.device_id, ts_iso,
                    payload.old_meter_no, payload.old_final_kwh,
                    payload.new_meter_no, payload.new_initial_kwh, digest,
                ),
            )
            _stale_current_intervals(conn, affected)
            if prev_device == payload.device_id:
                conn.execute(
                    "INSERT INTO correction_events (kind, source_id, revision, "
                    "device_id, submission_id, baseline_available, baseline_snapshot, "
                    "state) VALUES ('handover',?,?,?,?,?,?,'pending')",
                    (
                        payload.source_id, payload.revision, payload.device_id,
                        submission_id, 1 if baseline else 0,
                        json.dumps(baseline) if baseline else None,
                    ),
                )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return {
        "kind": "handover",
        "source_id": payload.source_id,
        "revision": payload.revision,
        "idempotent": idempotent,
        "affected_devices": affected,
    }


# ---------------------------------------------------------------------------
# Interval computation
# ---------------------------------------------------------------------------

def _latest_readings(conn: sqlite3.Connection, device_id: str) -> list[sqlite3.Row]:
    """Authoritative readings: highest revision of every source of a device."""
    return conn.execute(
        "WITH latest AS ("
        "  SELECT source_id, MAX(revision) AS max_rev FROM readings "
        "  WHERE device_id = ? GROUP BY source_id"
        ") SELECT r.* FROM readings r JOIN latest l "
        "ON r.source_id = l.source_id AND r.revision = l.max_rev "
        "ORDER BY r.ts ASC, r.id ASC",
        (device_id,),
    ).fetchall()


def _latest_handovers(conn: sqlite3.Connection, device_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "WITH latest AS ("
        "  SELECT source_id, MAX(revision) AS max_rev FROM handovers "
        "  WHERE device_id = ? GROUP BY source_id"
        ") SELECT h.* FROM handovers h JOIN latest l "
        "ON h.source_id = l.source_id AND h.revision = l.max_rev",
        (device_id,),
    ).fetchall()


def _settle_interval(
    conn: sqlite3.Connection,
    device_id: str,
    a: sqlite3.Row,
    b: sqlite3.Row,
) -> dict:
    """Compute the consumption between two adjacent readings.

    Same meter  -> simple difference; a backward reading is undeterminable
                   (never clamped to zero).
    Cross meter -> only settle when exactly one handover of this device falls
                   strictly inside the interval and its meter numbers match
                   both ends; otherwise undeterminable ('handover missing').
    """
    common = {
        "device_id": device_id,
        "start_ts": a["ts"], "end_ts": b["ts"],
        "start_meter_no": a["meter_no"], "end_meter_no": b["meter_no"],
        "start_source_id": a["source_id"], "start_revision": a["revision"],
        "end_source_id": b["source_id"], "end_revision": b["revision"],
        "handover_source_id": None, "handover_revision": None,
    }

    if a["meter_no"] == b["meter_no"]:
        diff = b["cumulative"] - a["cumulative"]
        if diff < 0:
            return {**common, "energy_kwh": None,
                    "reason": "negative_difference: meter reading went backward"}
        return {**common, "energy_kwh": round(diff, 9), "reason": None}

    # Cross-meter interval: look for the unique matching handover inside it.
    matches = conn.execute(
        "SELECT * FROM handovers WHERE device_id = ? AND ts > ? AND ts < ? "
        "AND old_meter_no = ? AND new_meter_no = ? "
        "AND source_id IN (SELECT source_id FROM handovers WHERE device_id = ?) "
        "ORDER BY id",
        (device_id, a["ts"], b["ts"], a["meter_no"], b["meter_no"], device_id),
    ).fetchall()
    # Keep only the latest revision per handover source.
    latest: dict[str, sqlite3.Row] = {}
    for h in matches:
        cur = latest.get(h["source_id"])
        if cur is None or h["revision"] > cur["revision"]:
            latest[h["source_id"]] = h
    unique = list(latest.values())

    if len(unique) == 0:
        return {**common, "energy_kwh": None,
                "reason": "handover_missing: no matching handover inside cross-meter interval"}
    if len(unique) > 1:
        return {**common, "energy_kwh": None,
                "reason": "ambiguous_handover: more than one matching handover inside interval"}

    h = unique[0]
    seg1 = h["old_final"] - a["cumulative"]
    seg2 = b["cumulative"] - h["new_initial"]
    if seg1 < 0 or seg2 < 0:
        return {**common, "energy_kwh": None,
                "handover_source_id": h["source_id"], "handover_revision": h["revision"],
                "reason": "negative_segment: reading backward relative to handover values"}
    return {
        **common,
        "energy_kwh": round(seg1 + seg2, 9),
        "reason": None,
        "handover_source_id": h["source_id"],
        "handover_revision": h["revision"],
    }


def devices_with_stale_intervals(conn: sqlite3.Connection) -> list[str]:
    return [
        r["device_id"]
        for r in conn.execute(
            "SELECT DISTINCT device_id FROM intervals WHERE status = 'stale' ORDER BY 1"
        )
    ]


def rebuild_device(
    conn: sqlite3.Connection,
    device_id: str,
    *,
    fail_device: str | None = None,
    fail_header_devices: set[str] | None = None,
    auto_trigger: bool = False,
) -> int:
    """Atomically rebuild all intervals of one device.

    Previously stale rows are kept for audit; only ``current`` rows are
    replaced, and new rows are inserted as ``current`` in one transaction.
    An injected failure raises *before* commit to simulate a crash: rollback
    leaves the run at status ``running`` and every previously stale interval
    untouched.  On service restart init_db flips dangling ``running`` runs to
    ``interrupted`` without reviving any interval.

    ``auto_trigger`` marks an automatic rebuild right after an accepted
    submission: pending correction events of the device are linked to this
    run.  On success, pending events without an intervening submission are
    published with the frozen new snapshot; failures only record the reason
    while the event stays pending, so a later retry can still deliver.
    """
    fail_header_devices = fail_header_devices or set()
    run = conn.execute(
        "INSERT INTO rebuild_runs (device_id, status) VALUES (?, 'running')",
        (device_id,),
    )
    run_id = run.lastrowid
    if auto_trigger:
        # The auto-rebuild belongs to every still-pending event of this device.
        conn.execute(
            "UPDATE correction_events SET trigger_run_id = ?, updated_at = ? "
            "WHERE device_id = ? AND state = 'pending' AND trigger_run_id IS NULL",
            (run_id, utc_now_iso(), device_id),
        )
    conn.commit()

    conn.execute("BEGIN IMMEDIATE")
    try:
        if fail_device == device_id or device_id in fail_header_devices:
            raise RebuildFailure(
                device_id,
                "injected failure before commit (test hook); stale intervals retained",
            )

        readings = _latest_readings(conn, device_id)
        # Touch the handover table so its latest revisions are considered even
        # when no cross-meter interval exists yet.
        _latest_handovers(conn, device_id)

        rows = [
            _settle_interval(conn, device_id, readings[i], readings[i + 1])
            for i in range(len(readings) - 1)
        ]

        # Stale rows stay as an audit trail; only current rows are replaced.
        conn.execute(
            "DELETE FROM intervals WHERE device_id = ? AND status = 'current'",
            (device_id,),
        )
        new_views: list[dict] = []
        for r in rows:
            conn.execute(
                "INSERT INTO intervals (device_id, start_ts, end_ts, start_meter_no, "
                "end_meter_no, energy_kwh, status, reason, start_source_id, "
                "start_revision, end_source_id, end_revision, handover_source_id, "
                "handover_revision) VALUES (?,?,?,?,?,?,'current',?,?,?,?,?,?,?)",
                (
                    r["device_id"], r["start_ts"], r["end_ts"],
                    r["start_meter_no"], r["end_meter_no"], r["energy_kwh"], r["reason"],
                    r["start_source_id"], r["start_revision"],
                    r["end_source_id"], r["end_revision"],
                    r["handover_source_id"], r["handover_revision"],
                ),
            )
            new_views.append({**r, "undeterminable": r["energy_kwh"] is None})
        # Attribute the new intervals to the pending correction(s) of this
        # device, unless another submission was accepted in the meantime.
        _finalize_pending_events(conn, device_id, run_id, new_views)
        conn.execute(
            "UPDATE rebuild_runs SET status = 'success', "
            "finished_at = ? WHERE id = ?",
            (utc_now_iso(), run_id),
        )
        conn.commit()
        return len(rows)
    except RebuildFailure:
        # Simulated crash: rollback and deliberately leave the run 'running'
        # so the restart path marks it 'interrupted'.  Stale rows survive.
        conn.rollback()
        # The correction event stays pending: only a successful retry may
        # deliver the new intervals.
        conn.execute(
            "UPDATE correction_events SET failure_reason = ?, updated_at = ? "
            "WHERE device_id = ? AND state = 'pending'",
            (
                "injected failure before commit (test hook); stale intervals retained",
                utc_now_iso(), device_id,
            ),
        )
        conn.commit()
        raise
    except Exception as exc:
        conn.rollback()
        conn.execute(
            "UPDATE rebuild_runs SET status = 'failed', detail = ?, "
            "finished_at = ? WHERE id = ?",
            (str(exc), utc_now_iso(), run_id),
        )
        conn.execute(
            "UPDATE correction_events SET failure_reason = ?, updated_at = ? "
            "WHERE device_id = ? AND state = 'pending'",
            (str(exc), utc_now_iso(), device_id),
        )
        conn.commit()
        raise


def rebuild(
    conn: sqlite3.Connection,
    device_id: str | None = None,
    *,
    devices: list[str] | None = None,
    all_devices: bool = False,
    only_stale: bool = False,
    fail_device: str | None = None,
    fail_header_devices: set[str] | None = None,
) -> dict:
    """Rebuild one device, or every device that needs it.

    ``devices`` limits the run to an explicit set (used by automatic rebuilds
    after submissions, including a device's very first reading when no stale
    rows exist yet).  By default all devices that have readings are rebuilt;
    ``only_stale=True`` limits the run to devices currently holding stale
    intervals.  A failure on one device is recorded and does not abort the
    others; stale intervals of a failed device remain queryable as stale and
    never regain current eligibility.
    """
    if device_id is not None:
        targets = [device_id]
    elif devices is not None:
        targets = list(dict.fromkeys(devices))  # de-duplicate, keep order
    elif all_devices or not only_stale:
        targets = [
            r["device_id"]
            for r in conn.execute(
                "SELECT DISTINCT device_id FROM readings ORDER BY 1"
            )
        ]
    else:
        targets = devices_with_stale_intervals(conn)

    rebuilt: list[str] = []
    failed: dict[str, str] = {}
    # Only the rebuild driven by a just-accepted submission links its run to
    # the pending correction events.  Manual retries attribute via the
    # intervening-submission check inside rebuild_device.
    auto_trigger = devices is not None and device_id is None and not all_devices
    for dev in targets:
        try:
            n = rebuild_device(
                conn, dev,
                fail_device=fail_device,
                fail_header_devices=fail_header_devices,
                auto_trigger=auto_trigger,
            )
            rebuilt.append(f"{dev} ({n} intervals)")
        except RebuildFailure as exc:
            failed[dev] = exc.reason

    success = not failed
    run_row = conn.execute(
        "INSERT INTO rebuild_runs (device_id, status, detail, finished_at) "
        "VALUES (?,?,?,?)",
        (
            device_id,
            "success" if success else "failed",
            json.dumps({"rebuilt": rebuilt, "failed": failed}),
            utc_now_iso(),
        ),
    )
    conn.commit()
    return {
        "run_id": run_row.lastrowid,
        "success": success,
        "rebuilt_devices": rebuilt,
        "failed_devices": failed,
    }


def list_intervals(
    conn: sqlite3.Connection,
    device_id: str | None,
    status: str,
) -> list[sqlite3.Row]:
    clauses: list[str] = []
    params: list[str] = []
    if status != "all":
        clauses.append("status = ?")
        params.append(status)
    if device_id is not None:
        clauses.append("device_id = ?")
        params.append(device_id)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    return conn.execute(
        f"SELECT * FROM intervals{where} ORDER BY device_id, start_ts, id",
        params,
    ).fetchall()


# ---------------------------------------------------------------------------
# Correction-impact query
# ---------------------------------------------------------------------------

def _interval_key(iv: dict) -> tuple[str, str]:
    return iv["start_ts"], iv["end_ts"]


def _interval_changed(before: dict, after: dict) -> bool:
    """A change means energy, the undeterminable verdict, its reason, or any
    provenance reference differs.  Unchanged intervals are not reported."""
    return any(before[f] != after[f] for f in _COMPARED_FIELDS)


def _diff_intervals(baseline: list[dict], published: list[dict]) -> list[dict]:
    """Compare two frozen interval sets by their (start_ts, end_ts) bounds.

    Identical bounds -> one ``modified`` entry only when something actually
    changed.  Moved boundaries are reported as separate ``removed`` /
    ``added`` entries carrying both sources; no difference amount is computed
    across non-matching intervals.
    """
    old_by_key: dict[tuple[str, str], dict] = {_interval_key(i): i for i in baseline}
    new_by_key: dict[tuple[str, str], dict] = {_interval_key(i): i for i in published}
    changes: list[dict] = []

    for key in sorted(set(old_by_key) & set(new_by_key)):
        before, after = old_by_key[key], new_by_key[key]
        if _interval_changed(before, after):
            changes.append({"change_type": "modified", "before": before, "after": after})
    for key in sorted(set(old_by_key) - set(new_by_key)):
        changes.append({"change_type": "removed", "before": old_by_key[key], "after": None})
    for key in sorted(set(new_by_key) - set(old_by_key)):
        changes.append({"change_type": "added", "before": None, "after": new_by_key[key]})
    return changes


def get_correction_impact(
    conn: sqlite3.Connection,
    source_id: str,
    revision: int,
    kind: str | None = None,
) -> dict:
    """Return the frozen before/after comparison for one accepted correction.

    Eligibility: only revisions whose previous revision still belongs to the
    same device have an impact event.  The answer is always built from frozen
    snapshots, so subsequent submissions or rebuilds cannot alter it.
    """
    if kind is None:
        kinds = [
            r["kind"]
            for r in conn.execute(
                "SELECT DISTINCT kind FROM submissions WHERE source_id = ? AND revision = ?",
                (source_id, revision),
            ).fetchall()
        ]
        if not kinds:
            raise ImpactNotFound(
                f"no accepted submission found for source_id='{source_id}' "
                f"revision={revision}"
            )
        if len(kinds) > 1:
            raise ImpactAmbiguous(
                f"source_id='{source_id}' revision={revision} exists as both a "
                f"reading and a handover; specify kind=reading or kind=handover"
            )
        kind = kinds[0]

    submission = conn.execute(
        "SELECT * FROM submissions WHERE kind = ? AND source_id = ? AND revision = ?",
        (kind, source_id, revision),
    ).fetchone()
    if submission is None:
        raise ImpactNotFound(
            f"no accepted {kind} found for source_id='{source_id}' revision={revision}"
        )

    event = conn.execute(
        "SELECT * FROM correction_events WHERE kind = ? AND source_id = ? AND revision = ?",
        (kind, source_id, revision),
    ).fetchone()
    if event is None:
        # The submission exists but is not an eligible same-device correction:
        # either it is the first revision or it moved the source to another
        # device.  Distinguish the two for a precise message.
        table = "readings" if kind == "reading" else "handovers"
        prev = conn.execute(
            f"SELECT device_id FROM {table} WHERE source_id = ? AND revision < ? "
            f"ORDER BY revision DESC LIMIT 1",
            (source_id, revision),
        ).fetchone()
        if prev is None:
            why = "revision 1 is the original submission, not a correction"
        else:
            why = (
                f"correction moves the source across devices "
                f"('{prev['device_id']}' -> '{submission['device_id']}'); impact "
                f"analysis only supports corrections that stay on the same device"
            )
        raise ImpactNotEligible(why)

    baseline: list[dict] = (
        json.loads(event["baseline_snapshot"]) if event["baseline_snapshot"] else []
    )
    published: list[dict] = (
        json.loads(event["published_snapshot"]) if event["published_snapshot"] else []
    )
    state = event["state"]
    baseline_available = bool(event["baseline_available"])
    changes: list[dict] = []
    notes: list[str] = []

    if not baseline_available:
        notes.append(
            "baseline unavailable: the device had no current-supply intervals "
            "before acceptance (old intervals were already pending rebuild), "
            "so no before/after comparison is possible"
        )

    if state == "published":
        if baseline_available:
            changes = _diff_intervals(baseline, published)
        else:
            notes.append("only the new intervals are delivered")
    elif state == "pending":
        notes.append(
            "rebuild pending: this correction's rebuild has not delivered new "
            "intervals yet; only the pre-acceptance side is available. A successful "
            "retry is required before the new intervals can be delivered"
        )
        if event["failure_reason"]:
            notes.append(f"last rebuild failure: {event['failure_reason']}")
    else:  # superseded
        notes.append(
            "superseded: another submission for this device was accepted before the "
            "rebuild delivered; the subsequent rebuild results are not attributed to "
            "this correction"
        )
        if event["failure_reason"]:
            notes.append(f"last rebuild failure: {event['failure_reason']}")

    return {
        "source_id": source_id,
        "revision": revision,
        "kind": kind,
        "device_id": event["device_id"],
        "state": state,
        "baseline_available": baseline_available,
        "comparison_possible": state == "published" and baseline_available,
        "changes": changes,
        "baseline_intervals": baseline,
        "published_intervals": published,
        "trigger_run_id": event["trigger_run_id"],
        "published_run_id": event["published_run_id"],
        "failure_reason": event["failure_reason"],
        "note": " | ".join(notes) if notes else None,
    }
