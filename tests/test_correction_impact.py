"""End-to-end tests for the correction-impact API.

Covers the contract from the analyst's perspective:

* only same-device corrections are served; revision 1 and cross-device moves
  are rejected with 422, unknown submissions with 404, ambiguous source ids
  (same id used by a reading and a handover) with 409;
* the answer is built from frozen pre-acceptance / post-rebuild snapshots and
  lists only intervals whose bounds, energy, undeterminable verdict or
  provenance changed, with both sides attached;
* moved interval boundaries are reported as separate removed/added rows and no
  difference amount is ever invented;
* a device with no current supply at acceptance yields no baseline and thus no
  comparison;
* a failed rebuild leaves the event pending (no new intervals delivered) until
  a retry succeeds;
* another submission accepted in the meantime supersedes the event, so later
  rebuild results are never attributed to the original correction.
"""
from __future__ import annotations

import os
import sys
import tempfile

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402


@pytest.fixture()
def ctx():
    tmp = tempfile.TemporaryDirectory()
    db_path = os.path.join(tmp.name, "test.db")
    settings = Settings(db_path=db_path, fail_device=None)
    app = create_app(settings)
    with TestClient(app) as client:
        yield client, db_path, tmp
    tmp.cleanup()


def reading(source, rev, device, ts, meter, kwh):
    return {
        "source_id": source, "revision": rev, "device_id": device,
        "ts": ts, "meter_no": meter, "cumulative_kwh": kwh,
    }


def handover(source, rev, device, ts, old_m, old_v, new_m, new_v):
    return {
        "source_id": source, "revision": rev, "device_id": device, "ts": ts,
        "old_meter_no": old_m, "old_final_kwh": old_v,
        "new_meter_no": new_m, "new_initial_kwh": new_v,
    }


def impact(client, source, rev, **params):
    params.update({"source_id": source, "revision": rev})
    return client.get("/correction-impact", params=params)


def _seed_two_readings(client, device="D1", second=130.0):
    client.post("/readings", json=reading("S1", 1, device, "2026-01-01T00:00:00Z", "M1", 100.0))
    client.post("/readings", json=reading("S2", 1, device, "2026-01-01T02:00:00Z", "M1", second))


# ---------------------------------------------------------------------------
# 1. Eligibility and error semantics
# ---------------------------------------------------------------------------

def test_unknown_submission_is_404(ctx):
    client, _, _ = ctx
    r = impact(client, "NOPE", 1)
    assert r.status_code == 404


def test_first_revision_is_not_a_correction(ctx):
    client, _, _ = ctx
    client.post("/readings", json=reading("S1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    r = impact(client, "S1", 1)
    assert r.status_code == 422
    assert "not a correction" in r.json()["detail"]


def test_cross_device_correction_rejected_but_supply_works(ctx):
    client, _, _ = ctx
    client.post("/readings", json=reading("S1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    # Revision 2 moves the source from D1 to D2.
    client.post("/readings", json=reading("S1", 2, "D2", "2026-01-01T00:00:00Z", "M1", 100.0))
    r = impact(client, "S1", 2)
    assert r.status_code == 422 and "same device" in r.json()["detail"]
    # Existing supply and revision flows are untouched.
    assert client.get("/health").json()["status"] == "ok"


def test_ambiguous_source_id_requires_kind(ctx):
    client, _, _ = ctx
    client.post("/readings", json=reading("X", 1, "D1", "2026-01-01T00:00:00Z", "M1", 10.0))
    client.post("/readings", json=reading("X", 2, "D1", "2026-01-01T00:00:00Z", "M1", 11.0))
    client.post("/readings", json=reading("Y", 1, "D1", "2026-01-01T02:00:00Z", "M1", 30.0))
    client.post("/handovers", json=handover(
        "X", 1, "D1", "2026-01-01T01:00:00Z", "M1", 20.0, "M2", 1.0))
    client.post("/handovers", json=handover(
        "X", 2, "D1", "2026-01-01T01:00:00Z", "M1", 22.0, "M2", 1.0))
    r = impact(client, "X", 2)
    assert r.status_code == 409 and "kind" in r.json()["detail"]
    assert impact(client, "X", 2, kind="reading").json()["kind"] == "reading"
    assert impact(client, "X", 2, kind="handover").json()["kind"] == "handover"


# ---------------------------------------------------------------------------
# 2. Happy-path diffs and provenance on both sides
# ---------------------------------------------------------------------------

def test_reading_correction_diff_energy_and_provenance(ctx):
    client, _, _ = ctx
    _seed_two_readings(client)
    client.post("/readings", json=reading("S2", 2, "D1", "2026-01-01T02:00:00Z", "M1", 145.0))

    body = impact(client, "S2", 2).json()
    assert body["state"] == "published"
    assert body["baseline_available"] is True and body["comparison_possible"] is True
    assert len(body["changes"]) == 1
    change = body["changes"][0]
    assert change["change_type"] == "modified"
    assert change["before"]["energy_kwh"] == pytest.approx(30.0)
    assert change["after"]["energy_kwh"] == pytest.approx(45.0)
    # Provenance on both sides: the revised end reading carries revision 2.
    assert change["before"]["end_revision"] == 1
    assert change["after"]["end_revision"] == 2
    assert change["before"]["start_source_id"] == "S1"
    # Full snapshots are attached on both sides.
    assert len(body["baseline_intervals"]) == 1
    assert len(body["published_intervals"]) == 1
    assert body["trigger_run_id"] == body["published_run_id"]


def test_unchanged_intervals_are_not_listed(ctx):
    client, _, _ = ctx
    client.post("/readings", json=reading("S1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    client.post("/readings", json=reading("S2", 1, "D1", "2026-01-01T01:00:00Z", "M1", 110.0))
    client.post("/readings", json=reading("S3", 1, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))
    # Correct the first reading only; the 01:00->02:00 interval keeps identical
    # bounds, energy and provenance and must not appear in the diff.
    client.post("/readings", json=reading("S1", 2, "D1", "2026-01-01T00:00:00Z", "M1", 95.0))

    body = impact(client, "S1", 2).json()
    assert body["state"] == "published"
    changed_keys = {
        (c["before"] or c["after"])["start_ts"]
        for c in body["changes"]
    }
    assert changed_keys == {"2026-01-01T00:00:00.000Z"}
    # Both snapshots still contain every interval.
    assert len(body["baseline_intervals"]) == 2
    assert len(body["published_intervals"]) == 2


def test_boundary_change_reports_old_and_new_separately_without_delta(ctx):
    client, _, _ = ctx
    client.post("/readings", json=reading("S1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 0.0))
    client.post("/readings", json=reading("S2", 1, "D1", "2026-01-01T04:00:00Z", "M1", 40.0))
    # Move the first reading one hour later: the old 00:00-04:00 interval
    # disappears and a new 01:00-04:00 interval appears.
    client.post("/readings", json=reading("S1", 2, "D1", "2026-01-01T01:00:00Z", "M1", 0.0))

    body = impact(client, "S1", 2).json()
    types = sorted(c["change_type"] for c in body["changes"])
    assert types == ["added", "removed"]
    removed = next(c for c in body["changes"] if c["change_type"] == "removed")
    added = next(c for c in body["changes"] if c["change_type"] == "added")
    assert removed["before"]["end_ts"] == "2026-01-01T04:00:00.000Z"
    assert added["after"]["start_ts"] == "2026-01-01T01:00:00.000Z"
    # No invented difference amount anywhere in the payload.
    assert "delta" not in str(body)


def test_undeterminable_verdict_change_is_reported(ctx):
    client, _, _ = ctx
    # A backward reading makes the interval undeterminable (energy null).
    client.post("/readings", json=reading("S1", 1, "D1", "2026-01-01T00:00:00Z", "M1", 100.0))
    client.post("/readings", json=reading("S2", 1, "D1", "2026-01-01T02:00:00Z", "M1", 90.0))
    assert client.get("/intervals").json()[0]["undeterminable"] is True
    # Correction restores a valid value.
    client.post("/readings", json=reading("S2", 2, "D1", "2026-01-01T02:00:00Z", "M1", 130.0))

    body = impact(client, "S2", 2).json()
    change = body["changes"][0]
    assert change["before"]["undeterminable"] is True and change["before"]["energy_kwh"] is None
    assert change["after"]["undeterminable"] is False
    assert change["after"]["energy_kwh"] == pytest.approx(30.0)


def test_handover_correction_impact(ctx):
    client, _, _ = ctx
    client.post("/readings", json=reading("S1", 1, "D1", "2026-01-01T00:00:00Z", "M-old", 100.0))
    client.post("/readings", json=reading("S2", 1, "D1", "2026-01-01T02:00:00Z", "M-new", 40.0))
    client.post("/handovers", json=handover(
        "H1", 1, "D1", "2026-01-01T01:00:00Z", "M-old", 150.0, "M-new", 10.0))
    assert client.get("/intervals").json()[0]["energy_kwh"] == pytest.approx(80.0)

    client.post("/handovers", json=handover(
        "H1", 2, "D1", "2026-01-01T01:00:00Z", "M-old", 160.0, "M-new", 10.0))
    body = impact(client, "H1", 2, kind="handover").json()
    assert body["kind"] == "handover" and body["state"] == "published"
    change = body["changes"][0]
    assert change["before"]["energy_kwh"] == pytest.approx(80.0)
    assert change["after"]["energy_kwh"] == pytest.approx(90.0)
    assert change["before"]["handover_revision"] == 1
    assert change["after"]["handover_revision"] == 2


# ---------------------------------------------------------------------------
# 3. Rebuild failure / retry and baseline availability
# ---------------------------------------------------------------------------

def test_failed_rebuild_pending_then_retry_publishes(ctx):
    client, _, _ = ctx
    _seed_two_readings(client)
    r = client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("S2", 2, "D1", "2026-01-01T02:00:00Z", "M1", 145.0),
    )
    assert r.json()["rebuild"]["success"] is False

    body = impact(client, "S2", 2).json()
    assert body["state"] == "pending"
    assert body["comparison_possible"] is False
    assert body["published_intervals"] == []
    assert body["failure_reason"]  # failure is reported, new intervals are not
    assert body["baseline_intervals"][0]["energy_kwh"] == pytest.approx(30.0)

    # A successful retry delivers the new intervals.
    assert client.post("/rebuild", params={"device_id": "D1"}).json()["success"] is True
    body = impact(client, "S2", 2).json()
    assert body["state"] == "published" and body["comparison_possible"] is True
    assert body["published_run_id"] != body["trigger_run_id"]
    assert body["changes"][0]["after"]["energy_kwh"] == pytest.approx(45.0)


def test_baseline_unavailable_when_device_already_pending_rebuild(ctx):
    client, _, _ = ctx
    _seed_two_readings(client)
    # First late submission fails: D1 loses current supply and stays stale.
    client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("S9", 1, "D1", "2026-01-01T01:00:00Z", "M1", 112.0),
    )
    assert client.get("/intervals", params={"device_id": "D1"}).json() == []

    # A correction accepted while the device has no current supply.
    client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("S9", 2, "D1", "2026-01-01T01:00:00Z", "M1", 115.0),
    )
    body = impact(client, "S9", 2).json()
    assert body["state"] == "pending"
    assert body["baseline_available"] is False
    assert body["baseline_intervals"] == [] and body["changes"] == []
    assert "baseline unavailable" in body["note"]

    # Even after a successful retry there is nothing to compare against.
    client.post("/rebuild", params={"device_id": "D1"})
    body = impact(client, "S9", 2).json()
    assert body["state"] == "published"
    assert body["comparison_possible"] is False and body["changes"] == []
    assert len(body["published_intervals"]) == 2  # new intervals are delivered


# ---------------------------------------------------------------------------
# 4. Intervening submissions must not be attributed to the correction
# ---------------------------------------------------------------------------

def test_intervening_submission_supersedes_correction(ctx):
    client, _, _ = ctx
    _seed_two_readings(client)
    # Correction whose auto-rebuild fails first.
    client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("S2", 2, "D1", "2026-01-01T02:00:00Z", "M1", 145.0),
    )
    # While pending, another submission for the same device is accepted and
    # rebuilt successfully.
    client.post("/readings", json=reading("S8", 1, "D1", "2026-01-01T03:00:00Z", "M1", 160.0))
    # Retrying the original correction's device must not credit it.
    client.post("/rebuild", params={"device_id": "D1"})

    body = impact(client, "S2", 2).json()
    assert body["state"] == "superseded"
    assert body["published_intervals"] == []
    assert "superseded" in body["note"]

    # The later correction is independently trackable with its own baseline.
    later = impact(client, "S8", 1)
    assert later.status_code == 422  # revision 1 is not a correction


def test_superseded_event_frozen_against_even_later_rebuilds(ctx):
    client, _, _ = ctx
    _seed_two_readings(client)
    client.post(
        "/readings", headers={"X-Fail-Rebuild": "true"},
        json=reading("S2", 2, "D1", "2026-01-01T02:00:00Z", "M1", 145.0),
    )
    client.post("/readings", json=reading("S8", 1, "D1", "2026-01-01T03:00:00Z", "M1", 160.0))
    client.post("/rebuild", params={"device_id": "D1"})
    first = impact(client, "S2", 2).json()
    # Yet another rebuild must not change the frozen answer.
    client.post("/rebuild", params={"device_id": "D1", "all_devices": False})
    second = impact(client, "S2", 2).json()
    assert first == second


def test_idempotent_resubmission_creates_no_event(ctx):
    client, _, _ = ctx
    _seed_two_readings(client)
    payload = reading("S2", 2, "D1", "2026-01-01T02:00:00Z", "M1", 145.0)
    client.post("/readings", json=payload)
    # Identical re-submission is idempotent and must not register a new event
    # beyond the one for revision 2 itself.
    again = client.post("/readings", json=payload)
    assert again.json()["idempotent"] is True
    body = impact(client, "S2", 2).json()
    assert body["state"] == "published"


def test_endpoint_documented_in_openapi_and_serves_swagger(ctx):
    client, _, _ = ctx
    spec = client.get("/openapi.json").json()
    assert "/correction-impact" in spec["paths"]
    assert client.get("/docs").status_code == 200
