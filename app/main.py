"""FastAPI application: meter readings, handovers, intervals, rebuild.

Swagger UI is served at ``/docs`` (ReDoc at ``/redoc``) and the OpenAPI
document at ``/openapi.json``.
"""
from __future__ import annotations

import sqlite3
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.responses import JSONResponse

from .config import Settings
from .database import ConnectionPool
from .errors import (
    ConflictError,
    ImpactAmbiguous,
    ImpactNotFound,
    ImpactNotEligible,
    RebuildFailure,
)
from .models import (
    AcceptOut,
    CorrectionImpactOut,
    HandoverIn,
    IntervalOut,
    ReadingIn,
    RebuildOut,
)
from . import service


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Thread-local SQLite connections; SQLite WAL + busy_timeout guard writes.
        app.state.pool = ConnectionPool(settings.db_path)
        app.state.pool.init_schema()
        # Deliberately NO automatic rebuild on startup: a restart must never
        # resurrect intervals that lost their current-supply eligibility.
        yield
        app.state.pool.close()

    app = FastAPI(
        title="工厂能耗区间结算 API",
        description=(
            "提交按设备区分的累计电表读数与换表交接，按相邻读数生成带来源编号的"
            "区间电量。倒退读数或交接缺失记为无法判定（energy_kwh=null），绝不报零。"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    def db(request: Request) -> sqlite3.Connection:
        return request.app.state.pool.get()

    @app.exception_handler(ConflictError)
    async def conflict_handler(request: Request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": exc.detail})

    @app.exception_handler(ImpactNotFound)
    async def impact_not_found_handler(request: Request, exc: ImpactNotFound):
        return JSONResponse(status_code=404, content={"detail": exc.detail})

    @app.exception_handler(ImpactNotEligible)
    async def impact_not_eligible_handler(request: Request, exc: ImpactNotEligible):
        return JSONResponse(status_code=422, content={"detail": exc.detail})

    @app.exception_handler(ImpactAmbiguous)
    async def impact_ambiguous_handler(request: Request, exc: ImpactAmbiguous):
        return JSONResponse(status_code=409, content={"detail": exc.detail})

    @app.get("/health", tags=["运维"])
    def health(conn: sqlite3.Connection = Depends(db)):
        row = conn.execute("SELECT COUNT(*) AS n FROM intervals WHERE status='current'").fetchone()
        stale = conn.execute("SELECT COUNT(*) AS n FROM intervals WHERE status='stale'").fetchone()
        return {"status": "ok", "current_intervals": row["n"], "stale_intervals": stale["n"]}

    @app.post("/readings", response_model=AcceptOut, status_code=201, tags=["读数"])
    def post_reading(
        payload: ReadingIn,
        request: Request,
        conn: sqlite3.Connection = Depends(db),
        x_fail_rebuild: bool = Header(default=False, include_in_schema=False),
    ):
        result = service.submit_reading(conn, payload)
        rebuild_info = _auto_rebuild(request, conn, result["affected_devices"], x_fail_rebuild)
        return {**result, "accepted": True, "rebuild": rebuild_info}

    @app.post("/handovers", response_model=AcceptOut, status_code=201, tags=["换表交接"])
    def post_handover(
        payload: HandoverIn,
        request: Request,
        conn: sqlite3.Connection = Depends(db),
        x_fail_rebuild: bool = Header(default=False, include_in_schema=False),
    ):
        result = service.submit_handover(conn, payload)
        rebuild_info = _auto_rebuild(request, conn, result["affected_devices"], x_fail_rebuild)
        return {**result, "accepted": True, "rebuild": rebuild_info}

    def _auto_rebuild(request: Request, conn: sqlite3.Connection, devices: list[str], fail: bool) -> dict:
        if not devices:
            return {"triggered": False}
        fail_set = set(devices) if fail else set()
        out = service.rebuild(
            conn,
            devices=devices,
            fail_device=request.app.state.settings.fail_device,
            fail_header_devices=fail_set,
        )
        out["triggered"] = True
        return out

    @app.post("/rebuild", response_model=RebuildOut, tags=["区间重建"])
    def post_rebuild(
        request: Request,
        device_id: str | None = Query(default=None, description="只重建指定设备"),
        all_devices: bool = Query(default=False, description="强制重建全部设备（默认只重建存在 stale 区间的设备）"),
        x_fail_rebuild: bool = Header(default=False, include_in_schema=False),
        conn: sqlite3.Connection = Depends(db),
    ):
        fail_set: set[str] = set()
        if x_fail_rebuild:
            if device_id:
                fail_set.add(device_id)
            else:
                fail_set.update(service.devices_with_stale_intervals(conn))
        try:
            return service.rebuild(
                conn, device_id=device_id, all_devices=all_devices,
                only_stale=not all_devices and device_id is None,
                fail_device=request.app.state.settings.fail_device,
                fail_header_devices=fail_set,
            )
        except RebuildFailure as exc:  # pragma: no cover - rebuild() records per-device failures
            return JSONResponse(status_code=500, content={"detail": str(exc)})

    @app.get("/intervals", response_model=list[IntervalOut], tags=["区间查询"])
    def get_intervals(
        device_id: str | None = Query(default=None),
        status: str = Query(default="current", pattern="^(current|stale|all)$"),
        conn: sqlite3.Connection = Depends(db),
    ):
        rows = service.list_intervals(conn, device_id, status)
        return [
            {
                "id": r["id"],
                "device_id": r["device_id"],
                "start_ts": r["start_ts"],
                "end_ts": r["end_ts"],
                "start_meter_no": r["start_meter_no"],
                "end_meter_no": r["end_meter_no"],
                "energy_kwh": r["energy_kwh"],
                "undeterminable": r["energy_kwh"] is None,
                "reason": r["reason"],
                "status": r["status"],
                "start_source_id": r["start_source_id"],
                "start_revision": r["start_revision"],
                "end_source_id": r["end_source_id"],
                "end_revision": r["end_revision"],
                "handover_source_id": r["handover_source_id"],
                "handover_revision": r["handover_revision"],
            }
            for r in rows
        ]

    @app.get(
        "/correction-impact",
        response_model=CorrectionImpactOut,
        tags=["更正影响"],
        summary="按来源编号与修订号查询一次更正对已发布区间的影响",
    )
    def get_correction_impact(
        source_id: str = Query(..., min_length=1, description="来源编号"),
        revision: int = Query(..., ge=1, description="修订号（必须是已受理的更正修订）"),
        kind: str | None = Query(
            default=None,
            pattern="^(reading|handover)$",
            description="来源类型；同编号同时存在于读数与交接时必填以消歧",
        ),
        conn: sqlite3.Connection = Depends(db),
    ):
        """只服务于修订前后仍属同一设备的更正。

        对照受理前实际可供数的区间快照与此次重建发布的区间快照，仅返回时段、
        电量、可判定状态或来源有变化的区间，并同时给出两侧完整快照。区间边界
        变化时以 removed/added 分别给出旧、新结果，不臆算差额。受理前旧区间
        已待重建（无 current 供数）时 baseline_available=false，不做比较；
        重建未成功只报 pending，重试成功后才发布新区间；期间该设备又受理其他
        数据则标记 superseded，后续结果不计入此次更正。
        """
        return service.get_correction_impact(conn, source_id, revision, kind)

    @app.get("/rebuild-runs", tags=["运维"])
    def get_rebuild_runs(conn: sqlite3.Connection = Depends(db)):
        return [
            dict(r)
            for r in conn.execute(
                "SELECT * FROM rebuild_runs ORDER BY id DESC LIMIT 50"
            ).fetchall()
        ]

    app.state.settings = settings
    return app


app = create_app()
