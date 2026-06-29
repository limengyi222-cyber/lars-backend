"""
LARS 后端服务 - 完整版
低空航路航线安全风险评估系统 API

引擎:
  - CREAM: 三维碰撞风险 (ICAO Doc 9689 + NTU ATMRI 方法论)
  - TVR:   总垂直风险 (LHD 三情形叠加)
  - Hotspot: K-means++ 热点聚类
  - Network: 航路网络分析 (NetworkX)
  - Complexity: 空域复杂度 (Interacting Particle System)

实时数据: adsb.fi (替代 OpenSky，避免云端 IP 封锁)
"""

from fastapi import FastAPI, HTTPException, Query, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import httpx
import os
import json
import logging
import time
import threading
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 引擎导入 ──────────────────────────────────────
from .engines.cream_engine import compute_3d_risk, compute_uav_collision
from .engines.tvr_engine import compute_total_vertical_risk
from .engines.hotspot_engine import detect_hotspots_kmeans
from .engines.network_engine import analyze_airway_network
from .engines.complexity_engine import compute_airspace_complexity
from .engines.crossing_detector import detect_crossings
from .engines.terrain_engine import compute_terrain_analysis
from .engines.airspace_engine import check_route_airspace
from .engines.ground_risk_engine import assess_ground_risk
from .engines.analytics_engine import (
    log_registration, log_assessment, log_export, get_admin_stats
)
from .engines.auth_engine import auth_register, auth_login, get_user_by_token, clear_session_token, set_user_role
from .engines.weather_engine import fetch_gba_weather
from .engines.sim_traffic_engine import generate_sim_traffic
from .engines.history_engine import save_assessment, get_history, get_stats_summary

app = FastAPI(
    title="LARS API",
    description="低空航路航线安全风险评估系统 — CREAM / TVR / 热点 / 网络 / 复杂度",
    version="2.0.0"
)

# gzip 压缩（适飞网格等大 JSON：5MB → ~1MB）
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lars-risk-assessment.pages.dev",
    ],
    # 生产锁定 pages.dev（含预览子域）；本地开发允许任意 localhost 端口
    allow_origin_regex=r"https://[a-z0-9-]+\.lars-risk-assessment\.pages\.dev|http://localhost:\d+|http://127\.0\.0\.1:\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 简单内存速率限制器（进程内，重启清零）─────────────────────────────
_RL_BUCKETS: dict = defaultdict(list)
_RL_LOCK = threading.Lock()

def _rate_check(request: Request, limit: int = 30, window: int = 60) -> None:
    """
    检查单 IP 速率，超限抛 429。
    limit: window 秒内最多允许的请求数
    """
    ip = (request.client.host if request.client else "unknown")
    now = time.monotonic()
    cutoff = now - window
    with _RL_LOCK:
        _RL_BUCKETS[ip] = [t for t in _RL_BUCKETS[ip] if t > cutoff]
        if len(_RL_BUCKETS[ip]) >= limit:
            raise HTTPException(
                status_code=429,
                detail=f"请求过于频繁，{window} 秒内最多 {limit} 次计算请求，请稍后再试"
            )
        _RL_BUCKETS[ip].append(now)

# ═══════════════════════════════════════════════════
# 鉴权依赖（FastAPI Depends）
# ═══════════════════════════════════════════════════

ADMIN_TOKEN = os.environ.get("LARS_ADMIN_TOKEN", "")

def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    """
    Bearer Token 鉴权：从 Authorization 头提取 token，查询 registrations 表
    用于需要登录的端点（历史记录保存/查询）
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未登录，请先注册/登录 LARS")
    token = authorization[7:].strip()
    user = get_user_by_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Session 已失效，请重新登录")
    return user


def get_optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """
    可选鉴权：有 token 则验证，无 token 则匿名（不报错）
    用于历史记录保存：匿名用户仍可保存，但不与账号关联
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None
    try:
        return get_current_user(authorization)
    except HTTPException:
        return None


def require_admin(authorization: Optional[str] = Header(None)) -> dict:
    """管理员鉴权：验证 LARS_ADMIN_TOKEN 或管理员账号"""
    # 方式1：直接用管理员 token（保持向后兼容）
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        if ADMIN_TOKEN and token == ADMIN_TOKEN:
            return {"role": "admin", "phone": "admin"}
        # 方式2：通过账号 role 字段
        user = get_user_by_token(token)
        if user and user.get("role") == "admin":
            return user
    raise HTTPException(status_code=403, detail="需要管理员权限")


# ── 缓存 ──────────────────────────────────────────
_GRIDS_CACHE = None

# ═══════════════════════════════════════════════════
# 请求/响应模型
# ═══════════════════════════════════════════════════

class CREAMRequest(BaseModel):
    """CREAM 三维碰撞风险请求（ICAO Doc 9689 + NTU ATMRI v2.1）"""
    Sx: float = Field(2.0,   description="纵向间隔 (NM)")
    Sy: float = Field(0.5,   description="侧向间隔 / 走廊宽度 (NM)")
    Sz: float = Field(30.0,  description="垂直间隔 (ft)")
    RNP: float = Field(0.1,  description="导航精度 RNP (NM)")
    V: float = Field(30.0,   description="平均速度 (knots)")
    y_dot: float = Field(5.0, description="侧向速率 (knots)")
    z_dot: float = Field(2.0, description="垂直速率 (ft/min)")
    lambda_x: float = Field(0.003, description="纵向机身半长 (NM)")
    lambda_y: float = Field(0.003, description="侧向机身半宽 (NM)")
    lambda_z: float = Field(0.002, description="垂直机身半高 (NM)")
    lambda_xy: float = Field(0.003, description="水平截面特征尺寸 (NM)")
    sigma_aad: float = Field(5.0, description="AAD 标准差 (ft) — 实际高度偏差")
    sigma_ase: float = Field(3.0, description="ASE 标准差 (ft) — 高度表系统误差")
    Pz_0: float = Field(0.0,  description="保留字段（v2.1 已从 TVE 模型自动计算，传 0 即可）")
    Ey_opp: float = Field(0.3, description="对向飞行暴露量 (0~1)")
    # ── v2.1 新增参数 ──────────────────────────────────────
    N_ac: float = Field(10.0,   description="评估期内走廊飞行架次（用于推导 n_z_equiv）")
    T_period: float = Field(3600.0, description="评估时间窗口 (s)")
    delta_V: float = Field(0.0, description="同向速度偏差 (knots)；0 = 自动取 5%V")


class UAVCollisionRequest(BaseModel):
    """无人机入侵碰撞概率请求"""
    d: float = Field(500.0, description="初始距离 (m)")
    beta: float = Field(0.0, description="俯仰角 (度)")
    theta: float = Field(30.0, description="水平交叉角 (度)")
    Vh: float = Field(60.0, description="水平速度 (km/h)")
    Rm: float = Field(50.0, description="水平保护半径 (m)")
    Qm: float = Field(30.0, description="垂直保护半径 (m)")
    sigma_h: float = Field(5.0, description="水平扩散系数")
    sigma_v: float = Field(2.0, description="垂直扩散系数")
    T: float = Field(60.0, description="评估时间窗口 (s)")


class TVRRequest(BaseModel):
    """总垂直风险请求"""
    s1: float = Field(20.0, description="正常 AAD 标准差 (ft)")
    s2: float = Field(100.0, description="异常 AAD 标准差 (ft)")
    alpha: float = Field(0.001, description="异常飞行比例")
    nCLD: float = Field(2.0, description="CLD 事件次数")
    nWL: float = Field(1.0, description="WL 事件次数")
    twl: float = Field(0.1, description="WL 持续时间比例")
    T: float = Field(3600.0, description="评估时间窗口 (s)")


class TerrainWaypoint(BaseModel):
    lat: float
    lon: float

class TerrainRequest(BaseModel):
    """地形 CFIT 风险分析请求"""
    waypoints:   List[TerrainWaypoint] = Field(..., description="航线节点列表 [{lat,lon}]，至少2个")
    altitude_m:  float = Field(120.0,  description="计划飞行高度 (m AMSL 或 AGL，见 altitude_type)")
    altitude_type: str = Field("amsl", description="高度类型: 'amsl'(海拔) | 'agl'(离地)")
    moc:         float = Field(50.0,   description="最低超障余度 (m)")
    sigma_alt:   float = Field(15.0,   description="垂直导航误差 1-sigma (m)")
    n_samples:   int   = Field(60,     description="剖面采样点数")


class AirspaceWaypoint(BaseModel):
    lat: float
    lon: float

class AirspaceRequest(BaseModel):
    """空域穿越检查请求"""
    waypoints: List[AirspaceWaypoint] = Field(..., description="航线节点 [{lat,lon}]，至少2个")
    n_samples: int = Field(80, description="沿线采样点数")


class HotspotRequest(BaseModel):
    """热点检测请求"""
    k_clusters: int = Field(9, description="聚类数量")
    bbox: List[float] = Field([112.5, 22.0, 115.0, 23.5], description="[minLon,minLat,maxLon,maxLat]")
    use_live_data: bool = Field(True, description="使用实时 ADS-B 数据")
    data_source: str = Field("live", description="数据源: 'live'(实时ADS-B有人机) | 'simulated'(仿真低空流量)")
    sim_density: float = Field(1.0, description="仿真流量密度系数（仅 simulated 时生效）")


# ═══════════════════════════════════════════════════
# 工具函数：adsb.fi 实时数据（替代 OpenSky）
# ═══════════════════════════════════════════════════

async def _fetch_adsb_flights(min_lon: float, min_lat: float, max_lon: float, max_lat: float):
    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2
    url = f"https://opendata.adsb.fi/api/v2/lat/{center_lat}/lon/{center_lon}/dist/250"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, timeout=20)
    if resp.status_code != 200:
        return []
    aircraft = resp.json().get("aircraft") or []
    flights = []
    for ac in aircraft:
        lat = ac.get("lat")
        lon = ac.get("lon")
        if lat is None or lon is None:
            continue
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            continue
        alt = ac.get("alt_baro") or ac.get("alt_geom")
        on_ground = isinstance(alt, str) and alt == "ground"
        flights.append({
            "icao24":      ac.get("hex", ""),
            "callsign":    (ac.get("flight") or "").strip(),
            "longitude":   lon,
            "latitude":    lat,
            "altitude":    None if on_ground else alt,
            "on_ground":   on_ground,
            "velocity":    ac.get("gs"),
            "heading_deg": ac.get("track"),
            "vertical_rate": ac.get("baro_rate"),
        })
    return flights


# ═══════════════════════════════════════════════════
# 基础端点
# ═══════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "LARS API v2",
        "status": "running",
        "engines": ["cream", "tvr", "hotspot", "network", "complexity"],
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


# ═══════════════════════════════════════════════════
# CREAM 三维碰撞风险
# ═══════════════════════════════════════════════════

@app.post("/api/v1/cream/compute")
def compute_cream(req: CREAMRequest, request: Request):
    """
    CREAM 三维碰撞风险计算
    基于 ICAO Doc 9689 + NTU ATMRI CREAM 方法论
    使用 SciPy 高精度数值积分（比前端 JS 近似更准确）
    """
    _rate_check(request, limit=30, window=60)
    try:
        result = compute_3d_risk(req.dict())
        return result
    except Exception as e:
        logger.exception("CREAM 计算失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cream/uav-collision")
def compute_uav(req: UAVCollisionRequest, request: Request):
    """
    无人机入侵碰撞概率（布朗运动模型）
    计算 UAV 与受保护空域的碰撞概率
    """
    _rate_check(request, limit=30, window=60)
    try:
        result = compute_uav_collision(req.dict())
        return result
    except Exception as e:
        logger.exception("UAV 碰撞计算失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 总垂直风险 (TVR)
# ═══════════════════════════════════════════════════

@app.post("/api/v1/tvr/compute")
def compute_tvr(req: TVRRequest, request: Request):
    """
    总垂直风险 Naz_total
    叠加技术误差、CLD（高度偏差）、WL（错误高度层）三种情形
    """
    _rate_check(request, limit=30, window=60)
    try:
        result = compute_total_vertical_risk(req.dict())
        return result
    except Exception as e:
        logger.exception("TVR 计算失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 地形 CFIT 分析
# ═══════════════════════════════════════════════════

@app.post("/api/v1/terrain/analyze")
def analyze_terrain(req: TerrainRequest, request: Request):
    """
    地形 CFIT 风险分析
    - 调用 OpenTopoData SRTM 30m API 获取沿线地形高程
    - 计算超障余度、MSA、CFIT 碰撞概率
    - 返回地形剖面 + 风险评级 (PASS/WARNING/FAIL)
    """
    _rate_check(request, limit=15, window=60)
    try:
        wps = [{"lat": w.lat, "lon": w.lon} for w in req.waypoints]
        result = compute_terrain_analysis({
            "waypoints":   wps,
            "altitude_m":  req.altitude_m,
            "moc":         req.moc,
            "sigma_alt":   req.sigma_alt,
            "n_samples":   req.n_samples,
        })
        return result
    except Exception as e:
        logger.exception("地形分析失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 热点检测
# ═══════════════════════════════════════════════════

@app.post("/api/v1/hotspots/detect")
async def detect_hotspots(req: HotspotRequest, request: Request):
    """
    K-means++ 热点聚类检测
    1. 从 adsb.fi 拉取实时航班
    2. 检测航迹交叉点
    3. K-means++ 聚类 → 识别高风险热点
    """
    _rate_check(request, limit=10, window=60)
    try:
        bbox = req.bbox
        if req.data_source == "simulated":
            flights = generate_sim_traffic(density=req.sim_density)
            src_label = "simulated"
            logger.info(f"生成 {len(flights)} 架次仿真低空流量")
        elif req.use_live_data:
            flights = await _fetch_adsb_flights(bbox[0], bbox[1], bbox[2], bbox[3])
            src_label = "adsb.fi (live)"
            logger.info(f"获取到 {len(flights)} 架次实时航班")
        else:
            flights = _demo_flights()
            src_label = "demo"

        crossings = detect_crossings(flights)
        logger.info(f"检测到 {len(crossings)} 个交叉点")

        if not crossings:
            return {"hotspots": [], "total_crossings": 0,
                    "flights_analyzed": len(flights), "message": "无有效交叉点"}

        hotspots = detect_hotspots_kmeans(crossings=crossings, k=req.k_clusters)
        return {
            "hotspots": hotspots,
            "total_crossings": len(crossings),
            "flights_analyzed": len(flights),
            "data_source": src_label,
        }
    except Exception as e:
        logger.exception("热点检测失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 航路网络分析
# ═══════════════════════════════════════════════════

@app.post("/api/v1/network/analyze")
async def analyze_network(
    request: Request,
    percolation_threshold: float = Query(0.3, description="渗流阈值"),
    bbox: str = Query("112.5,22.0,115.0,23.5", description="minLon,minLat,maxLon,maxLat"),
    data_source: str = Query("live", description="数据源: 'live' | 'simulated'"),
    sim_density: float = Query(1.0, description="仿真流量密度系数")
):
    """
    航路网络分析（NetworkX）
    - 边介数中心性 / 节点介数中心性 / 网络渗流 / 关键航段识别
    - data_source=simulated 时使用大湾区低空走廊仿真流量（带轨迹，可建图）
    """
    _rate_check(request, limit=10, window=60)
    try:
        if data_source == "simulated":
            flights = generate_sim_traffic(density=sim_density)
            src_label = "simulated"
        else:
            b = [float(x) for x in bbox.split(",")]
            flights = await _fetch_adsb_flights(b[0], b[1], b[2], b[3])
            src_label = "adsb.fi"
            if not flights:
                flights = _demo_flights()
                src_label = "demo"
        result = analyze_airway_network(
            flights=flights,
            percolation_threshold=percolation_threshold
        )
        return {**result, "flights_analyzed": len(flights), "data_source": src_label}
    except Exception as e:
        logger.exception("网络分析失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 空域复杂度
# ═══════════════════════════════════════════════════

@app.post("/api/v1/complexity/compute")
async def compute_complexity_endpoint(
    request: Request,
    rh_nm: float = Query(5.0, description="水平保护半径 (NM)"),
    rv_ft: float = Query(1000.0, description="垂直保护半径 (ft)"),
    look_ahead_sec: int = Query(600, description="预测时长 (s)"),
    grid_size: int = Query(10, description="网格大小"),
    bbox: str = Query("112.5,22.0,115.0,23.5"),
    data_source: str = Query("live", description="数据源: 'live' | 'simulated'"),
    sim_density: float = Query(1.0, description="仿真流量密度系数")
):
    """
    空域复杂度（Interacting Particle System）
    基于 Prandini 等人方法，计算各空域点的碰撞概率积分
    data_source=simulated 时使用大湾区低空走廊仿真流量
    """
    _rate_check(request, limit=10, window=60)
    try:
        if data_source == "simulated":
            flights = generate_sim_traffic(density=sim_density)
            src_label = "simulated"
        else:
            b = [float(x) for x in bbox.split(",")]
            flights = await _fetch_adsb_flights(b[0], b[1], b[2], b[3])
            src_label = "adsb.fi"
            if not flights:
                flights = _demo_flights()
                src_label = "demo"
        result = compute_airspace_complexity(
            flights=flights,
            rh_nm=rh_nm, rv_ft=rv_ft,
            look_ahead_sec=look_ahead_sec,
            grid_size=grid_size
        )
        return {**result, "live_aircraft": len(flights), "data_source": src_label}
    except Exception as e:
        logger.exception("复杂度计算失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 实时航班 & 空域网格（保留原有接口）
# ═══════════════════════════════════════════════════

@app.get("/api/v1/flights/live")
async def live_flights(
    min_lon: float = 112.5, min_lat: float = 22.0,
    max_lon: float = 115.0, max_lat: float = 23.5
):
    """实时航班（adsb.fi）"""
    try:
        flights = await _fetch_adsb_flights(min_lon, min_lat, max_lon, max_lat)
        return {"flights": flights, "total": len(flights)}
    except Exception as e:
        return {"flights": [], "error": str(e), "total": 0}


@app.post("/api/v1/airspace/route-check")
def airspace_route_check(req: AirspaceRequest):
    """
    航线空域穿越检查
    - 基于 46,331 个广东适飞网格（gba_grids.json）
    - 逐点判定：适飞区 / 受限区 / 无数据区
    - 返回剖面、受限段列表、综合评级
    """
    try:
        wps = [{"lat": w.lat, "lon": w.lon} for w in req.waypoints]
        result = check_route_airspace({
            "waypoints":  wps,
            "n_samples":  req.n_samples,
        })
        return result
    except Exception as e:
        logger.exception("空域穿越检查失败")
        raise HTTPException(status_code=500, detail=str(e))


class GroundRiskRequest(BaseModel):
    """地面风险（GRC）评估请求"""
    waypoints: List[AirspaceWaypoint] = Field(..., description="航线节点 [{lat,lon}]")
    ac_type: str = Field("small", description="机型: micro/light/small/medium/large")
    n_samples: int = Field(100, description="沿线采样点数")
    controlled: bool = Field(False, description="是否全程受控地面区域")
    tiny250g: bool = Field(False, description="是否≤250g且≤25m/s（GRC恒为1）")

@app.post("/api/v1/groundrisk/assess")
def groundrisk_assess(req: GroundRiskRequest):
    """
    地面风险等级（GRC）—— 中国民航局《特定类运行风险评估与缓解指南》
    沿线采样人口密度（WorldPop 2020 1km）→ 人口分级 → 查官方初始地面风险表
    """
    try:
        wps = [{"lat": w.lat, "lon": w.lon} for w in req.waypoints]
        return assess_ground_risk({
            "waypoints": wps, "ac_type": req.ac_type, "n_samples": req.n_samples,
            "controlled": req.controlled, "tiny250g": req.tiny250g,
        })
    except Exception as e:
        logger.exception("地面风险评估失败")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/airspace/grids")
async def airspace_grids():
    """广东适飞网格数据"""
    global _GRIDS_CACHE
    if _GRIDS_CACHE is None:
        grid_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gba_grids.json")
        with open(grid_path, "r") as f:
            _GRIDS_CACHE = f.read()
    return JSONResponse(content=json.loads(_GRIDS_CACHE))


# ═══════════════════════════════════════════════════
# 数据分析 & 管理后台
# ═══════════════════════════════════════════════════

class RegistrationLog(BaseModel):
    name:    str
    phone:   str
    company: str = ""
    ip:      str = ""

class AssessmentLog(BaseModel):
    phone:           str = ""
    from_city:       str = ""
    to_city:         str = ""
    altitude_m:      Optional[float] = None
    route_km:        Optional[float] = None
    aircraft_type:   str = ""
    cream_risk:      Optional[float] = None
    cream_verdict:   str = ""
    terrain_verdict: str = ""
    airspace_verdict:str = ""
    params:          dict = {}

class ExportLog(BaseModel):
    phone:     str = ""
    from_city: str = ""
    to_city:   str = ""
    mode:      str = ""


@app.post("/api/v1/analytics/register")
def analytics_register(req: RegistrationLog):
    log_registration(req.name, req.phone, req.company, req.ip)
    return {"ok": True}


@app.post("/api/v1/analytics/assessment")
def analytics_assessment(req: AssessmentLog):
    log_assessment(req.dict())
    return {"ok": True}


@app.post("/api/v1/analytics/export")
def analytics_export(req: ExportLog):
    log_export(req.phone, req.from_city, req.to_city, req.mode)
    return {"ok": True}


# ── 账号注册 / 登录 ────────────────────────────────────────────────

class AuthRegisterReq(BaseModel):
    name:     str
    phone:    str
    password: str
    company:  str = ""

class AuthLoginReq(BaseModel):
    phone:    str
    password: str

@app.post("/api/v1/auth/register")
def api_auth_register(req: AuthRegisterReq):
    if not req.name or not req.phone or not req.password:
        raise HTTPException(status_code=400, detail="姓名、手机号和密码不能为空")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    result = auth_register(req.name, req.phone, req.password, req.company)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "注册失败"))
    return result

@app.post("/api/v1/auth/login")
def api_auth_login(req: AuthLoginReq):
    if not req.phone or not req.password:
        raise HTTPException(status_code=400, detail="手机号和密码不能为空")
    result = auth_login(req.phone, req.password)
    if not result.get("ok"):
        raise HTTPException(status_code=401, detail=result.get("error", "登录失败"))
    return result


@app.post("/api/v1/auth/logout")
def api_auth_logout(current_user: dict = Depends(get_current_user)):
    """退出登录 — 清除服务器端 session_token"""
    phone = current_user.get("phone", "")
    result = clear_session_token(phone)
    return {"ok": True, "message": "已退出登录"}


# ── 管理员：设置用户角色 ────────────────────────────────────────────

class SetRoleRequest(BaseModel):
    role: str  # 'admin' 或 'user'

@app.patch("/api/v1/admin/user/{phone}/role")
def admin_set_role(
    phone: str,
    req: SetRoleRequest,
    current_user: dict = Depends(require_admin)
):
    """管理员设置用户角色（admin/user）"""
    if req.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="角色只能是 'admin' 或 'user'")
    result = set_user_role(phone, req.role)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "操作失败"))
    return result


# ═══════════════════════════════════════════════════
# 气象数据
# ═══════════════════════════════════════════════════

@app.get("/api/v1/weather/gba")
async def get_weather(
    bearing: float = Query(None, description="航路方位角(度)，用于计算侧风"),
    lat: float = Query(None, description="查询纬度（航线中点）；缺省用广州"),
    lon: float = Query(None, description="查询经度（航线中点）；缺省用广州"),
):
    """
    实时气象（OpenWeatherMap）
    返回风速/风向/能见度 + CREAM 参数建议（Vy, RNP）；可按 lat/lon 取航线所在地气象
    需配置环境变量 OWM_API_KEY
    """
    try:
        result = await fetch_gba_weather(route_bearing_deg=bearing, lat=lat, lon=lon)
        return result
    except Exception as e:
        logger.exception("气象获取失败")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════
# 评估历史
# ═══════════════════════════════════════════════════

class HistorySaveRequest(BaseModel):
    phone:          str   = ""
    mode:           str   = ""
    from_city:      str   = ""
    to_city:        str   = ""
    risk_value:     Optional[float] = None
    verdict:        str   = ""
    params:         dict  = {}
    result_summary: dict  = {}

@app.post("/api/v1/history/save")
def history_save(
    req: HistorySaveRequest,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    保存评估结果到历史记录（Supabase assessment_history 表）
    - 已登录用户：使用账号绑定的 phone，忽略请求体中的 phone
    - 匿名用户：使用请求体中的 phone（或空字符串）
    """
    data = req.dict()
    if current_user:
        data["phone"] = current_user.get("phone", data["phone"])
    result = save_assessment(data)
    return result


@app.get("/api/v1/history/list")
def history_list(
    phone: str = Query("", description="手机号（空=返回全部最近记录）"),
    limit: int = Query(20, description="最多返回条数"),
    offset: int = Query(0, description="分页偏移量"),
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    查询评估历史
    - 已登录用户：自动过滤自己的记录，忽略 phone 参数
    - 匿名/管理员：使用 phone 参数过滤
    - 支持 offset 分页
    """
    if current_user and current_user.get("role") != "admin":
        phone = current_user.get("phone", phone)
    rows = get_history(phone=phone, limit=limit, offset=offset)
    return {"history": rows, "total": len(rows), "offset": offset, "limit": limit}


@app.get("/api/v1/history/stats")
def history_stats(current_user: dict = Depends(require_admin)):
    """评估历史汇总统计（仅管理员）"""
    return get_stats_summary()


@app.get("/api/v1/admin/stats")
def admin_stats(current_user: dict = Depends(require_admin)):
    """管理后台统计（Bearer token 或原 query param token 均支持）"""
    return get_admin_stats()


# ═══════════════════════════════════════════════════
# 演示数据（无实时数据时的后备）
# ═══════════════════════════════════════════════════

def _demo_flights():
    import random
    random.seed(42)
    return [
        {
            "icao24": f"demo{i:04d}",
            "callsign": f"CSN{1000+i}",
            "longitude": 112.5 + random.random() * 2.5,
            "latitude": 22.0 + random.random() * 1.5,
            "altitude": random.randint(1000, 35000),
            "on_ground": False,
            "velocity": random.randint(200, 500),
            "heading_deg": random.randint(0, 359),
            "vertical_rate": random.randint(-500, 500),
        }
        for i in range(30)
    ]
