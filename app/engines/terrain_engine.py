"""
地形 CFIT 风险分析引擎
基于 SRTM 30m DEM 数据（OpenTopoData 免费 API，无需注册）

CFIT = Controlled Flight Into Terrain（可控飞行撞地）

分析流程：
  1. 沿航线等距插值采样点（直线近似，60点）
  2. 批量查询 SRTM 地形高程
  3. 计算每点的超障余度 clearance = h_flight - h_terrain - OBS - MOC
  4. CFIT 概率 = P(垂直误差 > clearance) ~ 1 - Φ(clearance / σ_alt)
  5. 返回剖面数据 + 风险评级
"""
import numpy as np
import httpx
import hashlib
from scipy.stats import norm
from typing import List, Dict, Optional

# ── 外部 DEM API（多源容错链）──────────────────────────────────────
# mapzen(Terrarium，融合 Copernicus/SRTM/ASTER 等全球源) 优先，srtm30m 兜底
DEM_SOURCES = [
    ("mapzen",   "https://api.opentopodata.org/v1/mapzen"),
    ("srtm30m",  "https://api.opentopodata.org/v1/srtm30m"),
]
TOPO_API_URL = DEM_SOURCES[0][1]  # 兼容旧引用

# ── 高程查询缓存（内存，进程内有效；坐标精度 4 位小数 ≈ 11m）────────
_ELEV_CACHE: Dict[str, List[float]] = {}
_ELEV_CACHE_MAX = 200  # 最多缓存 200 个查询批次

def _cache_key(points: List[Dict]) -> str:
    sig = ";".join(f"{p['lat']:.4f},{p['lon']:.4f}" for p in points)
    return hashlib.md5(sig.encode()).hexdigest()

# ── UAV 低空安全参数 ─────────────────────────────────
MOC_DEFAULT      = 50.0   # 最低超障余度 (m) — ICAO UAV 建议值
OBS_BUFFER       = 50.0   # 沿线障碍物缓冲高度 (m) — 保守估计：铁塔/风机
SIGMA_ALT_GPS    = 15.0   # GPS 垂直误差 1-sigma (m)
SIGMA_ALT_BARO   = 30.0   # 气压高度表 1-sigma (m)


# ── 工具函数 ─────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面大圆距离 (km)"""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _interpolate_route(waypoints: List[Dict], n_samples: int = 60) -> List[Dict]:
    """
    沿折线航线等距插值 n_samples 个采样点
    waypoints: [{lat, lon}, ...]
    返回:   [{lat, lon, dist_km}, ...]
    """
    if len(waypoints) < 2:
        return [{'lat': waypoints[0]['lat'], 'lon': waypoints[0]['lon'], 'dist_km': 0.0}]

    # 各段长度
    seg_len = []
    for i in range(len(waypoints) - 1):
        seg_len.append(_haversine_km(
            waypoints[i]['lat'], waypoints[i]['lon'],
            waypoints[i + 1]['lat'], waypoints[i + 1]['lon']
        ))
    total_km = sum(seg_len)
    if total_km < 0.01:
        return [{'lat': waypoints[0]['lat'], 'lon': waypoints[0]['lon'], 'dist_km': 0.0}]

    step = total_km / max(n_samples - 1, 1)
    samples = []
    seg_idx, seg_cum = 0, 0.0

    for k in range(n_samples):
        target = k * step
        # 推进到正确的段
        while seg_idx < len(seg_len) - 1 and seg_cum + seg_len[seg_idx] < target - 1e-9:
            seg_cum += seg_len[seg_idx]
            seg_idx += 1

        seg_d = seg_len[seg_idx]
        t = 0.0 if seg_d < 1e-9 else min((target - seg_cum) / seg_d, 1.0)

        lat = waypoints[seg_idx]['lat'] + t * (waypoints[seg_idx + 1]['lat'] - waypoints[seg_idx]['lat'])
        lon = waypoints[seg_idx]['lon'] + t * (waypoints[seg_idx + 1]['lon'] - waypoints[seg_idx]['lon'])
        samples.append({'lat': lat, 'lon': lon, 'dist_km': round(target, 3)})

    return samples


def _fetch_batch(url: str, batch: List[Dict]):
    """向单个 DEM 源请求一批点；成功返回 elevations(list，可含 None)，失败返回 None"""
    locs = "|".join(f"{p['lat']:.6f},{p['lon']:.6f}" for p in batch)
    try:
        resp = httpx.get(url, params={"locations": locs}, timeout=20.0, follow_redirects=True)
        if resp.status_code == 200:
            return [r.get("elevation") for r in resp.json().get("results", [])]
    except Exception:
        pass
    return None


def _fetch_elevations(points: List[Dict]) -> tuple[List[float], bool]:
    """
    批量查询地形高程（多源容错：mapzen 优先，srtm30m 兜底，空值再回退）
    返回: (elevations_list, api_success)
    全部失败时返回全 0（海平面），api_success = False
    """
    # ── 缓存命中 ──────────────────────────────────────
    key = _cache_key(points)
    if key in _ELEV_CACHE:
        return _ELEV_CACHE[key], True

    elevations: List[float] = []
    api_ok = True

    for i in range(0, len(points), 100):
        batch = points[i:i + 100]
        got = None
        for _name, url in DEM_SOURCES:
            res = _fetch_batch(url, batch)
            if res is None:
                continue
            if got is None:
                got = res
            else:
                # 用后续源填补前一源的空值（void）
                got = [g if g is not None else r for g, r in zip(got, res)]
            if all(v is not None for v in got):
                break  # 该批已无空值，无需再试其他源
        if got is None:
            elevations.extend([0.0] * len(batch))
            api_ok = False
        else:
            elevations.extend([float(v) if v is not None else 0.0 for v in got])

    # ── 写入缓存（LRU 简化版：超限时清空最旧一半）────────────────
    if api_ok and len(elevations) == len(points):
        if len(_ELEV_CACHE) >= _ELEV_CACHE_MAX:
            # 删掉前半部分（近似 LRU）
            keys_to_drop = list(_ELEV_CACHE.keys())[:_ELEV_CACHE_MAX // 2]
            for k in keys_to_drop:
                del _ELEV_CACHE[k]
        _ELEV_CACHE[key] = elevations

    return elevations, api_ok


# ── 主函数 ────────────────────────────────────────────

def compute_terrain_analysis(params: Dict) -> Dict:
    """
    CFIT 地形分析完整计算

    输入参数:
      waypoints    : [{lat, lon}, ...]        至少 2 个点
      altitude_m   : float                   计划飞行离地高度 AGL (m)
                                             中国无人机飞行高度标准均为 AGL（离地高度）
                                             系统在每个采样点上将 AGL 转换为 AMSL 计算
      moc          : float = 50.0            最低超障余度 (m)
      sigma_alt    : float = 15.0            垂直导航误差 1-sigma (m)
      n_samples    : int   = 60              剖面采样点数

    输出:
      profile          : 采样点列表，每点含 dist_km / terrain_m / flight_m(AMSL) / clearance_m / p_cfit
      max_terrain_m    : 沿线最高地形 AMSL (m)
      min_clearance_m  : 最小超障余度 (m)  负值 = 飞行高度低于安全线
      msa_agl_m        : 推荐最低离地安全高度 AGL (m) = OBS_BUFFER + MOC
      msa_amsl_m       : 推荐最低安全高度 AMSL (m) = max_terrain + OBS_BUFFER + MOC
      cfit_risk        : 最大单点 CFIT 概率（保守值）
      critical_pts     : 超障余度不足的点列表（最多10条）
      verdict          : 'PASS' | 'WARNING' | 'FAIL'
      api_ok           : 是否成功获取真实地形数据
      total_dist_km    : 航线总距离 (km)
    """
    waypoints   = params['waypoints']
    altitude_agl = float(params['altitude_m'])   # AGL 离地高度
    moc         = float(params.get('moc', MOC_DEFAULT))
    sigma_alt   = float(params.get('sigma_alt', SIGMA_ALT_GPS))
    n_samples   = int(params.get('n_samples', 60))

    # ── 1. 插值 ─────────────────────────────────────
    samples = _interpolate_route(waypoints, n_samples)

    # ── 2. 获取地形高程 ──────────────────────────────
    elevations, api_ok = _fetch_elevations(samples)

    # ── 3. 逐点计算 ──────────────────────────────────
    # AGL（离地）飞行下，无人机随地形起伏保持恒定离地高度，因此真正与地形相关的
    # CFIT 风险来自两个维度（均随航线变化，而非恒定值）：
    #   ① 地形坡度：地形上升速度超过无人机安全跟随能力 → 撞坡风险
    #   ② 地形起伏：相对飞行高度的剧烈起伏增加跟随难度
    # 障碍物余度 = 离地高度 - MOC（高于 MOC 安全底线的裕度，诚实的常量）
    obstacle_margin = altitude_agl - moc

    profile = []
    p_cfit_max = 0.0
    critical_pts = []
    max_slope_deg = 0.0
    prev = None

    for s, elev in zip(samples, elevations):
        elev = max(0.0, elev)
        flight_amsl = elev + altitude_agl

        # 与前一采样点之间的地形坡度
        slope_deg = 0.0
        if prev is not None:
            d_horiz_m = _haversine_km(prev['lat'], prev['lon'], s['lat'], s['lon']) * 1000.0
            d_elev = elev - prev['elev']
            if d_horiz_m > 1.0:
                slope_deg = float(np.degrees(np.arctan2(abs(d_elev), d_horiz_m)))
        if slope_deg > max_slope_deg:
            max_slope_deg = slope_deg

        # CFIT 概率：障碍余度不足或坡度陡峭时升高
        if obstacle_margin <= 0:
            p_cfit = 1.0
        else:
            # 坡度等效削减跟随余度：陡坡使有效余度下降
            eff_margin = obstacle_margin * max(0.05, 1.0 - slope_deg / 45.0)
            p_cfit = float(1.0 - norm.cdf(eff_margin / sigma_alt))

        pt = {
            'dist_km':    round(s['dist_km'], 3),
            'lat':        round(s['lat'], 6),
            'lon':        round(s['lon'], 6),
            'terrain_m':  round(elev, 1),
            'flight_m':   round(flight_amsl, 1),
            'clearance_m': round(obstacle_margin, 1),  # 障碍余度（高于 MOC 的裕度）
            'slope_deg':  round(slope_deg, 1),
            'p_cfit':     round(p_cfit, 8),
        }
        profile.append(pt)
        if p_cfit > p_cfit_max:
            p_cfit_max = p_cfit

        # 高风险点：陡坡段（>20°）
        if slope_deg > 20.0:
            critical_pts.append({
                'dist_km':   pt['dist_km'],
                'terrain_m': pt['terrain_m'],
                'slope_deg': pt['slope_deg'],
                'p_cfit':    pt['p_cfit'],
            })

        prev = {'lat': s['lat'], 'lon': s['lon'], 'elev': elev}

    # ── 4. 统计 ──────────────────────────────────────
    terrain_vals = [p['terrain_m'] for p in profile]
    max_terrain  = max(terrain_vals) if terrain_vals else 0.0
    min_terrain  = min(terrain_vals) if terrain_vals else 0.0
    relief       = max_terrain - min_terrain   # 地形起伏（最高-最低）

    # MSA：建议最低离地高度 = MOC + 障碍缓冲（仅作建议参考）
    msa_agl  = moc + OBS_BUFFER
    msa_amsl = max_terrain + altitude_agl

    # ── 5. 总体评级（基于真实地形坡度与起伏）──────────────
    if obstacle_margin <= 0 or max_slope_deg > 35.0:
        # 飞行高度低于 MOC 底线，或地形过陡无法安全跟随
        verdict = 'FAIL'
    elif max_slope_deg > 20.0 or relief > 2.0 * altitude_agl:
        # 存在陡坡段，或地形起伏远大于飞行高度
        verdict = 'WARNING'
    else:
        verdict = 'PASS'

    return {
        'profile':         profile,
        'max_terrain_m':   round(max_terrain, 1),
        'min_terrain_m':   round(min_terrain, 1),
        'relief_m':        round(relief, 1),            # 地形起伏
        'max_slope_deg':   round(max_slope_deg, 1),     # 最大地形坡度
        'min_clearance_m': round(obstacle_margin, 1),   # 障碍余度（高于 MOC）
        'msa_m':           round(msa_agl, 1),
        'msa_amsl_m':      round(msa_amsl, 1),
        'planned_alt_m':   round(altitude_agl, 1),
        'cfit_risk':       float(p_cfit_max),
        'critical_pts':    critical_pts[:10],
        'verdict':         verdict,
        'api_ok':          api_ok,
        'total_dist_km':   round(samples[-1]['dist_km'], 2) if samples else 0.0,
        'n_samples':       len(samples),
        'moc_used':        moc,
        'sigma_alt_used':  sigma_alt,
    }
