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
from scipy.stats import norm
from typing import List, Dict, Optional

# ── 外部 DEM API ──────────────────────────────────────
TOPO_API_URL = "https://api.opentopodata.org/v1/srtm30m"

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


def _fetch_elevations(points: List[Dict]) -> tuple[List[float], bool]:
    """
    批量查询 OpenTopoData SRTM 30m 高程
    返回: (elevations_list, api_success)
    失败时返回全 0（海平面），api_success = False
    """
    elevations: List[float] = []
    api_ok = True

    for i in range(0, len(points), 100):
        batch = points[i:i + 100]
        locs = "|".join(f"{p['lat']:.6f},{p['lon']:.6f}" for p in batch)

        try:
            resp = httpx.get(
                TOPO_API_URL,
                params={"locations": locs},
                timeout=20.0,
                follow_redirects=True
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                for r in results:
                    elevations.append(float(r.get("elevation") or 0.0))
            else:
                elevations.extend([0.0] * len(batch))
                api_ok = False
        except Exception:
            elevations.extend([0.0] * len(batch))
            api_ok = False

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
    profile = []
    p_cfit_max = 0.0
    critical_pts = []

    for s, elev in zip(samples, elevations):
        elev = max(0.0, elev)

        # AGL → AMSL：飞行海拔 = 地形海拔 + 离地高度
        flight_amsl = elev + altitude_agl

        # 超障余度 = 离地高度 - 障碍物缓冲 - MOC
        # （等价于 flight_amsl - elev - OBS_BUFFER - moc）
        clearance = altitude_agl - OBS_BUFFER - moc

        # CFIT 概率 = P(垂直误差 > clearance)
        if clearance <= 0:
            p_cfit = 1.0
        else:
            p_cfit = float(1.0 - norm.cdf(clearance / sigma_alt))

        pt = {
            'dist_km':    round(s['dist_km'], 3),
            'lat':        round(s['lat'], 6),
            'lon':        round(s['lon'], 6),
            'terrain_m':  round(elev, 1),
            'flight_m':   round(flight_amsl, 1),   # 显示 AMSL 飞行高度
            'clearance_m': round(clearance, 1),
            'p_cfit':     round(p_cfit, 8),
        }
        profile.append(pt)

        if p_cfit > p_cfit_max:
            p_cfit_max = p_cfit

        # 记录高风险点（仅当 clearance < MOC 才是真正不足）
        if clearance < moc or p_cfit > 1e-5:
            critical_pts.append({
                'dist_km':    pt['dist_km'],
                'terrain_m':  pt['terrain_m'],
                'clearance_m': pt['clearance_m'],
                'p_cfit':     pt['p_cfit'],
            })

    # ── 4. 统计 ──────────────────────────────────────
    terrain_vals   = [p['terrain_m']   for p in profile]
    clearance_vals = [p['clearance_m'] for p in profile]

    max_terrain    = max(terrain_vals)   if terrain_vals   else 0.0
    min_clearance  = min(clearance_vals) if clearance_vals else 0.0

    # MSA（AGL）= 障碍物缓冲 + MOC（与地形无关，因为是 AGL）
    msa_agl  = OBS_BUFFER + moc
    # MSA（AMSL）= 最高地形 + 障碍物缓冲 + MOC（用于图表显示）
    msa_amsl = max_terrain + OBS_BUFFER + moc

    # ── 5. 总体评级 ──────────────────────────────────
    if min_clearance >= moc and p_cfit_max < 1e-7:
        verdict = 'PASS'
    elif min_clearance >= 0 and p_cfit_max < 1e-3:
        verdict = 'WARNING'
    else:
        verdict = 'FAIL'

    return {
        'profile':         profile,
        'max_terrain_m':   round(max_terrain, 1),
        'min_clearance_m': round(min_clearance, 1),
        'msa_m':           round(msa_agl, 1),       # AGL 最低安全高度（前端主显示）
        'msa_amsl_m':      round(msa_amsl, 1),      # AMSL 最低安全高度（图表参考线）
        'planned_alt_m':   round(altitude_agl, 1),  # AGL 申报高度
        'cfit_risk':       float(p_cfit_max),
        'critical_pts':    critical_pts[:10],
        'verdict':         verdict,
        'api_ok':          api_ok,
        'total_dist_km':   round(samples[-1]['dist_km'], 2) if samples else 0.0,
        'n_samples':       len(samples),
        'moc_used':        moc,
        'sigma_alt_used':  sigma_alt,
    }
