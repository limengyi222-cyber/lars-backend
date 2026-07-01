"""
地面风险引擎 — 中国民航局《特定类运行风险评估与缓解指南》(基于 CCAR-92, SORA 2.5 本土化)

按官方"初始地面风险等级表格"计算 GRC：
  输入：航线坐标 + 机型(微/轻/小/中/大)[ + 是否受控区 / 是否微型250g]
  过程：沿线采样 WorldPop 人口密度 → 人口分级 → 查官方表 → 初始地面风险 GRC(1-9)
  输出：GRC 值、风险档(低/中/高)、沿线最大人口密度、各档里程占比

人口分级（官方）：受控区 / 极稀少≤5 / 稀少5-300 / 密集300-15000 / 人群上方>15000 (人/km²)
GRC 风险档：1-3 低、4-6 中、7-9 高
特例：重量≤250g 且 最大速度≤25m/s → GRC 恒为 1
"""
import os, json, math, logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# ── 官方初始地面风险表（行=人口分级，列=机型）──
# 列顺序: micro 微, light 轻, small 小, medium 中, large 大；None=不适用
GRC_TABLE = {
    "controlled": {"micro": 1, "light": 1, "small": 1, "medium": 1,    "large": 1},
    "v_sparse":   {"micro": 2, "light": 2, "small": 3, "medium": 3,    "large": 4},   # 极稀少 ≤5
    "sparse":     {"micro": 3, "light": 3, "small": 4, "medium": 5,    "large": 6},   # 稀少 5–300
    "dense":      {"micro": 5, "light": 5, "small": 6, "medium": 7,    "large": 8},   # 密集 300–15000
    "crowd":      {"micro": 7, "light": 8, "small": 9, "medium": None, "large": None},# 人群上方 >15000
}
BAND_CN = {"controlled": "受控地面区域", "v_sparse": "人口极其稀少区", "sparse": "人口稀少区",
           "dense": "人口密集区", "crowd": "人群上方"}
AC_CN = {"micro": "微型", "light": "轻型", "small": "小型", "medium": "中型", "large": "大型"}

# ── 人口密度网格（WorldPop 2020 1km，进程内单例）──
_POP = None

def _load_pop():
    global _POP
    if _POP is not None:
        return _POP
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pop_density_gd.json")
    try:
        _POP = json.load(open(path))
        logger.info(f"人口密度网格加载: {_POP['w']}x{_POP['h']}")
    except Exception as e:
        logger.warning(f"人口密度网格加载失败: {e}")
        _POP = {"bbox": [109, 20, 118, 26], "w": 0, "h": 0, "data": []}
    return _POP

def _density_at(lat: float, lon: float) -> float:
    """采样某点人口密度(人/km²)；超出覆盖返回 -1"""
    p = _load_pop()
    if not p["w"]:
        return -1.0
    W, S, E, N = p["bbox"]
    if not (W <= lon <= E and S <= lat <= N):
        return -1.0
    j = int((lon - W) / p["step_lon"])
    i = int((N - lat) / p["step_lat"])
    if 0 <= i < p["h"] and 0 <= j < p["w"]:
        return float(p["data"][i * p["w"] + j])
    return -1.0

def _band(density: float) -> str:
    if density < 0:      return "unknown"
    if density <= 5:     return "v_sparse"
    if density <= 300:   return "sparse"
    if density <= 15000: return "dense"
    return "crowd"

def _interp(waypoints: List[Dict], n: int = 100) -> List[Dict]:
    """沿折线等距采样 n 点（含里程）"""
    if len(waypoints) < 2:
        return [{"lat": waypoints[0]["lat"], "lon": waypoints[0]["lon"], "dist_km": 0.0}]
    seg = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        dl = math.radians(b["lat"] - a["lat"]); do = math.radians(b["lon"] - a["lon"])
        p1 = math.radians(a["lat"]); p2 = math.radians(b["lat"])
        h = math.sin(dl/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(do/2)**2
        seg.append(6371 * 2 * math.asin(min(1, math.sqrt(h))))
    total = sum(seg) or 1e-9
    step = total / max(n - 1, 1)
    out = []; si = 0; cum = 0.0
    for k in range(n):
        target = k * step
        while si < len(seg) - 1 and cum + seg[si] < target - 1e-9:
            cum += seg[si]; si += 1
        t = 0.0 if seg[si] < 1e-9 else min((target - cum) / seg[si], 1.0)
        lat = waypoints[si]["lat"] + t * (waypoints[si+1]["lat"] - waypoints[si]["lat"])
        lon = waypoints[si]["lon"] + t * (waypoints[si+1]["lon"] - waypoints[si]["lon"])
        out.append({"lat": lat, "lon": lon, "dist_km": round(target, 2)})
    return out


def assess_ground_risk(params: Dict) -> Dict:
    """
    输入: waypoints[{lat,lon}], ac_type(micro/light/small/medium/large),
          n_samples=100, controlled=False(是否全程受控区), tiny250g=False(≤250g且≤25m/s)
    """
    wps = params["waypoints"]
    ac = params.get("ac_type", "small")
    if ac not in AC_CN:
        ac = "small"
    n = int(params.get("n_samples", 100))
    controlled = bool(params.get("controlled", False))
    tiny = bool(params.get("tiny250g", False))

    samples = _interp(wps, n)
    # 逐点密度 + 分级
    band_km = {}  # band → 里程占比(点数)
    max_density = -1.0
    worst_band = "v_sparse"
    band_order = ["v_sparse", "sparse", "dense", "crowd"]
    profile = []
    for s in samples:
        d = _density_at(s["lat"], s["lon"])
        b = _band(d)
        profile.append({"dist_km": s["dist_km"], "lat": round(s["lat"], 6), "lon": round(s["lon"], 6),
                        "density": None if d < 0 else int(d), "band": b})
        eff_b = b if b != "unknown" else "sparse"  # 无数据保守按稀少
        band_km[eff_b] = band_km.get(eff_b, 0) + 1
        if d > max_density:
            max_density = d
        if eff_b in band_order and band_order.index(eff_b) > band_order.index(worst_band):
            worst_band = eff_b

    # 受控区声明 / 微型特例 优先
    if tiny:
        grc, used_band = 1, "tiny250g"
    elif controlled:
        grc, used_band = 1, "controlled"
    else:
        used_band = worst_band
        grc = GRC_TABLE[used_band][ac]

    # 风险档
    if grc is None:
        level = "N/A"
    elif grc <= 3:
        level = "低"
    elif grc <= 6:
        level = "中"
    else:
        level = "高"

    npts = len(samples) or 1
    band_pct = {BAND_CN.get(k, k): round(v / npts * 100, 1) for k, v in band_km.items()}

    return {
        "grc": grc,
        "level": level,
        "ac_cn": AC_CN[ac],
        "decisive_band": BAND_CN.get(used_band, used_band) if used_band in BAND_CN else (
            "微型(≤250g/≤25m·s⁻¹)豁免" if used_band == "tiny250g" else used_band),
        "max_density": None if max_density < 0 else int(max_density),
        "band_pct": band_pct,
        "profile": profile,
        "note": "中国民航局《特定类运行风险评估与缓解指南》初始地面风险；GRC>15000人/km²时中/大型不适用，须专项论证",
        "pop_source": _load_pop().get("src", ""),
    }
