"""
障碍物引擎 — 航线 × 高塔/桅杆/电力线 邻近性

为何自建数据而非实时查 Overpass：
  Overpass 是免费公共 API，本身限流；LARS 原实现每次评估都实时去打，
  航线一长必然 429/超时（前端只能显示"服务暂不可用"）。且公共镜像
  多数已不可用，实测仅 maps.mail.ru 稳定 —— 把评估结论押在一个
  不受控的第三方免费服务上，对安全工具是不可接受的依赖。
  故一次性抽取入库，与适飞网格同样本地查，查询 <10ms 且永不失败。

数据：obstacles.json（广东 + 四川，OSM power=tower / man_made=mast / power=line）
判定：沿线采样点 → 桶索引查邻近障碍 → 按距离分级
"""
import os, json, math, logging
from array import array
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

BUCKET = 0.05          # 度，约 5.5km —— 障碍物关注半径远小于适飞网格，用更细的桶

# 列式存储（浮点数组），~56 万点内存约 15MB；避免 list-of-dict 的百 MB 开销
_LAT: array = array("d")
_LON: array = array("d")
_T: str = ""
_BUCKETS: Dict[Tuple[int, int], List[int]] = {}
_LOADED = False

CH_CN = {"t": "高压电塔", "m": "桅杆/通信塔", "l": "电力线"}
CH_EN = {"t": "tower", "m": "mast", "l": "line"}


def _load():
    global _LAT, _LON, _T, _BUCKETS, _LOADED
    if _LOADED:
        return
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "obstacles.json")
    try:
        d = json.load(open(path))
        _LAT = array("d", d["lat"])
        _LON = array("d", d["lon"])
        _T = d["t"]
    except Exception as e:
        logger.warning(f"障碍物数据加载失败: {e}")
        _LAT, _LON, _T = array("d"), array("d"), ""
    for i in range(len(_LAT)):
        key = (int(_LAT[i] / BUCKET), int(_LON[i] / BUCKET))
        _BUCKETS.setdefault(key, []).append(i)
    _LOADED = True
    logger.info(f"障碍物加载: {len(_LAT)} 个")


def _hav_km(a_lat, a_lon, b_lat, b_lon) -> float:
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1, math.sqrt(h)))


def _interp(wps: List[Dict], n: int) -> List[Dict]:
    if len(wps) < 2:
        return [{"lat": wps[0]["lat"], "lon": wps[0]["lon"], "dist_km": 0.0}]
    seg = [_hav_km(wps[i]["lat"], wps[i]["lon"], wps[i+1]["lat"], wps[i+1]["lon"])
           for i in range(len(wps) - 1)]
    total = sum(seg) or 1e-9
    step = total / max(n - 1, 1)
    out, si, cum = [], 0, 0.0
    for k in range(n):
        target = k * step
        while si < len(seg) - 1 and cum + seg[si] < target - 1e-9:
            cum += seg[si]; si += 1
        t = 0.0 if seg[si] < 1e-9 else min((target - cum) / seg[si], 1.0)
        out.append({
            "lat": wps[si]["lat"] + t * (wps[si+1]["lat"] - wps[si]["lat"]),
            "lon": wps[si]["lon"] + t * (wps[si+1]["lon"] - wps[si]["lon"]),
            "dist_km": round(target, 2),
        })
    return out


def check_obstacles(params: Dict) -> Dict:
    """输入: waypoints[{lat,lon}], n_samples=100, radius_km=0.5"""
    _load()
    wps = params["waypoints"]
    n = int(params.get("n_samples", 100))
    radius = float(params.get("radius_km", 0.5))

    samples = _interp(wps, n)
    span = max(1, int(radius / 111.0 / BUCKET) + 1)   # 需要扫的桶半径

    hits: Dict[int, float] = {}      # 障碍索引 → 最近距离
    for s in samples:
        bi, bj = int(s["lat"] / BUCKET), int(s["lon"] / BUCKET)
        for di in range(-span, span + 1):
            for dj in range(-span, span + 1):
                for idx in _BUCKETS.get((bi + di, bj + dj), []):
                    d = _hav_km(s["lat"], s["lon"], _LAT[idx], _LON[idx])
                    if d <= radius and (idx not in hits or d < hits[idx]):
                        hits[idx] = d

    items = sorted(({"lat": round(_LAT[i], 5), "lon": round(_LON[i], 5),
                     "type": CH_EN.get(_T[i], "obstacle"), "type_cn": CH_CN.get(_T[i], "障碍物"),
                     "dist_km": round(d, 3)} for i, d in hits.items()),
                   key=lambda x: x["dist_km"])

    n_close = sum(1 for x in items if x["dist_km"] <= 0.1)   # 100m 内
    verdict = "CLEAR" if not items else ("WARN" if n_close else "NOTICE")
    return {
        "verdict": verdict,          # CLEAR 无 / NOTICE 有但不近 / WARN 100m内有
        "total": len(items),
        "n_close": n_close,
        "items": items[:80],
        "radius_km": radius,
        "obstacles_loaded": len(_LAT),
        "note": "障碍物为 OpenStreetMap 开源数据（power=tower / man_made=mast / power=line），"
                "属第三方参考，非官方权威；实际以现场勘察与属地资料为准。",
    }
