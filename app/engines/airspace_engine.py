"""
空域穿越检查引擎
基于 gba_grids.json 适飞网格数据（广东省 46,331 个适飞矩形）

检查逻辑：
  - 网格内 = 适飞空域（绿）
  - 网格外但在广东覆盖范围内 = 非适飞 / 需申请（红）
  - 超出广东覆盖范围 = 无数据，需人工核查（灰）

空间索引：0.1° 桶分组，查询复杂度 O(1) vs 原始 O(N=46331)
"""
import json
import os
from typing import List, Dict, Tuple

# ── 常量 ──────────────────────────────────────────────
BUCKET_SIZE = 0.1          # 度，约 11km
# 广东覆盖范围（来自网格数据实测）
GD_LON_MIN, GD_LON_MAX = 112.0, 115.5
GD_LAT_MIN, GD_LAT_MAX = 21.5,  24.5

# ── 单例缓存 ──────────────────────────────────────────
_GRIDS: List[List[float]] = []           # [[minLon,minLat,maxLon,maxLat], ...]
_BUCKETS: Dict[Tuple, List[int]] = {}    # (bi, bj) -> [grid_idx, ...]
_LOADED = False


def _load_grids():
    global _GRIDS, _BUCKETS, _LOADED
    if _LOADED:
        return

    grid_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "gba_grids.json"
    )
    with open(grid_path, "r") as f:
        data = json.load(f)

    _GRIDS = data["grids"]

    # 构建空间桶索引
    for idx, g in enumerate(_GRIDS):
        min_lon, min_lat, max_lon, max_lat = g
        # 格子跨越的所有桶（通常只有1个，极少2个）
        bi_start = int(min_lon / BUCKET_SIZE)
        bi_end   = int(max_lon / BUCKET_SIZE)
        bj_start = int(min_lat / BUCKET_SIZE)
        bj_end   = int(max_lat / BUCKET_SIZE)
        for bi in range(bi_start, bi_end + 1):
            for bj in range(bj_start, bj_end + 1):
                key = (bi, bj)
                if key not in _BUCKETS:
                    _BUCKETS[key] = []
                _BUCKETS[key].append(idx)

    _LOADED = True


def _point_in_grids(lon: float, lat: float) -> str:
    """
    判断点 (lon, lat) 的空域状态
    返回: 'flyable' | 'restricted' | 'no_data'
    """
    # 超出广东覆盖范围
    if not (GD_LON_MIN <= lon <= GD_LON_MAX and GD_LAT_MIN <= lat <= GD_LAT_MAX):
        return "no_data"

    bi = int(lon / BUCKET_SIZE)
    bj = int(lat / BUCKET_SIZE)
    candidates = _BUCKETS.get((bi, bj), [])

    for idx in candidates:
        g = _GRIDS[idx]
        if g[0] <= lon <= g[2] and g[1] <= lat <= g[3]:
            return "flyable"

    return "restricted"


# ── 主函数 ────────────────────────────────────────────

def check_route_airspace(params: Dict) -> Dict:
    """
    航线空域穿越检查

    输入:
      waypoints   : [{lat, lon}, ...]   至少 2 个点
      n_samples   : int = 80           沿线采样点数

    输出:
      profile     : 每采样点的空域状态
      summary     : 各状态统计
      segments    : 连续受限段列表
      verdict     : 'PASS' | 'WARNING' | 'RESTRICTED'
      coverage    : 广东范围内采样点占比
    """
    _load_grids()

    from .terrain_engine import _interpolate_route
    waypoints = params["waypoints"]
    n_samples = int(params.get("n_samples", 80))

    samples = _interpolate_route(waypoints, n_samples)

    profile = []
    counts = {"flyable": 0, "restricted": 0, "no_data": 0}

    for s in samples:
        status = _point_in_grids(s["lon"], s["lat"])
        counts[status] += 1
        profile.append({
            "dist_km": s["dist_km"],
            "lat":     round(s["lat"], 6),
            "lon":     round(s["lon"], 6),
            "status":  status,
        })

    total = len(profile)

    # 识别连续受限段
    segments = []
    in_seg = False
    seg_start = None

    for i, pt in enumerate(profile):
        bad = pt["status"] in ("restricted",)
        if bad and not in_seg:
            in_seg = True
            seg_start = pt
        elif not bad and in_seg:
            in_seg = False
            segments.append({
                "from_km":  seg_start["dist_km"],
                "to_km":    profile[i - 1]["dist_km"],
                "length_km": round(profile[i - 1]["dist_km"] - seg_start["dist_km"], 2),
                "status":   "restricted",
            })
    if in_seg:
        segments.append({
            "from_km":  seg_start["dist_km"],
            "to_km":    profile[-1]["dist_km"],
            "length_km": round(profile[-1]["dist_km"] - seg_start["dist_km"], 2),
            "status":   "restricted",
        })

    # 无数据段（超出广东范围）
    nd_segs = []
    in_nd = False
    nd_start = None
    for i, pt in enumerate(profile):
        if pt["status"] == "no_data" and not in_nd:
            in_nd = True
            nd_start = pt
        elif pt["status"] != "no_data" and in_nd:
            in_nd = False
            nd_segs.append({
                "from_km": nd_start["dist_km"],
                "to_km":   profile[i - 1]["dist_km"],
                "length_km": round(profile[i - 1]["dist_km"] - nd_start["dist_km"], 2),
                "status": "no_data",
            })
    if in_nd:
        nd_segs.append({
            "from_km": nd_start["dist_km"],
            "to_km":   profile[-1]["dist_km"],
            "length_km": round(profile[-1]["dist_km"] - nd_start["dist_km"], 2),
            "status": "no_data",
        })

    all_segments = sorted(segments + nd_segs, key=lambda x: x["from_km"])

    # 覆盖率（广东范围内的点占比）
    in_gd = counts["flyable"] + counts["restricted"]
    coverage_pct = round(in_gd / total * 100, 1) if total > 0 else 0.0

    restricted_pct = round(counts["restricted"] / total * 100, 1) if total > 0 else 0.0

    # 评级
    if counts["restricted"] == 0 and counts["no_data"] == 0:
        verdict = "PASS"
    elif counts["restricted"] == 0:
        verdict = "WARNING"   # 有超出广东范围的段，需人工核查
    else:
        verdict = "RESTRICTED"

    return {
        "profile":        profile,
        "summary":        counts,
        "segments":       all_segments,
        "restricted_pct": restricted_pct,
        "coverage_pct":   coverage_pct,
        "verdict":        verdict,
        "total_samples":  total,
        "grids_loaded":   len(_GRIDS),
    }
