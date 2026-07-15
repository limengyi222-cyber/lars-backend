"""
空域穿越检查引擎
基于各省适飞网格数据（广东 133,022 + 四川 237,232 个适飞矩形）

检查逻辑：
  - 网格内 = 适飞空域（绿）
  - 网格外但在【已收录省份覆盖范围】内 = 非适飞 / 需申请（红）
  - 超出所有已收录省份 = 无数据，需人工核查（灰）

覆盖范围由各省网格数据【实测得出】，不再写死常量：
  原实现硬编码广东框 (112.0-115.5, 21.5-24.5)，而 2026 V3 数据实际跨
  (109.677-117.175, 20.213-25.520) —— 导致 58,949 个适飞格(44.3%)被误判为
  "超出覆盖范围/需人工核查"（湛江、潮汕、韶关等全中）。改为实测后修复。

空间索引：0.1° 桶分组，查询复杂度 O(1) vs 原始 O(N)
"""
import json
import os
from typing import List, Dict, Tuple

# ── 常量 ──────────────────────────────────────────────
BUCKET_SIZE = 0.1          # 度，约 11km

# 已收录省份网格文件（新增省份只需在此登记）
PROVINCE_FILES = [
    ("广东", "gba_grids.json"),
    ("四川", "sc_grids.json"),
]

# ── 单例缓存 ──────────────────────────────────────────
_GRIDS: List[List[float]] = []           # [[minLon,minLat,maxLon,maxLat], ...]
_BUCKETS: Dict[Tuple, List[int]] = {}    # (bi, bj) -> [grid_idx, ...]
_COVERAGE: List[Dict] = []               # [{name, bbox:[W,S,E,N], count}] 各省实测覆盖范围
# 覆盖掩膜：粗网格(0.25°≈25km)中"有适飞格存在"的单元。用它替代省 bbox 判覆盖——
# 省界不是矩形：四川 bbox 会吞掉贵州/云南/重庆一角，导致无数据地区被谎报为"需申请"。
MASK_SIZE = 0.25
_MASK: set = set()
_LOADED = False


def _load_grids():
    global _GRIDS, _BUCKETS, _COVERAGE, _MASK, _LOADED
    if _LOADED:
        return

    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    for prov, fname in PROVINCE_FILES:
        path = os.path.join(root, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            data = json.load(f)
        gs = data["grids"]
        base = len(_GRIDS)
        _GRIDS.extend(gs)

        # 覆盖范围由该省网格实测得出（不写死常量）
        _COVERAGE.append({
            "name": prov,
            "bbox": [min(g[0] for g in gs), min(g[1] for g in gs),
                     max(g[2] for g in gs), max(g[3] for g in gs)],
            "count": len(gs),
        })

        # 构建空间桶索引
        for k, g in enumerate(gs):
            idx = base + k
            min_lon, min_lat, max_lon, max_lat = g
            for bi in range(int(min_lon / BUCKET_SIZE), int(max_lon / BUCKET_SIZE) + 1):
                for bj in range(int(min_lat / BUCKET_SIZE), int(max_lat / BUCKET_SIZE) + 1):
                    _BUCKETS.setdefault((bi, bj), []).append(idx)
            # 覆盖掩膜
            for mi in range(int(min_lon / MASK_SIZE), int(max_lon / MASK_SIZE) + 1):
                for mj in range(int(min_lat / MASK_SIZE), int(max_lat / MASK_SIZE) + 1):
                    _MASK.add((mi, mj))

    _LOADED = True


def _is_covered(lon: float, lat: float) -> bool:
    """
    是否落在已收录省份的实际数据覆盖内。
    以适飞格占用掩膜(0.25°)判定并向外膨胀1格(≈25-50km容差)，贴合省界实际形状；
    省内成片非适飞区(城区/机场等，通常远小于该容差)仍算已覆盖 → 判"需申请"；
    完全没有数据的省份(如贵州)则如实返回未覆盖 → 判"无数据/需人工核查"。
    """
    mi, mj = int(lon / MASK_SIZE), int(lat / MASK_SIZE)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (mi + di, mj + dj) in _MASK:
                return True
    return False


def _covering_province(lon: float, lat: float) -> str:
    """点落在哪个已收录省份的 bbox 内（仅用于标注省名）"""
    for c in _COVERAGE:
        w, s_, e, n = c["bbox"]
        if w <= lon <= e and s_ <= lat <= n:
            return c["name"]
    return ""


def _point_in_grids(lon: float, lat: float) -> str:
    """
    判断点 (lon, lat) 的空域状态
    返回: 'flyable' | 'restricted' | 'no_data'
    """
    # 超出已收录省份的实际数据覆盖
    if not _is_covered(lon, lat):
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

    # 覆盖率（落在已收录省份范围内的点占比）
    in_gd = counts["flyable"] + counts["restricted"]
    coverage_pct = round(in_gd / total * 100, 1) if total > 0 else 0.0

    restricted_pct = round(counts["restricted"] / total * 100, 1) if total > 0 else 0.0

    # 评级
    if counts["restricted"] == 0 and counts["no_data"] == 0:
        verdict = "PASS"
    elif counts["restricted"] == 0:
        verdict = "WARNING"   # 有超出已收录省份范围的段，需人工核查
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
        "provinces":      [{"name": c["name"], "count": c["count"]} for c in _COVERAGE],
    }
