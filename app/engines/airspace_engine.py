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
MASK_SIZE   = 0.25         # 覆盖掩膜粗网格 ≈25km

# ── 省份登记表（方案A · 按需分省加载）─────────────────────────────
# 全量并入会撑爆内存（约254万格→内存~400MB，Render免费层512MB必OOM）。
# 故只登记元数据（文件名、粗 bbox、格数），【不预加载】；查询命中哪个省才加载哪个省，
# 并以 LRU 淘汰常驻省份，内存峰值恒定在 _MAX_RESIDENT 个省以内。
# 新增省份：生成其 <省>_grids.json 放入后端根目录，在此加一行 _reg(...) 即可。
#   bbox = [W,S,E,N] 由该省网格实测得出；count 为格数（供前端显示网络规模）。
def _reg(name, file, bbox, count, pin=False):
    return {"name": name, "file": file, "bbox": bbox, "count": count, "pin": pin}

PROVINCES = [
    _reg("广东", "gba_grids.json", [109.677, 20.213, 117.175, 25.520], 133022, pin=True),
    _reg("四川", "sc_grids.json",  [97.347, 26.049, 108.544, 34.316], 237232),
]
_MAX_RESIDENT = 3          # 常驻省份数上限（含 pin 的广东）；超出按 LRU 淘汰。
                           # 实测单省网格约 100MB 量级，3 省峰值 ~300MB，为 Render 512MB 留足余量

# ── 每省独立缓存 ──────────────────────────────────────
# _PROV[name] = {"grids":[...], "buckets":{...}, "mask":set(), "coverage":{...}}
_PROV: Dict[str, Dict] = {}
_LRU: List[str] = []       # 访问顺序，队尾最新


def _load_province(name: str) -> Dict:
    """加载单个省网格（含桶索引与覆盖掩膜）；已在内存则直接返回。"""
    if name in _PROV:
        return _PROV[name]
    meta = next((p for p in PROVINCES if p["name"] == name), None)
    if meta is None:
        return None
    root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    path = os.path.join(root, meta["file"])
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        gs = json.load(f)["grids"]

    buckets: Dict[Tuple, List[int]] = {}
    mask: set = set()
    for idx, g in enumerate(gs):
        min_lon, min_lat, max_lon, max_lat = g
        for bi in range(int(min_lon / BUCKET_SIZE), int(max_lon / BUCKET_SIZE) + 1):
            for bj in range(int(min_lat / BUCKET_SIZE), int(max_lat / BUCKET_SIZE) + 1):
                buckets.setdefault((bi, bj), []).append(idx)
        for mi in range(int(min_lon / MASK_SIZE), int(max_lon / MASK_SIZE) + 1):
            for mj in range(int(min_lat / MASK_SIZE), int(max_lat / MASK_SIZE) + 1):
                mask.add((mi, mj))

    _PROV[name] = {"grids": gs, "buckets": buckets, "mask": mask,
                   "coverage": {"name": name, "bbox": meta["bbox"], "count": len(gs)}}
    _LRU.append(name)
    _evict()
    return _PROV[name]


def _evict():
    """LRU 淘汰：常驻省份数超上限时，释放最久未用的【非 pin】省份。"""
    pinned = {p["name"] for p in PROVINCES if p["pin"]}
    while len([n for n in _LRU if n not in pinned]) > (_MAX_RESIDENT - len(pinned)):
        for i, n in enumerate(_LRU):
            if n not in pinned:
                _LRU.pop(i)
                _PROV.pop(n, None)
                break


def _touch(name: str):
    if name in _LRU:
        _LRU.remove(name); _LRU.append(name)


def _provinces_for_point(lon: float, lat: float) -> List[Dict]:
    """按登记 bbox 找出可能包含该点的省份（省界重叠时可能多个）。"""
    return [p for p in PROVINCES
            if p["bbox"][0] <= lon <= p["bbox"][2] and p["bbox"][1] <= lat <= p["bbox"][3]]


def _load_grids():
    """兼容旧接口：预热 pin 省份（广东），其余按需加载。"""
    for p in PROVINCES:
        if p["pin"]:
            _load_province(p["name"])


def _point_in_grids(lon: float, lat: float) -> str:
    """
    判断点 (lon, lat) 的空域状态：'flyable' | 'restricted' | 'no_data'
    只加载该点 bbox 命中的省份（按需），内存不随省份总数增长。
    """
    cands = _provinces_for_point(lon, lat)
    if not cands:
        return "no_data"                    # 不在任何已登记省份的范围内

    covered = False
    for meta in cands:
        P = _load_province(meta["name"])
        if P is None:
            continue
        _touch(meta["name"])
        # 覆盖掩膜（0.25°膨胀1格）——贴合省界实际形状，避免 bbox 吞掉邻省一角
        mi, mj = int(lon / MASK_SIZE), int(lat / MASK_SIZE)
        in_mask = any((mi + di, mj + dj) in P["mask"] for di in (-1, 0, 1) for dj in (-1, 0, 1))
        if not in_mask:
            continue
        covered = True
        bi, bj = int(lon / BUCKET_SIZE), int(lat / BUCKET_SIZE)
        for idx in P["buckets"].get((bi, bj), []):
            g = P["grids"][idx]
            if g[0] <= lon <= g[2] and g[1] <= lat <= g[3]:
                return "flyable"
    return "restricted" if covered else "no_data"


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
        "grids_loaded":   sum(p["count"] for p in PROVINCES),   # 登记的网格总量（网络规模）
        "provinces":      [{"name": p["name"], "count": p["count"],
                            "resident": p["name"] in _PROV} for p in PROVINCES],
    }
