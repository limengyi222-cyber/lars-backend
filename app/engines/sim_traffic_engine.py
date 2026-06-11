"""
仿真低空流量生成引擎

背景：低空无人机不装 ADS-B 应答机，adsb.fi 抓到的是高空有人机航班，
与"低空航路风险评估"的对象错位。本引擎基于大湾区公开的物流/eVTOL
试点走廊构造仿真低空流量，供热点检测、网络分析、复杂度三个模块使用。

构造依据（公开资料）：
  - 美团/顺丰丰翼无人机深圳常态化配送运营报道
  - 丰翼科技深圳—珠海/中山跨珠江口物流航线报道
  - 亿航 EH216-S 广州载人观光试点
  - 大湾区城市群地理位置（节点坐标为城市中心近似值）

特性：
  - 走廊流量符合高斯扰动的强度参数（近似泊松到达）
  - 每架航班带 track 轨迹（网络分析引擎需要轨迹才能建图）
  - 种子按 5 分钟时间窗取整 → 同一窗口内结果可复现，跨窗口自然变化
"""
import math
import random
import time
from typing import List, Dict, Optional

# ── 大湾区低空走廊网络 ────────────────────────────────────────────
# 节点：城市中心近似坐标 (lat, lon)
NODES = {
    "GUANG":     (23.13, 113.26),
    "SZNB":      (22.54, 114.06),
    "ZHUHAI":    (22.27, 113.57),
    "DONGG":     (23.02, 113.75),
    "FOSHAN":    (23.02, 113.11),
    "ZHONGSHAN": (22.52, 113.39),
    "HUIZHOU":   (23.11, 114.42),
    "JIANGMEN":  (22.58, 113.08),
}

# 走廊：(起点, 终点, 流量强度[同时在航架次期望])
CORRIDORS = [
    ("GUANG", "SZNB",      14),  # 广深干线（最繁忙）
    ("SZNB",  "DONGG",     11),
    ("GUANG", "FOSHAN",    10),
    ("GUANG", "DONGG",      9),
    ("SZNB",  "ZHUHAI",     8),  # 跨珠江口物流（丰翼原型）
    ("GUANG", "ZHUHAI",     6),
    ("ZHUHAI","ZHONGSHAN",  6),
    ("FOSHAN","ZHONGSHAN",  5),
    ("SZNB",  "HUIZHOU",    5),
    ("DONGG", "HUIZHOU",    5),
    ("ZHONGSHAN","JIANGMEN",4),
    ("FOSHAN","JIANGMEN",   4),
]

# 城市即时配送圈：(名称, 中心lat, 中心lon, 半径[度], 流量强度)
CLUSTERS = [
    ("SZ-DELIV", 22.54, 114.03, 0.12, 18),  # 深圳（美团原型，最密集）
    ("GZ-DELIV", 23.12, 113.30, 0.12, 14),  # 广州
    ("DG-DELIV", 23.02, 113.74, 0.08,  8),  # 东莞
]

# 低空典型飞行高度层 (m AGL) 与速度范围
CORRIDOR_ALTS_M = [120, 150, 200, 250]
CLUSTER_ALTS_M  = [60, 80, 100, 120]


def _bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """近似航向角（小范围平面近似足够）"""
    return math.degrees(math.atan2(lon2 - lon1, lat2 - lat1)) % 360


def generate_sim_traffic(density: float = 1.0, seed: Optional[int] = None) -> List[Dict]:
    """
    生成仿真低空流量

    参数:
      density: 流量密度系数（1.0 = 基准，2.0 = 双倍密度压测）
      seed:    随机种子；None 时按 5 分钟时间窗取整（窗口内可复现）

    返回: 与 adsb.fi 兼容的航班列表，额外带 altitude_m / velocity_ms / track / sim 字段
    """
    if seed is None:
        seed = int(time.time() // 300)
    rng = random.Random(seed)
    flights: List[Dict] = []
    fid = 0

    # ── 走廊流量 ──────────────────────────────────────
    for a, b, inten in CORRIDORS:
        (la1, lo1), (la2, lo2) = NODES[a], NODES[b]
        n = max(1, int(rng.gauss(inten * density, inten * 0.25)))
        brg_fwd = _bearing_deg(la1, lo1, la2, lo2)

        for _ in range(n):
            t = rng.random()                    # 走廊上的位置比例
            fwd = rng.random() < 0.5            # 双向运行
            jit_lat = rng.gauss(0, 0.008)       # 侧向抖动 ≈ ±900m 走廊宽
            jit_lon = rng.gauss(0, 0.008)
            lat = la1 + t * (la2 - la1) + jit_lat
            lon = lo1 + t * (lo2 - lo1) + jit_lon
            hdg = brg_fwd if fwd else (brg_fwd + 180) % 360
            alt_m = rng.choice(CORRIDOR_ALTS_M)
            spd_kmh = rng.uniform(50, 110)

            # 轨迹：沿走廊前后各 3 个点（网络分析建图需要）
            track = []
            step = 0.08 if fwd else -0.08
            for k in range(-3, 4):
                tk = min(1.0, max(0.0, t + k * step))
                track.append([0,
                              round(la1 + tk * (la2 - la1) + jit_lat, 5),
                              round(lo1 + tk * (lo2 - lo1) + jit_lon, 5),
                              alt_m])

            fid += 1
            flights.append({
                "icao24":       f"sim{fid:04d}",
                "callsign":     f"UAV{1000 + fid}",
                "longitude":    round(lon, 5),
                "latitude":     round(lat, 5),
                "altitude":     round(alt_m * 3.28084),   # ft（兼容 adsb 格式）
                "altitude_m":   alt_m,
                "on_ground":    False,
                "velocity":     round(spd_kmh / 1.852, 1),  # knots
                "velocity_ms":  round(spd_kmh / 3.6, 1),
                "heading_deg":  round(hdg),
                "vertical_rate": 0,
                "track":        track,
                "corridor":     f"{a}-{b}",
                "sim":          True,
            })

    # ── 城市配送圈流量（短途、多向、低高度）────────────────
    for name, clat, clon, rad, inten in CLUSTERS:
        n = max(1, int(rng.gauss(inten * density, inten * 0.2)))
        for _ in range(n):
            ang = rng.uniform(0, 2 * math.pi)
            r = rad * math.sqrt(rng.random())   # 面积均匀分布
            lat = clat + r * math.cos(ang)
            lon = clon + r * math.sin(ang)
            hdg = rng.uniform(0, 360)
            alt_m = rng.choice(CLUSTER_ALTS_M)
            spd_kmh = rng.uniform(30, 65)

            # 轨迹：从配送中心放射状往返
            out_lat = clat + rad * 1.1 * math.cos(ang)
            out_lon = clon + rad * 1.1 * math.sin(ang)
            track = [
                [0, round(clat, 5),    round(clon, 5),    alt_m],
                [0, round((clat+lat)/2, 5), round((clon+lon)/2, 5), alt_m],
                [0, round(lat, 5),     round(lon, 5),     alt_m],
                [0, round(out_lat, 5), round(out_lon, 5), alt_m],
            ]

            fid += 1
            flights.append({
                "icao24":       f"sim{fid:04d}",
                "callsign":     f"DLV{1000 + fid}",
                "longitude":    round(lon, 5),
                "latitude":     round(lat, 5),
                "altitude":     round(alt_m * 3.28084),
                "altitude_m":   alt_m,
                "on_ground":    False,
                "velocity":     round(spd_kmh / 1.852, 1),
                "velocity_ms":  round(spd_kmh / 3.6, 1),
                "heading_deg":  round(hdg),
                "vertical_rate": 0,
                "track":        track,
                "corridor":     name,
                "sim":          True,
            })

    return flights
