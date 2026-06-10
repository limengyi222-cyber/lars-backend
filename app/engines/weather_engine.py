"""
气象引擎 — OpenWeatherMap 实时数据
获取大湾区气象并转换为 CREAM 参数建议

env: OWM_API_KEY (从 openweathermap.org 免费获取，1000次/天)
"""
import httpx
import os
import math
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

OWM_KEY  = os.environ.get("OWM_API_KEY", "")
GBA_LAT  = 23.13   # 广州
GBA_LON  = 113.26

# 无 API key 时的演示数据
_DEMO_WEATHER = {
    "wind_speed_ms": 4.2,
    "wind_speed_kt": 8.2,
    "wind_dir_deg": 135,
    "visibility_m": 8000,
    "visibility_km": 8.0,
    "pressure_hpa": 1012,
    "temp_c": 27,
    "humidity_pct": 72,
    "condition": "多云",
    "condition_code": 803,
    "location": "广州 / 大湾区",
    "source": "demo",
    "demo": True,
}


async def fetch_gba_weather(route_bearing_deg: float = None) -> dict:
    """
    获取大湾区实时气象 + CREAM 参数建议

    route_bearing_deg: 航路方位角（0=正北，90=正东），用于计算侧风分量
    """
    if not OWM_KEY:
        logger.warning("OWM_API_KEY 未配置，使用演示气象数据")
        result = dict(_DEMO_WEATHER)
        _append_crew_suggestions(result, route_bearing_deg)
        return result

    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={GBA_LAT}&lon={GBA_LON}"
        f"&appid={OWM_KEY}&units=metric&lang=zh_cn"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
        if resp.status_code != 200:
            logger.warning(f"OWM API 返回 {resp.status_code}，使用演示数据")
            result = dict(_DEMO_WEATHER)
            _append_crew_suggestions(result, route_bearing_deg)
            return result

        d = resp.json()
        wind      = d.get("wind", {})
        main_data = d.get("main", {})
        weather   = d.get("weather", [{}])[0]

        wind_speed_ms = float(wind.get("speed", 0) or 0)
        wind_dir_deg  = float(wind.get("deg",   0) or 0)
        visibility_m  = int(d.get("visibility", 10000) or 10000)

        result = {
            "wind_speed_ms":  round(wind_speed_ms, 1),
            "wind_speed_kt":  round(wind_speed_ms * 1.944, 1),
            "wind_dir_deg":   round(wind_dir_deg),
            "wind_gust_kt":   round(float(wind.get("gust", 0) or 0) * 1.944, 1),
            "visibility_m":   visibility_m,
            "visibility_km":  round(visibility_m / 1000, 1),
            "pressure_hpa":   int(main_data.get("pressure", 1013) or 1013),
            "temp_c":         round(float(main_data.get("temp", 20) or 20), 1),
            "humidity_pct":   int(main_data.get("humidity", 60) or 60),
            "condition":      weather.get("description", "晴"),
            "condition_code": weather.get("id", 800),
            "location":       "广州 / 大湾区",
            "source":         "OpenWeatherMap",
            "demo":           False,
            "fetched_at":     datetime.now(timezone.utc).isoformat(),
        }

        _append_crew_suggestions(result, route_bearing_deg)
        return result

    except Exception as e:
        logger.warning(f"气象获取失败: {e}，使用演示数据")
        result = dict(_DEMO_WEATHER)
        _append_crew_suggestions(result, route_bearing_deg)
        return result


def _append_crew_suggestions(result: dict, route_bearing_deg):
    """根据气象数据生成 CREAM 参数建议"""
    ws_ms  = result["wind_speed_ms"]
    vis_m  = result["visibility_m"]
    ws_kt  = result["wind_speed_kt"]

    # 侧风分量（如果提供了航路方位角）
    if route_bearing_deg is not None:
        angle_rad = math.radians(result["wind_dir_deg"] - route_bearing_deg)
        xw_ms = abs(ws_ms * math.sin(angle_rad))
        hw_ms = ws_ms * math.cos(angle_rad)
        result["crosswind_ms"] = round(xw_ms, 1)
        result["crosswind_kt"] = round(xw_ms * 1.944, 1)
        result["headwind_ms"]  = round(hw_ms, 1)
        vy_suggest = max(3.0, round(xw_ms * 1.944, 1))
    else:
        # 无方位角时，用风速估计侧风
        vy_suggest = max(3.0, round(ws_kt * 0.6, 1))   # 假设约 60% 为侧风

    # RNP 建议
    if vis_m < 1500:
        rnp_suggest, rnp_reason = 1.0, "能见度<1.5km，建议提高 RNP"
        wx_level = "danger"
    elif vis_m < 5000:
        rnp_suggest, rnp_reason = 0.3, "能见度<5km，适度降级 RNP"
        wx_level = "caution"
    else:
        rnp_suggest, rnp_reason = 0.1, "能见度良好，标准 RNP"
        wx_level = "normal"

    # 风速风险等级
    if ws_kt >= 25:
        wx_level = "danger"
    elif ws_kt >= 15:
        wx_level = max(wx_level, "caution") if wx_level != "danger" else "danger"

    result["suggestions"] = {
        "Vy_kt":     vy_suggest,
        "RNP_nm":    rnp_suggest,
        "rnp_reason": rnp_reason,
        "wx_level":  wx_level,
    }
