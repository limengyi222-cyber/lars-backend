"""空域复杂度引擎 - Interacting Particle System"""
import numpy as np
from typing import List, Dict

def compute_airspace_complexity(flights: List[Dict], rh_nm=5.0, rv_ft=1000.0, 
                                 look_ahead_sec=600, grid_size=20) -> Dict:
    """真实飞机位置的复杂度计算"""
    if not flights:
        return {"error": "no flight data"}
    
    # 大湾区边界
    lon_min, lon_max = 112.5, 115.0
    lat_min, lat_max = 22.0, 23.5
    
    positions = []
    for f in flights:
        if f.get('latitude') and f.get('longitude'):
            positions.append({
                'lon': f['longitude'],
                'lat': f['latitude'],
                'alt': f.get('altitude_m', 0) or 0,
                'speed': f.get('velocity_ms', 0) or 0,
                'heading': f.get('heading_deg', 0) or 0,
            })
    
    # 构建复杂度网格
    grid = np.zeros((grid_size, grid_size))
    for i in range(grid_size):
        for j in range(grid_size):
            cell_lon = lon_min + (i + 0.5) * (lon_max - lon_min) / grid_size
            cell_lat = lat_min + (j + 0.5) * (lat_max - lat_min) / grid_size
            
            # 计算该单元格附近飞机密度与交互度
            complexity = 0
            for p in positions:
                d_lon = (p['lon'] - cell_lon) * 60 * np.cos(np.radians(cell_lat))
                d_lat = (p['lat'] - cell_lat) * 60
                dist_nm = np.sqrt(d_lon**2 + d_lat**2)
                if dist_nm < rh_nm * 2:
                    complexity += np.exp(-dist_nm / rh_nm)
            grid[i, j] = complexity
    
    max_val = float(grid.max()) if grid.max() > 0 else 1
    mean_val = float(grid.mean())
    bottleneck_count = int((grid > mean_val * 1.5).sum())
    
    return {
        "grid_size": grid_size,
        "grid": grid.tolist(),
        "max_complexity": max_val,
        "mean_complexity": mean_val,
        "bottleneck_zones": bottleneck_count,
        "aircraft_count": len(positions),
        "protection_zone": {"rh_nm": rh_nm, "rv_ft": rv_ft},
        "look_ahead_sec": look_ahead_sec,
    }
