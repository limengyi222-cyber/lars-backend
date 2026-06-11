"""航迹交叉点检测 - 基于几何算法"""
import numpy as np
from typing import List, Dict

def detect_crossings(flights: List[Dict]) -> List[Dict]:
    """检测航班之间的交叉点"""
    crossings = []
    
    # 用每架航班的当前位置作为检测点
    # （注：下方交叉判定基于点邻近实现，线段相交未实现；
    #   带轨迹的航班同样取当前位置，否则会被静默跳过）
    segments = []
    for f in flights:
        if f.get('latitude') and f.get('longitude'):
            segments.append({
                'icao24': f.get('icao24'),
                'lon': f['longitude'],
                'lat': f['latitude'],
                'alt': f.get('altitude_m', 0) or 0,
            })
    
    # 简化：如果是实时数据（只有点），用空间邻近性近似交叉点
    for i, s1 in enumerate(segments):
        for s2 in segments[i+1:]:
            if 'lon' in s1 and 'lon' in s2:
                d_lon = (s1['lon'] - s2['lon']) * 60 * np.cos(np.radians(s1['lat']))
                d_lat = (s1['lat'] - s2['lat']) * 60
                dist_nm = np.sqrt(d_lon**2 + d_lat**2)
                if dist_nm < 10:
                    crossings.append({
                        'lon': (s1['lon'] + s2['lon']) / 2,
                        'lat': (s1['lat'] + s2['lat']) / 2,
                        'altitude_ft': (s1['alt'] + s2['alt']) / 2 * 3.28084,
                        'crossing_type': 'angle',
                        'risk_weight': max(0.1, 1 - dist_nm / 10),
                    })
    
    return crossings
