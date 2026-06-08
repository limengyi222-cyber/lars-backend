"""
热点检测引擎 - 真实 K-means++ 聚类

基于蔡教授 CREAM V3 的方法:
1. 对交叉点运行 K-means++ 聚类
2. 计算每个聚类的风险贡献
3. 识别贡献高于平均值的聚类为"热点"
"""
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def detect_hotspots_kmeans(crossings: List[Dict], k: int = 9) -> List[Dict]:
    """
    使用真实的 K-means++ 算法检测碰撞风险热点
    
    参数:
        crossings: 交叉点列表，每个点包含 lon/lat/altitude/crossing_type/risk_weight
        k: 聚类数量
    
    返回:
        热点聚类列表（按风险贡献降序）
    """
    if len(crossings) < k:
        logger.warning(f"交叉点数量 ({len(crossings)}) 少于 k ({k})，调整 k")
        k = max(2, len(crossings) // 10)
    
    if not crossings:
        return []
    
    # 提取坐标矩阵
    X = np.array([[c['lon'], c['lat']] for c in crossings])
    weights = np.array([c.get('risk_weight', 1.0) for c in crossings])
    
    # 使用 scikit-learn 的 K-means++ 初始化
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',  # 这是关键！使用 K-means++ 初始化
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    labels = kmeans.fit_predict(X, sample_weight=weights)
    
    # 计算每个聚类的风险贡献
    hotspots = []
    for i in range(k):
        cluster_mask = labels == i
        cluster_points = X[cluster_mask]
        cluster_weights = weights[cluster_mask]
        
        if len(cluster_points) == 0:
            continue
        
        # 风险贡献 = 交叉点数量 × 平均权重（类型严重度）
        contribution = len(cluster_points) * np.mean(cluster_weights)
        
        # 计算聚类半径（包含 95% 点的距离）
        center = kmeans.cluster_centers_[i]
        distances = np.linalg.norm(cluster_points - center, axis=1)
        radius = np.percentile(distances, 95) if len(distances) > 1 else 0.01
        
        # 计算标准差椭圆（SDE2 - Standard Deviational Ellipse）
        if len(cluster_points) > 2:
            cov = np.cov(cluster_points.T)
            eigenvalues, _ = np.linalg.eig(cov)
            sde2_area = 2 * np.pi * np.sqrt(np.abs(eigenvalues[0] * eigenvalues[1]))
        else:
            sde2_area = np.pi * radius ** 2
        
        hotspots.append({
            "id": i + 1,
            "cx": float(center[0]),
            "cy": float(center[1]),
            "radius": float(radius),
            "points_count": int(len(cluster_points)),
            "contribution": float(contribution),
            "sde2_area": float(sde2_area),
            # 风险严重度：相对于所有聚类的归一化值
            "severity": 0.0,  # 稍后计算
        })
    
    # 归一化严重度
    total_contrib = sum(h['contribution'] for h in hotspots)
    avg_contrib = total_contrib / len(hotspots)
    max_contrib = max(h['contribution'] for h in hotspots)
    
    for h in hotspots:
        h['severity'] = h['contribution'] / max_contrib if max_contrib > 0 else 0
        h['is_hotspot'] = h['contribution'] > avg_contrib  # 超过平均贡献则为热点
    
    # 按贡献降序排列
    hotspots.sort(key=lambda x: x['contribution'], reverse=True)
    
    logger.info(
        f"K-means++ 完成: {k} 个聚类, "
        f"{sum(1 for h in hotspots if h['is_hotspot'])} 个识别为热点"
    )
    
    return hotspots
