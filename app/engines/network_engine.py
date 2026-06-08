"""
航路网络分析引擎 - 真实 NetworkX 实现

基于蔡教授 Critical Links Detection 论文方法:
1. 从航班轨迹构建加权航路网络
2. 计算边介数中心性 (edge betweenness)
3. 运行网络渗流分析 (percolation)
4. 识别关键航段
"""
import networkx as nx
import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def analyze_airway_network(
    flights: List[Dict],
    percolation_threshold: float = 0.3
) -> Dict:
    """
    航路网络分析
    
    参数:
        flights: 航班数据列表（包含轨迹）
        percolation_threshold: 渗流阈值 [0, 1]
    """
    # ═══ 1. 构建加权网络 ═══
    G = nx.DiGraph()
    edge_weights = defaultdict(int)  # (from_wpt, to_wpt) -> 航班计数
    
    for flight in flights:
        track = flight.get('track', [])
        if len(track) < 2:
            continue
        
        # 把轨迹点量化到网格（近似航路点）
        # 0.1度 ≈ 11km，作为航路点的空间分辨率
        waypoints = []
        for pt in track:
            if len(pt) >= 3 and pt[1] and pt[2]:
                wpt = (round(pt[1] * 10) / 10, round(pt[2] * 10) / 10)
                if not waypoints or waypoints[-1] != wpt:
                    waypoints.append(wpt)
        
        # 构建连续的边
        for i in range(len(waypoints) - 1):
            from_wpt = f"{waypoints[i][0]:.1f}N_{waypoints[i][1]:.1f}E"
            to_wpt = f"{waypoints[i+1][0]:.1f}N_{waypoints[i+1][1]:.1f}E"
            edge_weights[(from_wpt, to_wpt)] += 1
    
    # 填充图
    for (u, v), count in edge_weights.items():
        G.add_edge(u, v, weight=count, flight_count=count)
    
    if G.number_of_nodes() == 0:
        logger.warning("网络为空，无法分析")
        return {"error": "insufficient data"}
    
    logger.info(f"构建网络: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")
    
    # ═══ 2. 边介数中心性（真实计算）═══
    # 使用 NetworkX 的 betweenness_centrality
    edge_btw = nx.edge_betweenness_centrality(G, weight='weight', normalized=True)
    node_btw = nx.betweenness_centrality(G, weight='weight', normalized=True)
    
    # 取介数最高的节点
    top_nodes = sorted(node_btw.items(), key=lambda x: -x[1])[:10]
    top_edges = sorted(edge_btw.items(), key=lambda x: -x[1])[:10]
    
    # ═══ 3. 网络渗流分析 ═══
    # 按流量阈值移除边，检查网络连通性变化
    max_weight = max(d['weight'] for _, _, d in G.edges(data=True))
    threshold_weight = max_weight * percolation_threshold
    
    G_filtered = G.copy()
    edges_removed = 0
    for u, v, data in list(G.edges(data=True)):
        if data['weight'] < threshold_weight:
            G_filtered.remove_edge(u, v)
            edges_removed += 1
    
    # 分析渗流后的连通性
    if G_filtered.number_of_edges() > 0:
        largest_cc = max(nx.weakly_connected_components(G_filtered), key=len)
        cc_ratio = len(largest_cc) / G_filtered.number_of_nodes()
    else:
        cc_ratio = 0
    
    # ═══ 4. 关键航段识别 ═══
    # 关键性 = α × 介数中心性 + β × 流量权重 + γ × 渗流影响
    critical_links = []
    max_btw = max(edge_btw.values()) if edge_btw else 1
    
    for (u, v), btw in sorted(edge_btw.items(), key=lambda x: -x[1])[:20]:
        weight = G[u][v]['weight']
        normalized_btw = btw / max_btw
        normalized_weight = weight / max_weight
        
        # 模拟移除该边对网络的影响
        G_test = G.copy()
        G_test.remove_edge(u, v)
        if nx.number_weakly_connected_components(G_test) > nx.number_weakly_connected_components(G):
            percolation_impact = 1.0  # 会分裂网络
        else:
            percolation_impact = 0.3
        
        criticality = 0.4 * normalized_btw + 0.3 * normalized_weight + 0.3 * percolation_impact
        
        critical_links.append({
            "from": u,
            "to": v,
            "betweenness": float(btw),
            "flight_count": int(weight),
            "criticality": float(criticality),
            "fragments_on_removal": percolation_impact > 0.5
        })
    
    # ═══ 5. 返回结果 ═══
    return {
        "network_stats": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": nx.density(G),
            "avg_clustering": nx.average_clustering(G.to_undirected()),
        },
        "top_nodes_by_betweenness": [
            {"id": n, "betweenness": float(b)} for n, b in top_nodes
        ],
        "top_edges_by_betweenness": [
            {"from": u, "to": v, "betweenness": float(b)}
            for (u, v), b in top_edges
        ],
        "percolation": {
            "threshold": percolation_threshold,
            "edges_removed": edges_removed,
            "largest_component_ratio": float(cc_ratio),
            "fragmented": cc_ratio < 0.7,
        },
        "critical_links": critical_links,
    }
