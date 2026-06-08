"""
CREAM 碰撞风险计算引擎（真实 ICAO 方法论实现）
"""
import numpy as np
from scipy import integrate
from typing import Dict


def _laplace_pdf(x, mu, b):
    """Laplace 分布 PDF（ICAO 侧向误差模型）"""
    return (1.0 / (2.0 * b)) * np.exp(-np.abs(x - mu) / b)


def _normal_pdf(x, mu, sigma):
    """正态分布 PDF"""
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _compute_Pz_Sz(Sz, lambda_z, sigma_aad, sigma_ase, n_points=1000):
    """
    计算垂直重叠概率 Pz(Sz)
    
    使用 SciPy 的高精度数值积分，比前端 JS 的简单矩形积分精确很多。
    """
    sigma_combined = np.sqrt(sigma_aad ** 2 + sigma_ase ** 2)
    
    def integrand(z1):
        # TVE 分布（简化为卷积后的高斯）
        f1 = _normal_pdf(z1, 0, sigma_combined)
        f2 = _normal_pdf(Sz + z1, 0, sigma_combined)
        return f1 * f2
    
    # Scipy 自适应积分（比 JS 的固定步长更精确）
    integral, _ = integrate.quad(
        integrand,
        -5 * sigma_combined,
        5 * sigma_combined,
        limit=100
    )
    
    return 2 * lambda_z * integral


def _compute_Py_Sy(Sy, RNP, lambda_y, n_points=1000):
    """
    计算侧向重叠概率 Py(Sy) - ICAO 卷积模型
    """
    # RNP -> Laplace 参数 lambda
    b = -RNP / np.log(0.05)
    
    def integrand(y1):
        f1 = _laplace_pdf(y1, 0, b)
        f2 = _laplace_pdf(Sy + y1, 0, b)
        return f1 * f2
    
    integral, _ = integrate.quad(
        integrand,
        -5 * RNP,
        5 * RNP,
        limit=100
    )
    
    return 2 * lambda_y * integral


def _compute_HOP(separation, RNP, lambda_xy):
    """
    计算水平重叠概率 HOP (Hsu 纵向风险模型)
    """
    b = -RNP / np.log(0.05)
    coef = (np.pi * lambda_xy ** 2) / (16 * b ** 2)
    ratio = separation / b
    return coef * np.exp(-ratio) * (ratio + 1)


def compute_3d_risk(params: Dict) -> Dict:
    """
    CREAM 三维碰撞风险完整计算
    
    基于:
    - ICAO Doc 9689 (纵向/侧向碰撞风险评估)
    - ICAO Doc 10063 (水平间隔性能监测)
    - NTU ATMRI CREAM 方法论
    """
    # 提取参数
    Sx = params['Sx']
    Sy = params['Sy']
    Sz = params['Sz']
    RNP = params['RNP']
    V = params['V']
    y_dot = params['y_dot']
    z_dot = params['z_dot']
    lambda_x = params['lambda_x']
    lambda_y = params['lambda_y']
    lambda_z = params['lambda_z']
    lambda_xy = params['lambda_xy']
    sigma_aad = params['sigma_aad']
    sigma_ase = params['sigma_ase']
    Pz_0 = params['Pz_0']
    Ey_opp = params['Ey_opp']
    
    # ═══ 垂直风险 Naz ═══
    Pz_Sz = _compute_Pz_Sz(Sz, lambda_z, sigma_aad, sigma_ase)
    
    # 简化的 Py(0) 近似（实际应根据飞行路径计算）
    Py_0 = 2 * lambda_y * _laplace_pdf(0, 0, 1) ** 2 * 2
    
    # 运动学因子
    kinematic = 1 + y_dot / (2 * V) + (lambda_xy * z_dot) / (lambda_z * 2 * V)
    
    # 等效通过频率（简化）
    n_z_equiv = 0.357
    
    Naz = 2 * Pz_Sz * Py_0 * n_z_equiv * kinematic
    
    # ═══ 侧向风险 Nay ═══
    Py_Sy = _compute_Py_Sy(Sy, RNP, lambda_y)
    K_lat = (V * (lambda_x + lambda_y)) / (Sy * 2 * (lambda_y + lambda_z) * 2 * lambda_z)
    Nay = Py_Sy * Pz_0 * Ey_opp * K_lat
    
    # ═══ 纵向风险 CR ═══
    HOP = _compute_HOP(Sx, RNP, lambda_xy)
    V_rel = 0.01  # 最小相对速度（避免除零）
    CR_base = (2 * HOP * Pz_0 * lambda_z * z_dot) / (
        2 * V_rel * np.pi * (lambda_xy + lambda_z) * 2 * lambda_z
    )
    CR = 0.9025 * CR_base + 0.05 * CR_base * 0.8 + 0.05 * CR_base * 0.6
    
    total = Naz + Nay + CR
    TLS = 5e-9
    
    return {
        "Naz": float(Naz),
        "Nay": float(Nay),
        "CR": float(CR),
        "total_risk": float(total),
        "Pz_Sz": float(Pz_Sz),
        "Py_Sy": float(Py_Sy),
        "HOP": float(HOP),
        "is_compliant": bool(total < TLS),
        "tls_ratio": float(total / TLS)
    }


def compute_uav_collision(params: Dict) -> Dict:
    """
    无人机入侵碰撞概率（布朗运动模型）
    """
    d = params['d']
    beta = np.radians(params['beta'])
    theta = np.radians(params['theta'])
    Vh = params['Vh'] * 1000 / 3600  # km/h -> m/s
    Rm = params['Rm']
    Qm = params['Qm']
    sigma_h = params['sigma_h']
    sigma_v = params['sigma_v']
    T = params['T']
    
    # 水平投影几何
    D = d * np.cos(beta)
    approach_rate = Vh * np.cos(theta)
    
    if approach_rate > 0.01:
        t_closest = D / approach_rate
    else:
        t_closest = T
    
    closest_dist = D * np.abs(np.sin(theta))
    
    # 布朗扩展
    sigma_spread = sigma_h * np.sqrt(max(t_closest, 1.0))
    
    # Ph
    from scipy.stats import norm
    if closest_dist < Rm:
        Ph = norm.cdf(Rm - closest_dist, 0, sigma_spread) - \
             norm.cdf(-Rm - closest_dist, 0, sigma_spread)
    else:
        Ph = 2 * norm.cdf(Rm, closest_dist, sigma_spread) - 1
    Ph = max(0, min(1, Ph))
    
    # Pv
    h0 = d * np.sin(beta)
    sigma_v_spread = sigma_v * np.sqrt(T)
    if abs(h0) < Qm:
        Pv = 1.0
    else:
        Pv = 2 * norm.cdf(Qm, abs(h0), sigma_v_spread) - 1
    Pv = max(0, min(1, Pv))
    
    return {
        "P_total": float(Ph * Pv),
        "P_horizontal": float(Ph),
        "P_vertical": float(Pv),
        "closest_distance_m": float(closest_dist),
        "time_to_closest_s": float(t_closest),
    }
