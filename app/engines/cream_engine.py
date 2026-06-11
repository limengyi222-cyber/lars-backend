"""
CREAM 碰撞风险计算引擎 — 修正版
基于 ICAO Doc 9689 + NTU ATMRI CREAM 方法论

修正记录 (v2.1):
  1. TVE 分布: Laplace(ASE) ⊛ Normal(AAD) 解析卷积，取代简化高斯
  2. Py(0): λy/(2b) 解析解，b = -RNP/ln(0.05)，不再硬编码 b=1
  3. 运动学因子: 改为乘法结构 (ICAO Doc 9689 A.3.3)
  4. CR 纵向风险: 区分同向/对向 V_rel，不再使用 0.01 硬编码
  5. n_z_equiv: 从流量密度参数推导，不再固定 0.357
  6. HOP: 修正系数 π/4（原 π/16 有误），ratio = 2s/b
  7. K_lat: 修正侧向运动学因子公式
  8. Pz(0): 全程从 TVE 模型计算，不依赖传入常数
"""
import numpy as np
from scipy import integrate
from scipy.special import erfc
from typing import Dict


# ── 基础分布函数 ──────────────────────────────────────────────────────────────

def _laplace_pdf(x: float, mu: float, b: float) -> float:
    """Laplace 分布 PDF — f(x) = (1/2b) exp(-|x-μ|/b)"""
    return (1.0 / (2.0 * b)) * np.exp(-np.abs(x - mu) / b)


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    """正态分布 PDF"""
    return (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _tve_pdf(z: float, sigma_aad: float, b_ase: float) -> float:
    """
    TVE (总垂直误差) 解析 PDF

    TVE = ASE + AAD
      ASE ~ Laplace(0, b_ase)   高度表系统误差（双指数分布）
      AAD ~ Normal(0, sigma_aad) 实际高度偏差（正态分布）

    解析卷积结果（Reed & Jorgensen 2004 正态-拉普拉斯分布）:
      f(z) = exp(σ²/2b²)/(4b) × [exp(-z/b)·erfc(-α) + exp(z/b)·erfc(β)]
    其中:
      α = (z/σ - σ/b) / √2     β = (z/σ + σ/b) / √2

    数值稳定性：
      - 当 z >> 0：exp(-z/b) 大，但 erfc(-α) = erfc(大正数) ≈ 0，相互抵消
      - 当 z << 0：exp(z/b) 大，但 erfc(β) = erfc(大正数) ≈ 0，相互抵消
      → 整个参数范围均无溢出风险
    """
    # 退化情形
    if b_ase < 1e-10:
        return _normal_pdf(z, 0.0, sigma_aad)
    if sigma_aad < 1e-10:
        return _laplace_pdf(z, 0.0, b_ase)

    s, b = float(sigma_aad), float(b_ase)

    alpha = (z / s - s / b) / np.sqrt(2.0)   # (z - σ²/b) / (σ√2)
    beta  = (z / s + s / b) / np.sqrt(2.0)   # (z + σ²/b) / (σ√2)

    # 前置因子：exp(σ²/2b²) / (4b)
    log_scale = s ** 2 / (2.0 * b ** 2)      # 典型值 ~1–5，无溢出

    term1 = np.exp(log_scale - z / b) * float(erfc(-alpha))
    term2 = np.exp(log_scale + z / b) * float(erfc(beta))

    return float(np.clip((term1 + term2) / (4.0 * b), 0.0, None))


# ── 重叠概率 ─────────────────────────────────────────────────────────────────

def _compute_Pz_Sz(Sz: float, lambda_z: float,
                   sigma_aad: float, sigma_ase: float) -> float:
    """
    垂直重叠概率 Pz(Sz) — 修正版

    Pz(Sz) = 2λz × ∫ f_TVE(z) · f_TVE(Sz + z) dz

    使用解析 TVE PDF（Laplace⊛Normal），SciPy quad 数值积分
    相比原版：
      - 不再将 TVE 近似为 Normal(0, √(σ_aad²+σ_ase²))
      - 正确保留 Laplace 分布的重尾特性 → 更保守的风险估计
    """
    b_ase = sigma_ase / np.sqrt(2.0)          # 标准差 → Laplace 尺度参数
    half_width = 10.0 * max(sigma_aad, b_ase * np.sqrt(2.0))

    def integrand(z1: float) -> float:
        return _tve_pdf(z1, sigma_aad, b_ase) * _tve_pdf(Sz + z1, sigma_aad, b_ase)

    integral, _ = integrate.quad(
        integrand,
        -half_width, half_width,
        limit=200, epsabs=1e-14, epsrel=1e-10
    )
    # 概率上限 1.0：当 2λz 超过误差分布宽度时线性化近似失效，需截断
    return min(1.0, 2.0 * lambda_z * integral)


def _compute_Py_Sy(Sy: float, RNP: float, lambda_y: float) -> float:
    """
    侧向重叠概率 Py(Sy) — Laplace 卷积（ICAO 标准）

    b = -RNP / ln(0.05) 为 Laplace 尺度参数
    Py(Sy) = 2λy × ∫ f_Laplace(y, b) · f_Laplace(Sy+y, b) dy
    """
    b = -RNP / np.log(0.05)

    def integrand(y1: float) -> float:
        return _laplace_pdf(y1, 0.0, b) * _laplace_pdf(Sy + y1, 0.0, b)

    integral, _ = integrate.quad(
        integrand,
        -10.0 * b, 10.0 * b,
        limit=100, epsabs=1e-14, epsrel=1e-10
    )
    return 2.0 * lambda_y * integral


def _compute_Py_0(RNP: float, lambda_y: float) -> float:
    """
    Py(0) 解析解 — 修正版

    Py(0) = 2λy × ∫ [f_Laplace(y,b)]² dy
          = 2λy × ∫ (1/2b)² exp(-2|y|/b) dy
          = 2λy × (1/4b²) × 2 × (b/2)
          = λy / (2b)

    原代码错误：使用 b=1（硬编码），与 RNP 完全无关
    修正后：b = -RNP / ln(0.05)，随导航精度正确变化
    """
    b = -RNP / np.log(0.05)
    return lambda_y / (2.0 * b)


def _compute_HOP(separation: float, RNP: float, lambda_xy: float) -> float:
    """
    水平重叠概率 HOP — Hsu 纵向风险模型（修正版）

    ICAO Doc 9689 A.2 标准公式:
      HOP(s) = (π·λ²) / (4b²) × (1 + 2s/b) × exp(-2s/b)

    原代码错误：系数为 π/16，ratio = s/b（ICAO 原式为 π/4，ratio = 2s/b）
    修正后系数提高 4×，衰减更快（exp(-2s/b) vs exp(-s/b)）
    """
    b = -RNP / np.log(0.05)
    ratio = 2.0 * separation / b                          # 修正：乘以 2
    coef  = (np.pi * lambda_xy ** 2) / (4.0 * b ** 2)    # 修正：4 非 16
    return coef * (1.0 + ratio) * np.exp(-ratio)


# ── 主计算函数 ────────────────────────────────────────────────────────────────

def compute_3d_risk(params: Dict) -> Dict:
    """
    CREAM 三维碰撞风险完整计算 — 修正版 v2.1

    参数说明（带 * 为新增）:
      N_ac*      : 评估期内走廊飞行架次 (default 10)
      T_period*  : 评估时间窗口 (s, default 3600)
      delta_V*   : 同向速度偏差，用于 V_rel_same (same unit as V, default 5% V)

    修正内容:
      1. TVE = Laplace⊛Normal 解析卷积（非高斯近似）
      2. Py(0) = λy/(2b)，b 由 RNP 导出
      3. 运动学因子 kf = (1+ẏ/2V) × (1+λxy·ż/λz/2V)，乘法结构
      4. n_z_equiv = N_ac·2λz / (V·T_period)，从流量推导
      5. Pz(0) 从 TVE 模型计算，不使用传入常数
      6. CR = (1-Ey_opp)·CR_same + Ey_opp·CR_opp，区分 V_rel
      7. HOP 系数修正为 π/4
      8. K_lat 使用修正公式
    """
    # ── 提取参数 ─────────────────────────────────────────
    Sx        = float(params['Sx'])
    Sy        = float(params['Sy'])
    Sz        = float(params['Sz'])
    RNP       = float(params['RNP'])
    V         = float(params['V'])
    y_dot     = float(params['y_dot'])
    z_dot     = float(params['z_dot'])
    lambda_x  = float(params['lambda_x'])
    lambda_y  = float(params['lambda_y'])
    lambda_z  = float(params['lambda_z'])
    lambda_xy = float(params['lambda_xy'])
    sigma_aad = float(params['sigma_aad'])
    sigma_ase = float(params['sigma_ase'])
    Ey_opp    = float(params['Ey_opp'])

    # 新增可选参数（带默认值）
    N_ac      = float(params.get('N_ac',      10.0))
    T_period  = float(params.get('T_period',  3600.0))
    delta_V   = float(params.get('delta_V',   max(0.05 * V, 0.5)))  # 同向速度偏差

    V = max(V, 1e-6)

    # ═══ 垂直重叠概率 ════════════════════════════════════
    # 单位修正：TVE 积分在英尺域（σ_aad/σ_ase 为 ft），λz 必须从 NM 换算为 ft，
    # 否则 Pz 被低估约 6076 倍（曾导致高密度场景误判 PASS）
    lambda_z_ft = lambda_z * 6076.115
    Pz_Sz   = _compute_Pz_Sz(Sz,  lambda_z_ft, sigma_aad, sigma_ase)
    Pz_0    = _compute_Pz_Sz(0.0, lambda_z_ft, sigma_aad, sigma_ase)  # Pz(0) 从模型计算

    # ═══ 垂直风险 Naz ════════════════════════════════════
    # Py(0) 解析解：λy / (2b)，b = -RNP/ln(0.05)
    Py_0    = _compute_Py_0(RNP, lambda_y)

    # 运动学因子 — 乘法结构 (ICAO Doc 9689 A.3.3)
    # kf_y = 1 + ẏ/(2V)          ← 侧向偏差率贡献
    # kf_z = 1 + λxy·ż/(λz·2V)   ← 垂直偏差率贡献
    kf_y      = 1.0 + y_dot / (2.0 * V)
    kf_z      = 1.0 + (lambda_xy / lambda_z) * (z_dot / (2.0 * V))
    kinematic = kf_y * kf_z                               # 乘法（非原来的加法）

    # 等效通过频率 — 从流量密度推导
    # n_z_equiv ≈ N_ac · 2λz / (V · T_period)  [ICAO Doc 9689 eq. A-6]
    # 原固定值 0.357 来自传统民航，对低空 UAV 走廊不适用
    # 单位修正：V 为 knots(NM/h)，T_period 为秒 → 换算为小时，否则低估 3600 倍
    T_hours = max(T_period / 3600.0, 1e-6)
    n_z_equiv = (N_ac * 2.0 * lambda_z) / (V * T_hours)
    n_z_equiv = max(n_z_equiv, 1e-8)

    Naz = 2.0 * Pz_Sz * Py_0 * n_z_equiv * kinematic

    # ═══ 侧向风险 Nay ════════════════════════════════════
    Py_Sy = _compute_Py_Sy(Sy, RNP, lambda_y)

    # 侧向运动学因子 (ICAO Doc 9689 B.2)
    # K_lat = V·(λx + λy) / (λz · 2 · Sy)
    # 原公式有冗余的 (λy+λz)·2λz 项，不符合 ICAO 原式
    K_lat = (V * (lambda_x + lambda_y)) / (lambda_z * 2.0 * max(Sy, 1e-6))

    Nay = Py_Sy * Pz_0 * Ey_opp * K_lat

    # ═══ 纵向风险 CR ════════════════════════════════════
    HOP = _compute_HOP(Sx, RNP, lambda_xy)

    # 修正：区分同向/对向相对速度（原代码 V_rel = 0.01 硬编码，导致 CR 虚高）
    V_rel_same = max(abs(delta_V), 0.001 * V)    # 同向：速度偏差（永不为零）
    V_rel_opp  = 2.0 * V                         # 对向：速度之和

    def _cr_component(V_rel_i: float) -> float:
        """单方向 CR 分量"""
        return (2.0 * HOP * Pz_0 * lambda_z * z_dot) / (
            2.0 * max(V_rel_i, 1e-8) * np.pi
            * (lambda_xy + lambda_z) * 2.0 * lambda_z
        )

    # 按对向比例加权 (ICAO Doc 9689 A.2.4)
    CR = ((1.0 - Ey_opp) * _cr_component(V_rel_same)
          + Ey_opp        * _cr_component(V_rel_opp))

    # ═══ 汇总 ═══════════════════════════════════════════
    total = Naz + Nay + CR
    TLS   = 5e-9   # 目标安全水平 (每飞行小时)

    return {
        "Naz":          float(Naz),
        "Nay":          float(Nay),
        "CR":           float(CR),
        "total_risk":   float(total),
        "Pz_Sz":        float(Pz_Sz),
        "Py_Sy":        float(Py_Sy),
        "HOP":          float(HOP),
        "Pz_0":         float(Pz_0),      # 新增：方便前端展示
        "Py_0":         float(Py_0),      # 新增
        "is_compliant": bool(total < TLS),
        "tls_ratio":    float(total / TLS),
    }


# ── UAV 入侵碰撞概率（布朗运动模型，不变）────────────────────────────────────

def compute_uav_collision(params: Dict) -> Dict:
    """
    无人机入侵碰撞概率（布朗运动扩散模型）
    该模块计算方法已与 NTU ATMRI 对齐，本次不修改
    """
    d       = params['d']
    beta    = np.radians(params['beta'])
    theta   = np.radians(params['theta'])
    Vh      = params['Vh'] * 1000 / 3600      # km/h → m/s
    Rm      = params['Rm']
    Qm      = params['Qm']
    sigma_h = params['sigma_h']
    sigma_v = params['sigma_v']
    T       = params['T']

    D             = d * np.cos(beta)
    approach_rate = Vh * np.cos(theta)

    t_closest = D / approach_rate if approach_rate > 0.01 else T
    closest_dist = D * np.abs(np.sin(theta))

    sigma_spread = sigma_h * np.sqrt(max(t_closest, 1.0))

    from scipy.stats import norm
    if closest_dist < Rm:
        Ph = (norm.cdf(Rm - closest_dist, 0, sigma_spread)
              - norm.cdf(-Rm - closest_dist, 0, sigma_spread))
    else:
        Ph = 2 * norm.cdf(Rm, closest_dist, sigma_spread) - 1
    Ph = float(np.clip(Ph, 0.0, 1.0))

    h0 = d * np.sin(beta)
    sigma_v_spread = sigma_v * np.sqrt(T)
    Pv = 1.0 if abs(h0) < Qm else (2 * norm.cdf(Qm, abs(h0), sigma_v_spread) - 1)
    Pv = float(np.clip(Pv, 0.0, 1.0))

    return {
        "P_total":           float(Ph * Pv),
        "P_horizontal":      float(Ph),
        "P_vertical":        float(Pv),
        "closest_distance_m": float(closest_dist),
        "time_to_closest_s": float(t_closest),
    }
