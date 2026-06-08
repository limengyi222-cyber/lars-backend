"""总垂直风险 (LHD) 计算引擎"""
import numpy as np
from scipy import integrate
from typing import Dict

def _normal_pdf(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def _compute_Pz_Sz_star(s1, s2, alpha, Sz=1000, lambda_z=0.0099):
    """计算带混合 AAD 分布的 Pz(Sz)*"""
    sigma_eff = np.sqrt(s1**2 * (1-alpha) + s2**2 * alpha)
    def integrand(z1):
        return _normal_pdf(z1, 0, sigma_eff) * _normal_pdf(Sz + z1, 0, sigma_eff)
    integral, _ = integrate.quad(integrand, -5*sigma_eff, 5*sigma_eff, limit=100)
    return 2 * lambda_z * integral

def compute_total_vertical_risk(params: Dict) -> Dict:
    """
    总垂直风险 Naz_total = Naz* + Naz_cld + Naz_wl
    三种 LHD 情形的风险叠加
    """
    Py_0 = 0.149
    n_z_equiv = 0.357
    kinematic = 1.027
    
    Pz_base = 1.43e-10
    Naz_tech = 2 * Pz_base * Py_0 * n_z_equiv * kinematic
    
    Pz_star = _compute_Pz_Sz_star(params['s1'], params['s2'], params['alpha'])
    Naz_nwl = 2 * Pz_star * Py_0 * n_z_equiv * kinematic
    
    Pz_cld = params['nCLD'] / params['T'] * 0.5 * params.get('tcld', 0.08)
    Naz_cld = 2 * Pz_cld * Py_0 * n_z_equiv * kinematic
    
    Pz_wl = 0.538 * params['nWL'] * params['twl'] / params['T']
    Naz_wl = 2 * Pz_wl * Py_0 * n_z_equiv * kinematic
    
    return {
        "Naz_technical": float(Naz_tech),
        "Naz_nwl": float(Naz_nwl),
        "Naz_cld": float(Naz_cld),
        "Naz_wl": float(Naz_wl),
        "Naz_total": float(Naz_nwl + Naz_cld + Naz_wl),
        "Pz_star": float(Pz_star),
        "Pz_cld": float(Pz_cld),
        "Pz_wl": float(Pz_wl),
    }
