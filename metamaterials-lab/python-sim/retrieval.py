"""
NRW Parameter Retrieval from S-parameters
"""

import numpy as np
import pandas as pd
from typing import Tuple

def nrw_retrieval(f_hz, S11, S21, thickness_m):
    """
    NRW retrieval method with branch selection
    Returns: n, z, eps, mu
    """
    c = 299792458.0
    k0 = 2 * np.pi * f_hz / c
    
    S11 = np.asarray(S11, dtype=np.complex128)
    S21 = np.asarray(S21, dtype=np.complex128)
    
    # Calculate impedance
    num = (1 + S11)**2 - S21**2
    den = (1 - S11)**2 - S21**2
    z = np.sqrt(num / (den + 1e-18))
    z = np.where(np.real(z) < 0, -z, z)
    
    # Calculate refractive index
    X = (1 - S11**2 + S21**2) / (2 * S21 + 1e-18)
    X = np.clip(X, -1+1e-10, 1-1e-10)
    
    k0d = k0 * thickness_m
    n = np.arccos(X) / (k0d + 1e-18)
    
    # Phase unwrapping
    n_real = np.unwrap(np.real(n))
    n = n_real + 1j * np.imag(n)
    
    # Calculate permittivity and permeability
    eps = n / z
    mu = n * z
    
    return n, z, eps, mu

if __name__ == "__main__":
    # Load and process data
    df = pd.read_csv("sample_data/metamaterial_data.csv")
    
    f_hz = df["f_THz"].values * 1e12
    S11 = df["S11_real"].values + 1j * df["S11_imag"].values
    S21 = df["S21_real"].values + 1j * df["S21_imag"].values
    
    n, z, eps, mu = nrw_retrieval(f_hz, S11, S21, 200e-6)
    
    print(f"✅ Retrieved parameters")
    
    # Find negative index region
    neg_idx = np.where(np.real(n) < 0)[0]
    if len(neg_idx) > 0:
        f_thz = df["f_THz"].values
        print(f"🎯 Negative n: {f_thz[neg_idx[0]]:.2f} - {f_thz[neg_idx[-1]]:.2f} THz")