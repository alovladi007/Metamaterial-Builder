"""
Generate synthetic metamaterial data with negative index
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def lorentz_model(omega, omega0, F, gamma):
    """Lorentz model for magnetic permeability"""
    return 1 + (F * omega**2) / (omega0**2 - omega**2 - 1j * gamma * omega)

def drude_model(omega, omega_p, gamma):
    """Drude model for electric permittivity"""
    return 1 - (omega_p**2) / (omega * (omega + 1j * gamma))

def generate_metamaterial_data():
    """Generate synthetic S-parameters for negative index metamaterial"""
    
    # Frequency range
    f_thz = np.linspace(0.5, 2.0, 200)
    f_hz = f_thz * 1e12
    omega = 2 * np.pi * f_hz
    
    # Material parameters for negative index
    omega0 = 2 * np.pi * 1.1e12  # Magnetic resonance at 1.1 THz
    F = 0.9
    gamma_m = 2 * np.pi * 0.06e12
    omega_p = 2 * np.pi * 1.6e12  # Plasma frequency
    gamma_e = 2 * np.pi * 0.1e12
    
    # Calculate effective parameters
    mu = lorentz_model(omega, omega0, F, gamma_m)
    eps = drude_model(omega, omega_p, gamma_e)
    
    # Effective index and impedance
    # For negative index, we need to choose the correct branch
    n_squared = mu * eps
    n = np.sqrt(n_squared)
    # Choose negative branch when both eps and mu are negative
    for i in range(len(omega)):
        if np.real(eps[i]) < 0 and np.real(mu[i]) < 0:
            if np.real(n[i]) > 0:
                n[i] = -n[i]
    
    z = np.sqrt(mu / eps)
    
    # S-parameters for 200 μm slab
    d = 200e-6
    k0 = omega / 3e8
    
    # Fresnel coefficients
    r = (z - 1) / (z + 1)
    t = 2 * z / (z + 1)
    
    # Propagation
    phase = np.exp(-1j * n * k0 * d)
    
    # S-parameters with multiple reflections
    S11 = r * (1 - phase**2) / (1 - r**2 * phase**2)
    S21 = t**2 * phase / (1 - r**2 * phase**2)
    
    # Create dataframe
    df = pd.DataFrame({
        "f_THz": f_thz,
        "S11_real": np.real(S11),
        "S11_imag": np.imag(S11),
        "S21_real": np.real(S21),
        "S21_imag": np.imag(S21),
        "n_real": np.real(n),
        "n_imag": np.imag(n),
        "eps_real": np.real(eps),
        "eps_imag": np.imag(eps),
        "mu_real": np.real(mu),
        "mu_imag": np.imag(mu)
    })
    
    # Save data
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "metamaterial_data.csv", index=False)
    print(f"✅ Generated {len(df)} data points")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Refractive index
    axes[0, 0].plot(f_thz, np.real(n), 'b-', label="Re(n)", linewidth=2)
    axes[0, 0].plot(f_thz, np.imag(n), 'r--', label="Im(n)", linewidth=2)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel("Frequency (THz)")
    axes[0, 0].set_ylabel("Refractive Index")
    axes[0, 0].set_title("Effective Refractive Index (Negative Band)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # S-parameters
    axes[0, 1].plot(f_thz, np.abs(S21), 'g-', label="|S21|", linewidth=2)
    axes[0, 1].plot(f_thz, np.abs(S11), 'm-', label="|S11|", linewidth=2)
    axes[0, 1].set_xlabel("Frequency (THz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_title("S-Parameters")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Permittivity
    axes[1, 0].plot(f_thz, np.real(eps), 'b-', label="Re(ε)", linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel("Frequency (THz)")
    axes[1, 0].set_ylabel("Permittivity")
    axes[1, 0].set_title("Electric Permittivity (Drude)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Permeability
    axes[1, 1].plot(f_thz, np.real(mu), 'r-', label="Re(μ)", linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel("Frequency (THz)")
    axes[1, 1].set_ylabel("Permeability")
    axes[1, 1].set_title("Magnetic Permeability (Lorentz)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metamaterial_plots.png", dpi=150)
    print("✅ Saved plots")
    
    # Find negative index region
    neg_n = f_thz[np.real(n) < 0]
    if len(neg_n) > 0:
        print(f"🎯 Negative index band: {neg_n[0]:.2f} - {neg_n[-1]:.2f} THz")
    
    return df

if __name__ == "__main__":
    print("🔬 Generating metamaterial data...")
    df = generate_metamaterial_data()
    print("✅ Complete!")