"""
Metamaterials Research API
Complete electromagnetic simulation backend
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json
import uuid
from datetime import datetime
import os

app = FastAPI(
    title="Metamaterials API",
    version="1.0.0",
    description="API for metamaterials simulation and analysis"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path("../python-sim/sample_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

class SimulationRequest(BaseModel):
    name: str
    frequencies: List[float]  # THz
    lattice_x: float = 60.0  # micrometers
    lattice_y: float = 60.0
    thickness: float = 200.0
    epsilon_substrate: float = 3.0
    mu_substrate: float = 1.0

class SimulationResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict] = None

@app.get("/")
async def root():
    return {
        "message": "Metamaterials Research API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/simulate")
async def run_simulation(request: SimulationRequest):
    """
    Run metamaterial simulation
    Implements negative index metamaterial with Lorentz-Drude model
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Convert frequencies to Hz
    frequencies_hz = np.array(request.frequencies) * 1e12
    omega = 2 * np.pi * frequencies_hz
    
    # Material parameters for negative index
    omega0 = 2 * np.pi * 1.1e12  # Magnetic resonance at 1.1 THz
    F = 0.9  # Oscillator strength
    gamma_m = 2 * np.pi * 0.06e12  # Magnetic damping
    omega_p = 2 * np.pi * 1.6e12  # Plasma frequency
    gamma_e = 2 * np.pi * 0.1e12  # Electric damping
    
    # Lorentz model for permeability
    mu = 1 + (F * omega**2) / (omega0**2 - omega**2 - 1j * gamma_m * omega)
    
    # Drude model for permittivity
    eps = 1 - (omega_p**2) / (omega * (omega + 1j * gamma_e))
    
    # Effective parameters
    # For negative index, we need to choose the correct branch
    n_squared = mu * eps
    n = np.sqrt(n_squared)
    # Choose negative branch when both eps and mu are negative
    for i in range(len(omega)):
        if np.real(eps[i]) < 0 and np.real(mu[i]) < 0:
            if np.real(n[i]) > 0:
                n[i] = -n[i]
    
    z = np.sqrt(mu / eps)
    
    # Calculate S-parameters
    d = request.thickness * 1e-6
    k0 = omega / 3e8
    
    # Transmission and reflection coefficients
    r = (z - 1) / (z + 1)
    t = 2 * z / (z + 1)
    
    # Propagation
    phase = np.exp(-1j * n * k0 * d)
    
    # S-parameters with Fabry-Perot
    S11 = r * (1 - phase**2) / (1 - r**2 * phase**2)
    S21 = t**2 * phase / (1 - r**2 * phase**2)
    
    # Prepare results
    results = []
    for i, freq in enumerate(frequencies_hz):
        results.append({
            "frequency": float(freq),
            "S11_real": float(np.real(S11[i])),
            "S11_imag": float(np.imag(S11[i])),
            "S21_real": float(np.real(S21[i])),
            "S21_imag": float(np.imag(S21[i])),
            "n_real": float(np.real(n[i])),
            "n_imag": float(np.imag(n[i])),
            "eps_real": float(np.real(eps[i])),
            "mu_real": float(np.real(mu[i]))
        })
    
    # Save results
    df = pd.DataFrame(results)
    output_file = DATA_DIR / f"simulation_{job_id}.csv"
    df.to_csv(output_file, index=False)
    
    return SimulationResponse(
        job_id=job_id,
        status="completed",
        result={
            "data": results,
            "file": str(output_file),
            "negative_index_band": identify_negative_index_band(df)
        }
    )

def identify_negative_index_band(df):
    """Identify frequency range with negative refractive index"""
    neg_n = df[df["n_real"] < 0]
    if len(neg_n) > 0:
        f_min = neg_n["frequency"].min() / 1e12
        f_max = neg_n["frequency"].max() / 1e12
        return {"min_THz": f_min, "max_THz": f_max}
    return None

@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get simulation results"""
    file_path = DATA_DIR / f"simulation_{job_id}.csv"
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    
    df = pd.read_csv(file_path)
    return {
        "job_id": job_id,
        "data": df.to_dict(orient="records")
    }

@app.get("/api/materials")
async def get_materials():
    """Material database"""
    return {
        "metals": {
            "gold": {"plasma_freq": 2.175e15, "collision_freq": 6.5e12},
            "silver": {"plasma_freq": 2.175e15, "collision_freq": 3.5e12},
            "copper": {"plasma_freq": 1.36e16, "collision_freq": 1.05e14}
        },
        "dielectrics": {
            "silicon": {"epsilon": 11.7},
            "silica": {"epsilon": 2.13},
            "alumina": {"epsilon": 9.8}
        },
        "substrates": {
            "rogers": {"epsilon": 3.0, "loss_tangent": 0.001},
            "fr4": {"epsilon": 4.4, "loss_tangent": 0.02}
        }
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload COMSOL or measurement data"""
    contents = await file.read()
    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = DATA_DIR / filename
    
    with open(file_path, "wb") as f:
        f.write(contents)
    
    return {"filename": filename, "size": len(contents)}

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    
    load_dotenv()
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    print(f"🚀 Starting API at http://{host}:{port}")
    print(f"📚 API docs at http://{host}:{port}/docs")
    
    uvicorn.run("main:app", host=host, port=port, reload=True)