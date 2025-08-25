#!/usr/bin/env python3
"""
Metamaterials Research Lab - Working Full-Stack Builder
This script creates a complete, working full-stack application
"""

import os
import json
import shutil
from pathlib import Path

def create_metamaterials_project(base_dir="metamaterials-lab"):
    """Create the complete working project"""
    
    root = Path(base_dir)
    print(f"🔬 Building Metamaterials Research Lab at: {root.absolute()}")
    
    # Clean and create directory structure
    if root.exists():
        shutil.rmtree(root)
    
    # Create all directories
    dirs = [
        root,
        root / "api",
        root / "python-sim",
        root / "python-sim" / "sample_data",
        root / "client",
        root / "client" / "app",
        root / "client" / "app" / "api",
        root / "client" / "app" / "simulate",
        root / "client" / "app" / "results",
        root / "client" / "components",
        root / "client" / "public",
        root / "docker",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    # ============================================
    # ROOT FILES
    # ============================================
    
    # README.md
    readme = """# Metamaterials Research Lab

## Quick Start

### Option 1: Using Docker (Recommended)
```bash
cd docker
docker-compose up
```

### Option 2: Manual Setup

#### Backend (Terminal 1):
```bash
cd api
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

#### Frontend (Terminal 2):
```bash
cd client
npm install
npm run dev
```

#### Python Simulation (Terminal 3):
```bash
cd python-sim
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python generate_data.py
```

## Access Points
- Frontend: http://localhost:3000
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
"""
    (root / "README.md").write_text(readme)
    
    # .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
venv/
.venv/
*.egg-info/

# Node
node_modules/
.next/
out/
dist/
*.log

# Data
*.h5
*.hdf5
!sample_data/*.csv

# IDE
.vscode/
.idea/
.DS_Store

# Environment
.env
.env.local
"""
    (root / ".gitignore").write_text(gitignore)
    
    # ============================================
    # API BACKEND
    # ============================================
    
    # api/requirements.txt
    api_requirements = """fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
numpy==1.24.3
pandas==2.0.3
python-multipart==0.0.6
python-dotenv==1.0.0
httpx==0.25.0
"""
    (root / "api" / "requirements.txt").write_text(api_requirements)
    
    # api/.env
    api_env = """API_HOST=0.0.0.0
API_PORT=8000
"""
    (root / "api" / ".env").write_text(api_env)
    
    # api/main.py
    api_main = '''"""
FastAPI Backend for Metamaterials Research
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import json
import uuid
from datetime import datetime
import os

# Create FastAPI app
app = FastAPI(
    title="Metamaterials API",
    version="1.0.0",
    description="API for metamaterials simulation and analysis"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data directory
DATA_DIR = Path("../python-sim/sample_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============ Models ============

class SimulationRequest(BaseModel):
    name: str
    frequencies: List[float]  # in THz
    lattice_x: float = 60.0  # in micrometers
    lattice_y: float = 60.0
    thickness: float = 200.0
    epsilon_substrate: float = 3.0

class SimulationResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict] = None

# ============ Routes ============

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
    """Run metamaterial simulation"""
    try:
        job_id = str(uuid.uuid4())[:8]
        
        # Convert frequencies to Hz
        frequencies_hz = [f * 1e12 for f in request.frequencies]
        
        # Generate synthetic S-parameters (simplified model)
        results = []
        for freq in frequencies_hz:
            # Simple resonance model
            f0 = 1.1e12  # Resonance at 1.1 THz
            gamma = 0.05e12
            resonance = 1.0 / (1.0 + ((freq - f0) / gamma) ** 2)
            
            s11 = complex(0.1 * resonance, -0.05)
            s21 = complex(np.sqrt(1 - abs(s11)**2), -0.1 * resonance)
            
            results.append({
                "frequency": freq,
                "S11_real": s11.real,
                "S11_imag": s11.imag,
                "S21_real": s21.real,
                "S21_imag": s21.imag
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
                "file": str(output_file)
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    """Get material database"""
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
        }
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload data file"""
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
    
    print(f"🚀 Starting API server at http://{host}:{port}")
    print(f"📚 API documentation at http://{host}:{port}/docs")
    
    uvicorn.run(app, host=host, port=port, reload=True)
'''
    (root / "api" / "main.py").write_text(api_main)
    
    # ============================================
    # PYTHON SIMULATION
    # ============================================
    
    # python-sim/requirements.txt
    sim_requirements = """numpy==1.24.3
pandas==2.0.3
matplotlib==3.6.3
scipy==1.10.1
"""
    (root / "python-sim" / "requirements.txt").write_text(sim_requirements)
    
    # python-sim/generate_data.py
    generate_data = '''"""
Generate synthetic metamaterial data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def lorentz_model(omega, omega0, F, gamma):
    """Lorentz model for permeability"""
    return 1 + (F * omega**2) / (omega0**2 - omega**2 - 1j * gamma * omega)

def drude_model(omega, omega_p, gamma):
    """Drude model for permittivity"""
    return 1 - (omega_p**2) / (omega * (omega + 1j * gamma))

def generate_metamaterial_data():
    """Generate synthetic S-parameters for negative index metamaterial"""
    
    # Frequency range
    f_thz = np.linspace(0.5, 2.0, 200)
    f_hz = f_thz * 1e12
    omega = 2 * np.pi * f_hz
    
    # Material parameters
    omega0 = 2 * np.pi * 1.1e12  # Magnetic resonance at 1.1 THz
    F = 0.9
    gamma_m = 2 * np.pi * 0.06e12
    
    omega_p = 2 * np.pi * 1.6e12  # Plasma frequency
    gamma_e = 2 * np.pi * 0.1e12
    
    # Calculate effective parameters
    mu = lorentz_model(omega, omega0, F, gamma_m)
    eps = drude_model(omega, omega_p, gamma_e)
    
    # Effective index and impedance
    n = np.sqrt(mu * eps)
    z = np.sqrt(mu / eps)
    
    # Calculate S-parameters (simplified)
    d = 200e-6  # 200 micrometers thickness
    k0 = omega / 3e8
    
    # Fresnel coefficients
    r = (z - 1) / (z + 1)
    t = 2 * z / (z + 1)
    
    # Phase factor
    phase = np.exp(-1j * n * k0 * d)
    
    # S-parameters
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
        "mu_real": np.real(mu)
    })
    
    # Save data
    output_dir = Path("sample_data")
    output_dir.mkdir(exist_ok=True)
    
    df.to_csv(output_dir / "metamaterial_data.csv", index=False)
    print(f"✅ Generated {len(df)} data points")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Refractive index
    axes[0, 0].plot(f_thz, np.real(n), label="Re(n)", linewidth=2)
    axes[0, 0].plot(f_thz, np.imag(n), label="Im(n)", linewidth=2)
    axes[0, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[0, 0].set_xlabel("Frequency (THz)")
    axes[0, 0].set_ylabel("Refractive Index")
    axes[0, 0].set_title("Effective Refractive Index")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: S-parameters
    axes[0, 1].plot(f_thz, np.abs(S21), label="|S21|", linewidth=2)
    axes[0, 1].plot(f_thz, np.abs(S11), label="|S11|", linewidth=2)
    axes[0, 1].set_xlabel("Frequency (THz)")
    axes[0, 1].set_ylabel("Magnitude")
    axes[0, 1].set_title("S-Parameters")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Permittivity
    axes[1, 0].plot(f_thz, np.real(eps), label="Re(ε)", linewidth=2)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 0].set_xlabel("Frequency (THz)")
    axes[1, 0].set_ylabel("Permittivity")
    axes[1, 0].set_title("Electric Permittivity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Permeability
    axes[1, 1].plot(f_thz, np.real(mu), label="Re(μ)", linewidth=2)
    axes[1, 1].axhline(y=0, color="k", linestyle="--", alpha=0.3)
    axes[1, 1].set_xlabel("Frequency (THz)")
    axes[1, 1].set_ylabel("Permeability")
    axes[1, 1].set_title("Magnetic Permeability")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metamaterial_plots.png", dpi=150)
    print("✅ Saved plots to sample_data/metamaterial_plots.png")
    
    # Find negative index region
    neg_n = f_thz[np.real(n) < 0]
    if len(neg_n) > 0:
        print(f"🎯 Negative index band: {neg_n[0]:.2f} - {neg_n[-1]:.2f} THz")
    
    return df

if __name__ == "__main__":
    print("🔬 Generating metamaterial data...")
    df = generate_metamaterial_data()
    print("✅ Data generation complete!")
'''
    (root / "python-sim" / "generate_data.py").write_text(generate_data)
    
    # ============================================
    # NEXT.JS CLIENT
    # ============================================
    
    # client/package.json
    package_json = {
        "name": "metamaterials-client",
        "version": "1.0.0",
        "private": True,
        "scripts": {
            "dev": "next dev",
            "build": "next build",
            "start": "next start",
            "lint": "next lint"
        },
        "dependencies": {
            "next": "14.0.4",
            "react": "^18",
            "react-dom": "^18",
            "axios": "^1.6.2",
            "@types/node": "^20",
            "@types/react": "^18",
            "@types/react-dom": "^18",
            "autoprefixer": "^10.0.1",
            "postcss": "^8",
            "tailwindcss": "^3.3.0",
            "typescript": "^5"
        }
    }
    (root / "client" / "package.json").write_text(json.dumps(package_json, indent=2))
    
    # client/tsconfig.json
    tsconfig = {
        "compilerOptions": {
            "target": "es5",
            "lib": ["dom", "dom.iterable", "esnext"],
            "allowJs": True,
            "skipLibCheck": True,
            "strict": True,
            "noEmit": True,
            "esModuleInterop": True,
            "module": "esnext",
            "moduleResolution": "bundler",
            "resolveJsonModule": True,
            "isolatedModules": True,
            "jsx": "preserve",
            "incremental": True,
            "paths": {
                "@/*": ["./*"]
            }
        },
        "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx"],
        "exclude": ["node_modules"]
    }
    (root / "client" / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2))
    
    # client/next.config.js
    next_config = """/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
}

module.exports = nextConfig
"""
    (root / "client" / "next.config.js").write_text(next_config)
    
    # client/tailwind.config.js
    tailwind_config = """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
"""
    (root / "client" / "tailwind.config.js").write_text(tailwind_config)
    
    # client/postcss.config.js
    postcss_config = """module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
"""
    (root / "client" / "postcss.config.js").write_text(postcss_config)
    
    # client/app/globals.css
    globals_css = """@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  padding: 0;
}
"""
    (root / "client" / "app" / "globals.css").write_text(globals_css)
    
    # client/app/layout.tsx
    layout = """import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Metamaterials Research Lab',
  description: 'Negative Index Materials Platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
"""
    (root / "client" / "app" / "layout.tsx").write_text(layout)
    
    # client/app/page.tsx
    home_page = """'use client'

import Link from 'next/link'
import { useEffect, useState } from 'react'

export default function Home() {
  const [apiStatus, setApiStatus] = useState('checking...')
  
  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(res => res.json())
      .then(data => setApiStatus('online'))
      .catch(() => setApiStatus('offline'))
  }, [])
  
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 to-blue-600">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center text-white mb-12">
          <h1 className="text-5xl font-bold mb-4">
            Metamaterials Research Lab
          </h1>
          <p className="text-xl">Negative Index Materials at THz Frequencies</p>
          <p className="mt-4">
            API Status: <span className={apiStatus === 'online' ? 'text-green-300' : 'text-red-300'}>
              {apiStatus}
            </span>
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl mx-auto">
          <Link href="/simulate" className="bg-white rounded-lg p-8 hover:shadow-xl transition">
            <h2 className="text-2xl font-bold mb-3">Simulate</h2>
            <p className="text-gray-600">Run electromagnetic simulations</p>
          </Link>
          
          <Link href="/results" className="bg-white rounded-lg p-8 hover:shadow-xl transition">
            <h2 className="text-2xl font-bold mb-3">Results</h2>
            <p className="text-gray-600">View simulation results</p>
          </Link>
          
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"
            className="bg-white rounded-lg p-8 hover:shadow-xl transition">
            <h2 className="text-2xl font-bold mb-3">API Docs</h2>
            <p className="text-gray-600">Interactive API documentation</p>
          </a>
          
          <div className="bg-white rounded-lg p-8">
            <h2 className="text-2xl font-bold mb-3">Materials</h2>
            <p className="text-gray-600">Gold, Silver, Silicon, Silica</p>
          </div>
        </div>
      </div>
    </div>
  )
}
"""
    (root / "client" / "app" / "page.tsx").write_text(home_page)
    
    # client/app/simulate/page.tsx
    simulate_page = """'use client'

import { useState } from 'react'
import axios from 'axios'
import Link from 'next/link'

export default function SimulatePage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [formData, setFormData] = useState({
    name: 'Test Simulation',
    freqStart: 0.5,
    freqEnd: 2.0,
    numPoints: 50,
    latticeX: 60,
    latticeY: 60,
    thickness: 200,
    epsilonSubstrate: 3.0
  })
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    
    try {
      // Generate frequency array
      const frequencies = []
      for (let i = 0; i < formData.numPoints; i++) {
        const freq = formData.freqStart + 
          (i / (formData.numPoints - 1)) * (formData.freqEnd - formData.freqStart)
        frequencies.push(freq)
      }
      
      const response = await axios.post('http://localhost:8000/api/simulate', {
        name: formData.name,
        frequencies: frequencies,
        lattice_x: formData.latticeX,
        lattice_y: formData.latticeY,
        thickness: formData.thickness,
        epsilon_substrate: formData.epsilonSubstrate
      })
      
      setResult(response.data)
    } catch (error) {
      console.error('Error:', error)
      alert('Simulation failed. Make sure the API is running.')
    } finally {
      setLoading(false)
    }
  }
  
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <Link href="/" className="text-blue-600 hover:underline mb-4 inline-block">
          ← Back to Home
        </Link>
        
        <h1 className="text-3xl font-bold mb-8">Run Simulation</h1>
        
        <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Simulation Name</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({...formData, name: e.target.value})}
              className="w-full border rounded px-3 py-2"
              required
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Start Frequency (THz)</label>
              <input
                type="number"
                step="0.1"
                value={formData.freqStart}
                onChange={(e) => setFormData({...formData, freqStart: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">End Frequency (THz)</label>
              <input
                type="number"
                step="0.1"
                value={formData.freqEnd}
                onChange={(e) => setFormData({...formData, freqEnd: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-1">Number of Points</label>
            <input
              type="number"
              value={formData.numPoints}
              onChange={(e) => setFormData({...formData, numPoints: parseInt(e.target.value)})}
              className="w-full border rounded px-3 py-2"
              min="10"
              max="1000"
              required
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Lattice X (μm)</label>
              <input
                type="number"
                value={formData.latticeX}
                onChange={(e) => setFormData({...formData, latticeX: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Lattice Y (μm)</label>
              <input
                type="number"
                value={formData.latticeY}
                onChange={(e) => setFormData({...formData, latticeY: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Running...' : 'Run Simulation'}
          </button>
        </form>
        
        {result && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Results</h2>
            <p>Job ID: {result.job_id}</p>
            <p>Status: {result.status}</p>
            <p>Data points: {result.result?.data?.length || 0}</p>
            <Link href={`/results?job_id=${result.job_id}`} 
              className="mt-4 inline-block bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700">
              View Results
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}
"""
    (root / "client" / "app" / "simulate" / "page.tsx").write_text(simulate_page)
    
    # client/app/results/page.tsx
    results_page = """'use client'

import { useEffect, useState } from 'react'
import { useSearchParams } from 'next/navigation'
import Link from 'next/link'
import axios from 'axios'

export default function ResultsPage() {
  const searchParams = useSearchParams()
  const jobId = searchParams.get('job_id')
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  
  useEffect(() => {
    if (jobId) {
      axios.get(`http://localhost:8000/api/results/${jobId}`)
        .then(res => {
          setData(res.data)
          setLoading(false)
        })
        .catch(err => {
          console.error(err)
          setLoading(false)
        })
    } else {
      setLoading(false)
    }
  }, [jobId])
  
  if (loading) return <div className="p-8">Loading...</div>
  
  if (!data) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <Link href="/" className="text-blue-600 hover:underline mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-3xl font-bold mb-4">No Results</h1>
          <p>Please run a simulation first.</p>
          <Link href="/simulate" className="mt-4 inline-block bg-blue-600 text-white px-4 py-2 rounded">
            Go to Simulate
          </Link>
        </div>
      </div>
    )
  }
  
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <Link href="/" className="text-blue-600 hover:underline mb-4 inline-block">
          ← Back to Home
        </Link>
        
        <h1 className="text-3xl font-bold mb-8">Simulation Results</h1>
        <p className="mb-4">Job ID: {jobId}</p>
        
        <div className="bg-white rounded-lg shadow overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-4 py-2 text-left">Frequency (THz)</th>
                <th className="px-4 py-2 text-left">S11 Real</th>
                <th className="px-4 py-2 text-left">S11 Imag</th>
                <th className="px-4 py-2 text-left">S21 Real</th>
                <th className="px-4 py-2 text-left">S21 Imag</th>
              </tr>
            </thead>
            <tbody>
              {data.data.slice(0, 10).map((row: any, i: number) => (
                <tr key={i} className="border-t">
                  <td className="px-4 py-2">{(row.frequency / 1e12).toFixed(3)}</td>
                  <td className="px-4 py-2">{row.S11_real.toFixed(4)}</td>
                  <td className="px-4 py-2">{row.S11_imag.toFixed(4)}</td>
                  <td className="px-4 py-2">{row.S21_real.toFixed(4)}</td>
                  <td className="px-4 py-2">{row.S21_imag.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {data.data.length > 10 && (
            <div className="p-4 text-center text-gray-500">
              Showing first 10 of {data.data.length} results
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
"""
    (root / "client" / "app" / "results" / "page.tsx").write_text(results_page)
    
    # ============================================
    # DOCKER CONFIGURATION
    # ============================================
    
    # docker/docker-compose.yml
    docker_compose = """version: '3.8'

services:
  api:
    build:
      context: ../api
      dockerfile: Dockerfile
    container_name: metamaterials-api
    ports:
      - "8000:8000"
    volumes:
      - ../api:/app
      - ../python-sim/sample_data:/app/sample_data
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
    command: python main.py
    
  client:
    build:
      context: ../client
      dockerfile: Dockerfile
    container_name: metamaterials-client
    ports:
      - "3000:3000"
    volumes:
      - ../client:/app
      - /app/node_modules
      - /app/.next
    environment:
      - NODE_ENV=development
    depends_on:
      - api

networks:
  default:
    name: metamaterials-network
"""
    (root / "docker" / "docker-compose.yml").write_text(docker_compose)
    
    # api/Dockerfile
    api_dockerfile = """FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
"""
    (root / "api" / "Dockerfile").write_text(api_dockerfile)
    
    # client/Dockerfile
    client_dockerfile = """FROM node:20-alpine

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
"""
    (root / "client" / "Dockerfile").write_text(client_dockerfile)
    
    print("\n" + "="*60)
    print("✅ METAMATERIALS PROJECT CREATED SUCCESSFULLY!")
    print("="*60)
    print(f"\n📁 Project location: {root.absolute()}")
    print("\n🚀 TO START THE PROJECT:")
    print("\n  Option 1 - Using Docker:")
    print("  -------------------------")
    print(f"  cd {root}/docker")
    print("  docker-compose up")
    print("\n  Option 2 - Manual Setup:")
    print("  ------------------------")
    print("  Terminal 1 (Backend):")
    print(f"    cd {root}/api")
    print("    python3 -m venv venv")
    print("    source venv/bin/activate  # Windows: venv\\Scripts\\activate")
    print("    pip install -r requirements.txt")
    print("    python main.py")
    print("\n  Terminal 2 (Frontend):")
    print(f"    cd {root}/client")
    print("    npm install")
    print("    npm run dev")
    print("\n  Terminal 3 (Generate Data):")
    print(f"    cd {root}/python-sim")
    print("    python3 -m venv venv")
    print("    source venv/bin/activate")
    print("    pip install -r requirements.txt")
    print("    python generate_data.py")
    print("\n📍 ACCESS POINTS:")
    print("  - Frontend: http://localhost:3000")
    print("  - API: http://localhost:8000")
    print("  - API Docs: http://localhost:8000/docs")
    print("\n✨ Features:")
    print("  - Run simulations through web interface")
    print("  - View S-parameters and results")
    print("  - Interactive API documentation")
    print("  - Negative index metamaterial modeling")

if __name__ == "__main__":
    create_metamaterials_project()
