#!/usr/bin/env python3
"""
Test script for the Metamaterials Research Lab
"""

import requests
import json
import numpy as np

def test_api():
    """Test the API endpoints"""
    print("🧪 Testing Metamaterials Research Lab API\n")
    
    # Test health endpoint
    print("1. Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    if response.status_code == 200:
        print("   ✅ API is healthy")
    else:
        print("   ❌ API health check failed")
        return False
    
    # Test materials endpoint
    print("\n2. Testing materials database...")
    response = requests.get("http://localhost:8000/api/materials")
    if response.status_code == 200:
        materials = response.json()
        print(f"   ✅ Found {len(materials['metals'])} metals")
        print(f"   ✅ Found {len(materials['dielectrics'])} dielectrics")
    else:
        print("   ❌ Materials endpoint failed")
        return False
    
    # Test simulation
    print("\n3. Running simulation...")
    frequencies = np.linspace(0.5, 2.0, 20).tolist()
    
    simulation_data = {
        "name": "Test Metamaterial",
        "frequencies": frequencies,
        "lattice_x": 60,
        "lattice_y": 60,
        "thickness": 200,
        "epsilon_substrate": 3.0
    }
    
    response = requests.post("http://localhost:8000/api/simulate", json=simulation_data)
    if response.status_code == 200:
        result = response.json()
        job_id = result['job_id']
        print(f"   ✅ Simulation completed (Job ID: {job_id})")
        
        # Check for negative index
        data = result['result']['data']
        neg_n_count = sum(1 for d in data if d['n_real'] < 0)
        print(f"   ✅ Found {neg_n_count}/{len(data)} points with negative index")
        
        if result['result'].get('negative_index_band'):
            band = result['result']['negative_index_band']
            print(f"   ✅ Negative index band: {band['min_THz']:.2f} - {band['max_THz']:.2f} THz")
        
        # Test results retrieval
        print(f"\n4. Testing results retrieval...")
        response = requests.get(f"http://localhost:8000/api/results/{job_id}")
        if response.status_code == 200:
            print(f"   ✅ Results retrieved successfully")
        else:
            print("   ❌ Results retrieval failed")
            return False
            
    else:
        print("   ❌ Simulation failed")
        return False
    
    print("\n5. Testing frontend...")
    response = requests.get("http://localhost:3000")
    if response.status_code == 200:
        print("   ✅ Frontend is accessible")
    else:
        print("   ❌ Frontend not accessible")
        return False
    
    print("\n" + "="*50)
    print("✅ All tests passed!")
    print("\nAccess the application at:")
    print("  • Frontend: http://localhost:3000")
    print("  • API Docs: http://localhost:8000/docs")
    print("="*50)
    
    return True

if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError as e:
        print("❌ Error: Could not connect to services")
        print("   Make sure both API (port 8000) and frontend (port 3000) are running")
    except Exception as e:
        print(f"❌ Error: {e}")