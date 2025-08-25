'use client'

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
            🔬 Metamaterials Research Lab
          </h1>
          <p className="text-xl">Negative Index Materials at THz/Optical Frequencies</p>
          <p className="mt-4">
            API Status: <span className={apiStatus === 'online' ? 'text-green-300' : 'text-red-300'}>
              {apiStatus}
            </span>
          </p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
          <Link href="/simulate" className="bg-white rounded-lg p-8 hover:shadow-xl transition transform hover:scale-105">
            <div className="text-4xl mb-4">⚡</div>
            <h2 className="text-2xl font-bold mb-3">Simulate</h2>
            <p className="text-gray-600">Run electromagnetic simulations with Lorentz-Drude models</p>
          </Link>
          
          <Link href="/results" className="bg-white rounded-lg p-8 hover:shadow-xl transition transform hover:scale-105">
            <div className="text-4xl mb-4">📊</div>
            <h2 className="text-2xl font-bold mb-3">Results</h2>
            <p className="text-gray-600">View S-parameters and retrieved n, ε, μ</p>
          </Link>
          
          <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer"
            className="bg-white rounded-lg p-8 hover:shadow-xl transition transform hover:scale-105">
            <div className="text-4xl mb-4">📚</div>
            <h2 className="text-2xl font-bold mb-3">API Docs</h2>
            <p className="text-gray-600">Interactive API documentation</p>
          </a>
        </div>
        
        <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4 max-w-5xl mx-auto">
          <div className="bg-white/10 backdrop-blur rounded-lg p-4 text-white">
            <h3 className="font-bold">🌊 Cloaking</h3>
            <p className="text-sm mt-1">EM invisibility</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4 text-white">
            <h3 className="font-bold">🔍 Superlensing</h3>
            <p className="text-sm mt-1">Sub-wavelength imaging</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4 text-white">
            <h3 className="font-bold">📡 Antennas</h3>
            <p className="text-sm mt-1">Miniaturized designs</p>
          </div>
          <div className="bg-white/10 backdrop-blur rounded-lg p-4 text-white">
            <h3 className="font-bold">🔬 Research</h3>
            <p className="text-sm mt-1">Novel photonics</p>
          </div>
        </div>
      </div>
    </div>
  )
}