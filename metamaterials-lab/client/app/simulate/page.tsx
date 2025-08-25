'use client'

import { useState } from 'react'
import axios from 'axios'
import Link from 'next/link'

export default function SimulatePage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [formData, setFormData] = useState({
    name: 'Metamaterial Simulation',
    freqStart: 0.5,
    freqEnd: 2.0,
    numPoints: 100,
    latticeX: 60,
    latticeY: 60,
    thickness: 200,
    epsilonSubstrate: 3.0
  })
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    
    try {
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
        
        <h1 className="text-3xl font-bold mb-8">Run Metamaterial Simulation</h1>
        
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
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Start Freq (THz)</label>
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
              <label className="block text-sm font-medium mb-1">End Freq (THz)</label>
              <input
                type="number"
                step="0.1"
                value={formData.freqEnd}
                onChange={(e) => setFormData({...formData, freqEnd: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Points</label>
              <input
                type="number"
                value={formData.numPoints}
                onChange={(e) => setFormData({...formData, numPoints: parseInt(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                min="10"
                max="500"
                required
              />
            </div>
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
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Thickness (μm)</label>
              <input
                type="number"
                value={formData.thickness}
                onChange={(e) => setFormData({...formData, thickness: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-1">Substrate εr</label>
              <input
                type="number"
                step="0.1"
                value={formData.epsilonSubstrate}
                onChange={(e) => setFormData({...formData, epsilonSubstrate: parseFloat(e.target.value)})}
                className="w-full border rounded px-3 py-2"
                required
              />
            </div>
          </div>
          
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 text-white py-3 rounded-lg hover:opacity-90 disabled:opacity-50 font-semibold"
          >
            {loading ? 'Running Simulation...' : 'Run Simulation'}
          </button>
        </form>
        
        {result && (
          <div className="mt-8 bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-bold mb-4">Simulation Results</h2>
            <div className="space-y-2">
              <p><strong>Job ID:</strong> {result.job_id}</p>
              <p><strong>Status:</strong> <span className="text-green-600">{result.status}</span></p>
              <p><strong>Data Points:</strong> {result.result?.data?.length || 0}</p>
              {result.result?.negative_index_band && (
                <p className="text-purple-600 font-semibold">
                  🎯 Negative Index Band: {result.result.negative_index_band.min_THz.toFixed(2)} - {result.result.negative_index_band.max_THz.toFixed(2)} THz
                </p>
              )}
            </div>
            <Link href={`/results?job_id=${result.job_id}`} 
              className="mt-4 inline-block bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700">
              View Detailed Results →
            </Link>
          </div>
        )}
      </div>
    </div>
  )
}