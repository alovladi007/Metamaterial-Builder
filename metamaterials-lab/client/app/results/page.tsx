'use client'

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
  
  if (loading) return <div className="p-8">Loading results...</div>
  
  if (!data) {
    return (
      <div className="min-h-screen bg-gray-50 p-8">
        <div className="max-w-4xl mx-auto">
          <Link href="/" className="text-blue-600 hover:underline mb-4 inline-block">
            ← Back to Home
          </Link>
          <h1 className="text-3xl font-bold mb-4">No Results Available</h1>
          <p>Please run a simulation first.</p>
          <Link href="/simulate" className="mt-4 inline-block bg-blue-600 text-white px-6 py-2 rounded">
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
        <p className="mb-4 text-gray-600">Job ID: {jobId}</p>
        
        <div className="bg-white rounded-lg shadow overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gradient-to-r from-purple-600 to-blue-600 text-white">
              <tr>
                <th className="px-4 py-3 text-left">Freq (THz)</th>
                <th className="px-4 py-3 text-left">|S11|</th>
                <th className="px-4 py-3 text-left">|S21|</th>
                <th className="px-4 py-3 text-left">Re(n)</th>
                <th className="px-4 py-3 text-left">Re(ε)</th>
                <th className="px-4 py-3 text-left">Re(μ)</th>
              </tr>
            </thead>
            <tbody>
              {data.data.slice(0, 20).map((row: any, i: number) => {
                const S11_mag = Math.sqrt(row.S11_real**2 + row.S11_imag**2)
                const S21_mag = Math.sqrt(row.S21_real**2 + row.S21_imag**2)
                return (
                  <tr key={i} className={i % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                    <td className="px-4 py-2">{(row.frequency / 1e12).toFixed(3)}</td>
                    <td className="px-4 py-2">{S11_mag.toFixed(3)}</td>
                    <td className="px-4 py-2">{S21_mag.toFixed(3)}</td>
                    <td className={`px-4 py-2 ${row.n_real < 0 ? 'text-red-600 font-semibold' : ''}`}>
                      {row.n_real.toFixed(3)}
                    </td>
                    <td className={`px-4 py-2 ${row.eps_real < 0 ? 'text-blue-600' : ''}`}>
                      {row.eps_real.toFixed(3)}
                    </td>
                    <td className={`px-4 py-2 ${row.mu_real < 0 ? 'text-purple-600' : ''}`}>
                      {row.mu_real.toFixed(3)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {data.data.length > 20 && (
            <div className="p-4 text-center text-gray-500 bg-gray-50">
              Showing first 20 of {data.data.length} results
            </div>
          )}
        </div>
        
        <div className="mt-6 text-sm text-gray-600">
          <p>• Red values indicate negative refractive index (n &lt; 0)</p>
          <p>• Blue values indicate negative permittivity (ε &lt; 0)</p>
          <p>• Purple values indicate negative permeability (μ &lt; 0)</p>
        </div>
      </div>
    </div>
  )
}