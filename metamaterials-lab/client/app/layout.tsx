import type { Metadata } from 'next'
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