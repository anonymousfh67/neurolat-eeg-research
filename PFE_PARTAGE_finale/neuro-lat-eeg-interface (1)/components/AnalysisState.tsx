'use client'

import { useEffect, useState } from 'react'

export default function AnalysisState() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 95) return prev
        return prev + Math.random() * 30
      })
    }, 400)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="space-y-16">
      {/* Headline */}
      <div>
        <h2 className="text-5xl md:text-6xl font-serif text-text mb-6">
          Analyzing your brain.
        </h2>
        <p className="text-lg text-text-secondary font-light">
          Our neural network is processing your EEG recording through a two-stage classification pipeline.
        </p>
      </div>

      {/* Main animated visual - organic wave analysis */}
      <div className="relative h-96 flex items-center justify-center">
        {/* Background decorative grid */}
        <div className="absolute inset-0 opacity-10">
          <svg
            className="w-full h-full"
            viewBox="0 0 400 400"
            preserveAspectRatio="none"
          >
            <defs>
              <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
                <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#8b5cf6" strokeWidth="0.5" />
              </pattern>
            </defs>
            <rect width="400" height="400" fill="url(#grid)" />
          </svg>
        </div>

        {/* Animated nested rings */}
        <div className="relative w-64 h-64">
          {/* Outer ring - pulsing */}
          <div className="absolute inset-0 rounded-full border-2 border-primary/30" 
            style={{
              animation: 'pulse 3s ease-in-out infinite'
            }}
          />
          
          {/* Middle ring - rotating */}
          <div className="absolute inset-8 rounded-full border border-accent/40"
            style={{
              animation: 'spin 8s linear infinite'
            }}
          />
          
          {/* Inner circle */}
          <div className="absolute inset-16 rounded-full bg-gradient-to-br from-primary/20 to-accent/10 flex items-center justify-center">
            {/* Central animated wave */}
            <svg
              width="80"
              height="80"
              viewBox="0 0 80 80"
              fill="none"
              style={{
                animation: 'pulse 2s ease-in-out infinite'
              }}
            >
              <path
                d="M 20 40 Q 30 20 40 40 T 60 40"
                stroke="currentColor"
                strokeWidth="3"
                className="text-primary"
                fill="none"
                strokeLinecap="round"
              />
              <circle cx="40" cy="40" r="4" fill="currentColor" className="text-primary" />
            </svg>
          </div>

          {/* Orbiting particles */}
          {[0, 120, 240].map((angle, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 rounded-full bg-primary"
              style={{
                top: '50%',
                left: '50%',
                transform: `rotate(${angle}deg) translateY(-110px)`,
                animation: `spin ${6 + i}s linear infinite`,
                opacity: 0.6
              }}
            />
          ))}
        </div>

        {/* Loading text overlay */}
        <div className="absolute bottom-0 text-center">
          <p className="text-sm text-text-secondary mb-2">Stage 1 & 2 Classification</p>
          <div className="text-xs text-text-tertiary">{Math.round(progress)}% complete</div>
        </div>
      </div>

      {/* Progress visualization - stages */}
      <div className="space-y-6">
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-text">Seizure Detection</span>
            <span className="text-text-secondary">{progress > 50 ? '✓' : 'Processing'}</span>
          </div>
          <div className="h-2 bg-surface-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
              style={{ width: `${Math.min(progress, 50)}%` }}
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span className="font-medium text-text">Focus Classification</span>
            <span className="text-text-secondary">{progress > 75 ? '✓' : 'Queued'}</span>
          </div>
          <div className="h-2 bg-surface-secondary rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-accent to-primary transition-all duration-500"
              style={{ width: `${Math.max(0, progress - 50)}%` }}
            />
          </div>
        </div>
      </div>

      {/* Info cards - what's happening */}
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-surface border border-border rounded-2xl p-6">
          <h4 className="font-serif text-lg text-text mb-2">Stage 1</h4>
          <p className="text-sm text-text-secondary">Segmenting 847 time windows and detecting seizure activity using convolutional neural networks.</p>
        </div>
        <div className="bg-surface border border-border rounded-2xl p-6">
          <h4 className="font-serif text-lg text-text mb-2">Stage 2</h4>
          <p className="text-sm text-text-secondary">Classifying detected seizure windows into left, right, or bilateral focal origin categories.</p>
        </div>
      </div>

      <style jsx>{`
        @keyframes spin {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        @keyframes pulse {
          0%, 100% {
            opacity: 1;
          }
          50% {
            opacity: 0.6;
          }
        }
      `}</style>
    </div>
  )
}
