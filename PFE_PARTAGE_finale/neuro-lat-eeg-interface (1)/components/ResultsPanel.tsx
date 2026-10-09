'use client'

import { useEffect, useState } from 'react'

interface ResultsPanelProps {
  results: {
    fileName: string
    totalWindows: number
    seizureWindows: number
    classification: {
      left: number
      right: number
      bilateral: number
    }
    primaryFocus: 'left' | 'right' | 'bilateral'
    confidence: number
    leftConfidence: number
    rightConfidence: number
    bilateralConfidence: number
  }
  onNewAnalysis: () => void
}

export default function ResultsPanel({ results, onNewAnalysis }: ResultsPanelProps) {
  const [isRevealed, setIsRevealed] = useState(false)

  useEffect(() => {
    // Stagger the reveal animation
    setIsRevealed(true)
  }, [])

  const totalClassified = results.classification.left + results.classification.right + results.classification.bilateral

  const focusLabel = {
    left: 'Left Focal Origin',
    right: 'Right Focal Origin',
    bilateral: 'Bilateral'
  }

  const focusColor = {
    left: 'text-blue-600',
    right: 'text-rose-600',
    bilateral: 'text-purple-600'
  }

  const focusGradient = {
    left: 'from-blue-500 to-blue-600',
    right: 'from-rose-500 to-rose-600',
    bilateral: 'from-purple-500 to-purple-600'
  }

  const focusBg = {
    left: 'bg-blue-50',
    right: 'bg-rose-50',
    bilateral: 'bg-purple-50'
  }

  return (
    <div className="space-y-16">
      {/* Main result - prominent reveal */}
      <div
        className={`transform transition-all duration-700 ${
          isRevealed
            ? 'opacity-100 translate-y-0'
            : 'opacity-0 translate-y-4'
        }`}
      >
        <div className={`rounded-3xl p-12 md:p-16 border border-border ${focusBg[results.primaryFocus]}`}>
          <p className="text-sm font-medium text-text-secondary mb-4 uppercase tracking-wide">Analysis Result</p>
          
          <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-8">
            <div>
              <h2 className={`text-6xl md:text-7xl font-serif font-normal mb-6 ${focusColor[results.primaryFocus]}`}>
                {focusLabel[results.primaryFocus]}
              </h2>
              <p className="text-lg text-text-secondary max-w-xl">
                The model detected <span className="font-medium text-text">{results.seizureWindows} seizure windows</span> out of {results.totalWindows} analyzed, with the primary focus identified as <span className="font-medium">{results.primaryFocus === 'bilateral' ? 'bilateral' : `the ${results.primaryFocus} hemisphere`}</span>.
              </p>
            </div>

            {/* Confidence badge */}
            <div className="flex-shrink-0">
              <div className="relative w-48 h-48 flex items-center justify-center">
                {/* SVG circular progress */}
                <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="text-border"
                  />
                  <circle
                    cx="50"
                    cy="50"
                    r="45"
                    fill="none"
                    stroke="url(#confGradient)"
                    strokeWidth="2"
                    strokeDasharray={`${2 * Math.PI * 45}`}
                    strokeDashoffset={`${2 * Math.PI * 45 * (1 - results.confidence)}`}
                    className="transition-all duration-1000"
                    strokeLinecap="round"
                  />
                  <defs>
                    <linearGradient id="confGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#8b5cf6" />
                      <stop offset="100%" stopColor="#ec4899" />
                    </linearGradient>
                  </defs>
                </svg>
                
                <div className="absolute flex flex-col items-center">
                  <span className="text-5xl font-serif font-normal text-text">
                    {Math.round(results.confidence * 100)}%
                  </span>
                  <span className="text-xs text-text-secondary mt-1 font-medium">CONFIDENCE</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed breakdown section */}
      <div className={`transform transition-all duration-700 delay-200 ${
        isRevealed
          ? 'opacity-100 translate-y-0'
          : 'opacity-0 translate-y-4'
      }`}>
        <div className="mb-8">
          <h3 className="text-2xl font-serif text-text mb-2">Classification Breakdown</h3>
          <p className="text-text-secondary">Distribution of the {totalClassified} detected seizure windows across focal origins</p>
        </div>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Left focus */}
          <div className="bg-surface border border-border rounded-2xl p-8">
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-blue-100">
                <span className="text-blue-600 font-serif text-lg">L</span>
              </div>
            </div>
            <div className="mb-6">
              <div className="text-4xl font-serif text-blue-600 mb-2">
                {results.classification.left}
              </div>
              <p className="text-sm text-text-secondary">windows classified</p>
            </div>
            <div className="pt-6 border-t border-border">
              <p className="text-xs text-text-tertiary mb-2">Model Confidence</p>
              <div className="h-2 bg-surface-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-1000"
                  style={{ width: `${results.leftConfidence * 100}%` }}
                />
              </div>
              <p className="text-sm font-medium text-blue-600 mt-2">
                {Math.round(results.leftConfidence * 100)}%
              </p>
            </div>
          </div>

          {/* Right focus */}
          <div className="bg-surface border border-border rounded-2xl p-8">
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-rose-100">
                <span className="text-rose-600 font-serif text-lg">R</span>
              </div>
            </div>
            <div className="mb-6">
              <div className="text-4xl font-serif text-rose-600 mb-2">
                {results.classification.right}
              </div>
              <p className="text-sm text-text-secondary">windows classified</p>
            </div>
            <div className="pt-6 border-t border-border">
              <p className="text-xs text-text-tertiary mb-2">Model Confidence</p>
              <div className="h-2 bg-surface-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-rose-500 rounded-full transition-all duration-1000"
                  style={{ width: `${results.rightConfidence * 100}%` }}
                />
              </div>
              <p className="text-sm font-medium text-rose-600 mt-2">
                {Math.round(results.rightConfidence * 100)}%
              </p>
            </div>
          </div>

          {/* Bilateral */}
          <div className="bg-surface border border-border rounded-2xl p-8">
            <div className="mb-6">
              <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-purple-100">
                <span className="text-purple-600 font-serif text-lg">B</span>
              </div>
            </div>
            <div className="mb-6">
              <div className="text-4xl font-serif text-purple-600 mb-2">
                {results.classification.bilateral}
              </div>
              <p className="text-sm text-text-secondary">windows classified</p>
            </div>
            <div className="pt-6 border-t border-border">
              <p className="text-xs text-text-tertiary mb-2">Model Confidence</p>
              <div className="h-2 bg-surface-secondary rounded-full overflow-hidden">
                <div
                  className="h-full bg-purple-500 rounded-full transition-all duration-1000"
                  style={{ width: `${results.bilateralConfidence * 100}%` }}
                />
              </div>
              <p className="text-sm font-medium text-purple-600 mt-2">
                {Math.round(results.bilateralConfidence * 100)}%
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Statistics section */}
      <div className={`transform transition-all duration-700 delay-300 ${
        isRevealed
          ? 'opacity-100 translate-y-0'
          : 'opacity-0 translate-y-4'
      }`}>
        <div className="grid md:grid-cols-2 gap-8">
          <div className="bg-surface border border-border rounded-2xl p-8">
            <h4 className="font-serif text-lg text-text mb-6">Analysis Summary</h4>
            <div className="space-y-4">
              <div className="flex justify-between items-start pb-4 border-b border-border">
                <span className="text-text-secondary">File analyzed</span>
                <span className="font-medium text-text text-right max-w-xs break-words">{results.fileName}</span>
              </div>
              <div className="flex justify-between items-center pb-4 border-b border-border">
                <span className="text-text-secondary">Total windows</span>
                <span className="font-medium text-text">{results.totalWindows}</span>
              </div>
              <div className="flex justify-between items-center pb-4 border-b border-border">
                <span className="text-text-secondary">Seizure windows detected</span>
                <span className="font-medium text-text">{results.seizureWindows}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-text-secondary">Detection rate</span>
                <span className="font-medium text-text">
                  {((results.seizureWindows / results.totalWindows) * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          </div>

          <div className="bg-surface border border-border rounded-2xl p-8">
            <h4 className="font-serif text-lg text-text mb-6">Clinical Notes</h4>
            <ul className="space-y-3 text-sm text-text-secondary">
              <li className="flex gap-3">
                <span className="text-primary flex-shrink-0 mt-1">•</span>
                <span>Results represent automated analysis and should be reviewed by a neurologist</span>
              </li>
              <li className="flex gap-3">
                <span className="text-primary flex-shrink-0 mt-1">•</span>
                <span>Confidence scores reflect model certainty; lower scores warrant additional investigation</span>
              </li>
              <li className="flex gap-3">
                <span className="text-primary flex-shrink-0 mt-1">•</span>
                <span>Bilateral classification may indicate multifocal or generalized seizure activity</span>
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div className={`transform transition-all duration-700 delay-400 ${
        isRevealed
          ? 'opacity-100 translate-y-0'
          : 'opacity-0 translate-y-4'
      }`}>
        <div className="flex flex-col sm:flex-row gap-4 pt-8 border-t border-border">
          <button
            onClick={onNewAnalysis}
            className="flex-1 px-8 py-4 bg-primary text-white rounded-xl font-medium hover:bg-primary/90 transition-colors"
          >
            Analyze Another File
          </button>
          <button className="flex-1 px-8 py-4 border border-border text-text rounded-xl font-medium hover:bg-surface-secondary transition-colors">
            Export Results
          </button>
        </div>
      </div>
    </div>
  )
}
