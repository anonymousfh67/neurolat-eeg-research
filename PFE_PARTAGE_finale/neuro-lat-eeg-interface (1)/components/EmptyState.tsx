'use client'

import { useRef } from 'react'
import FileUpload from './FileUpload'

interface EmptyStateProps {
  onFileSelect: (file: File) => void
}

export default function EmptyState({ onFileSelect }: EmptyStateProps) {
  return (
    <div className="space-y-20">
      {/* Headline section - asymmetric layout */}
      <div className="max-w-2xl">
        <h2 className="text-5xl md:text-6xl font-serif text-text leading-tight mb-6">
          Analyze EEG<br />
          <span className="text-primary">in moments.</span>
        </h2>
        <p className="text-lg text-text-secondary font-light leading-relaxed">
          Upload your EEG recording and our two-stage deep learning pipeline will automatically detect seizures and pinpoint their focal origin — left hemisphere, right hemisphere, or bilateral.
        </p>
      </div>

      {/* Main upload area - organic wavy design */}
      <div className="grid md:grid-cols-2 gap-12 items-start">
        {/* Left column - upload zone */}
        <div className="relative">
          {/* Decorative wavy background */}
          <div className="absolute inset-0 -z-10 opacity-20">
            <svg
              viewBox="0 0 400 400"
              className="w-full h-full"
              preserveAspectRatio="none"
            >
              <defs>
                <linearGradient id="waveGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.3" />
                  <stop offset="100%" stopColor="#ec4899" stopOpacity="0.1" />
                </linearGradient>
              </defs>
              <path
                d="M 0,100 Q 100,80 200,100 T 400,100 L 400,0 L 0,0 Z"
                fill="url(#waveGrad)"
              />
              <path
                d="M 0,150 Q 100,130 200,150 T 400,150 L 400,200 L 0,200 Z"
                fill="#8b5cf6"
                opacity="0.05"
              />
            </svg>
          </div>

          <FileUpload onFileSelect={onFileSelect} />
        </div>

        {/* Right column - information cards */}
        <div className="space-y-6">
          {/* Card 1 */}
          <div className="bg-surface border border-border rounded-2xl p-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-lg bg-primary/10">
                  <svg className="h-6 w-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                </div>
              </div>
              <div>
                <h3 className="font-serif text-lg text-text">Two-Stage Pipeline</h3>
                <p className="text-sm text-text-secondary mt-1">First detects seizures, then classifies their focal origin.</p>
              </div>
            </div>
          </div>

          {/* Card 2 */}
          <div className="bg-surface border border-border rounded-2xl p-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-lg bg-accent/10">
                  <svg className="h-6 w-6 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
              </div>
              <div>
                <h3 className="font-serif text-lg text-text">Confidence Scores</h3>
                <p className="text-sm text-text-secondary mt-1">See model confidence for each classification.</p>
              </div>
            </div>
          </div>

          {/* Card 3 */}
          <div className="bg-surface border border-border rounded-2xl p-6">
            <div className="flex gap-4">
              <div className="flex-shrink-0">
                <div className="flex items-center justify-center h-12 w-12 rounded-lg bg-primary/10">
                  <svg className="h-6 w-6 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
              </div>
              <div>
                <h3 className="font-serif text-lg text-text">Detailed Breakdown</h3>
                <p className="text-sm text-text-secondary mt-1">Analyze window-by-window classification votes.</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom info section */}
      <div className="border-t border-border pt-12">
        <div className="grid md:grid-cols-3 gap-8 text-center md:text-left">
          <div>
            <div className="text-3xl font-serif text-primary mb-2">847</div>
            <p className="text-sm text-text-secondary">Windows analyzed per file</p>
          </div>
          <div>
            <div className="text-3xl font-serif text-accent mb-2">3s</div>
            <p className="text-sm text-text-secondary">Average processing time</p>
          </div>
          <div>
            <div className="text-3xl font-serif text-primary mb-2">94%</div>
            <p className="text-sm text-text-secondary">Average model accuracy</p>
          </div>
        </div>
      </div>
    </div>
  )
}
