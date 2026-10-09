import React from 'react'

interface HeaderProps {
  onDemoEmpty?: () => void
  onDemoLoading?: () => void
  onDemoResults?: () => void
}

export default function Header({ onDemoEmpty, onDemoLoading, onDemoResults }: HeaderProps) {
  const [showDemo, setShowDemo] = React.useState(false)

  return (
    <header className="border-b border-border bg-surface">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 md:py-8">
        <div className="flex items-baseline justify-between gap-3">
          <div className="flex items-baseline gap-3">
            {/* Logo/Brand mark */}
            <div className="flex items-center justify-center w-10 h-10 rounded-full bg-gradient-to-br from-primary to-accent">
              <svg
                className="w-6 h-6 text-white"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            
            <div>
              <h1 className="text-3xl md:text-4xl font-serif font-normal text-text">
                NeuroLat
              </h1>
              <p className="text-sm text-text-secondary font-normal mt-1">
                Seizure Focus Detection
              </p>
            </div>
          </div>

          {/* Demo button in top right */}
          <div className="relative">
            <button
              onClick={() => setShowDemo(!showDemo)}
              className="text-xs font-medium text-text-secondary hover:text-text px-2 py-1"
              title="Show demo controls"
            >
              Demo
            </button>

            {showDemo && (
              <div className="absolute top-full right-0 mt-2 bg-white border border-border rounded-lg shadow-lg p-3 space-y-2 z-50">
                <p className="text-xs font-semibold text-text-secondary">DEMO STATES</p>
                <button
                  onClick={() => {
                    onDemoEmpty?.()
                    setShowDemo(false)
                  }}
                  className="block w-full text-left px-3 py-2 text-sm bg-surface-secondary hover:bg-border rounded text-text"
                >
                  Empty State
                </button>
                <button
                  onClick={() => {
                    onDemoLoading?.()
                    setShowDemo(false)
                  }}
                  className="block w-full text-left px-3 py-2 text-sm bg-primary/10 hover:bg-primary/20 rounded text-primary"
                >
                  Loading State
                </button>
                <button
                  onClick={() => {
                    onDemoResults?.()
                    setShowDemo(false)
                  }}
                  className="block w-full text-left px-3 py-2 text-sm bg-accent/10 hover:bg-accent/20 rounded text-accent"
                >
                  Results State
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
