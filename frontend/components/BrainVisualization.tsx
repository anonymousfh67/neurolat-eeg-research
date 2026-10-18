'use client'

type FocusType = 'left' | 'right' | 'bilateral'

interface BrainVisualizationProps {
  focus: FocusType
  confidence?: number
  size?: 'sm' | 'md' | 'lg'
}

export function BrainVisualization({ 
  focus, 
  confidence = 0.9,
  size = 'md'
}: BrainVisualizationProps) {
  // Size mappings
  const sizeMap = {
    sm: { container: 120, viewBox: '0 0 200 240' },
    md: { container: 200, viewBox: '0 0 200 240' },
    lg: { container: 300, viewBox: '0 0 200 240' },
  }

  const config = sizeMap[size]

  // Professional medtech color scheme
  const getColors = () => {
    // Dark purple theme
    const activeBold = '#250042'
    const activeLight = 'rgba(37, 0, 66, 0.25)'
    const inactiveGray = '#d1d5db'
    const inactiveGrayLight = 'rgba(209, 213, 219, 0.08)'

    switch (focus) {
      case 'left':
        return {
          left: activeBold,
          leftLight: activeLight,
          right: inactiveGray,
          rightLight: inactiveGrayLight,
        }
      case 'right':
        return {
          left: inactiveGray,
          leftLight: inactiveGrayLight,
          right: activeBold,
          rightLight: activeLight,
        }
      case 'bilateral':
        return {
          left: activeBold,
          leftLight: activeLight,
          right: activeBold,
          rightLight: activeLight,
        }
      default:
        return {
          left: inactiveGray,
          leftLight: inactiveGrayLight,
          right: inactiveGray,
          rightLight: inactiveGrayLight,
        }
    }
  }

  const colors = getColors()

  return (
    <div className="flex flex-col items-center gap-6">
      <svg
        viewBox={config.viewBox}
        className="filter drop-shadow-sm"
        style={{ width: config.container, height: 'auto' }}
      >
        {/* Professional medical brain visualization */}

        {/* Left hemisphere - background glow */}
        <defs>
          <filter id="glow-left">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
          <filter id="glow-right">
            <feGaussianBlur stdDeviation="2" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Left hemisphere */}
        <g filter={focus === 'left' ? 'url(#glow-left)' : 'none'}>
          <path
            d="M 60 75 Q 45 90 45 110 Q 45 135 60 155 Q 75 163 90 160 L 90 80 Q 78 70 60 75"
            fill={colors.left}
            opacity={focus === 'left' ? '0.9' : '0.6'}
          />
          {/* Subtle folds on left */}
          <path
            d="M 65 90 L 70 95 M 65 110 L 70 115 M 65 130 L 70 135"
            stroke="rgba(255, 255, 255, 0.2)"
            strokeWidth="0.8"
            fill="none"
            opacity="0.5"
          />
        </g>

        {/* Right hemisphere */}
        <g filter={focus === 'right' ? 'url(#glow-right)' : 'none'}>
          <path
            d="M 140 75 Q 155 90 155 110 Q 155 135 140 155 Q 125 163 110 160 L 110 80 Q 122 70 140 75"
            fill={colors.right}
            opacity={focus === 'right' ? '0.9' : '0.6'}
          />
          {/* Subtle folds on right */}
          <path
            d="M 135 90 L 130 95 M 135 110 L 130 115 M 135 130 L 130 135"
            stroke="rgba(255, 255, 255, 0.2)"
            strokeWidth="0.8"
            fill="none"
            opacity="0.5"
          />
        </g>

        {/* Central fissure (professional line) */}
        <line
          x1="100"
          y1="70"
          x2="100"
          y2="165"
          stroke="#e5e7eb"
          strokeWidth="1.5"
          opacity="0.8"
        />

        {/* Cerebellum - minimal */}
        <ellipse
          cx="100"
          cy="190"
          rx="16"
          ry="14"
          fill="#d1d5db"
          opacity="0.4"
        />

        {/* Brain stem */}
        <line
          x1="100"
          y1="165"
          x2="100"
          y2="185"
          stroke="#d1d5db"
          strokeWidth="3"
          opacity="0.5"
          strokeLinecap="round"
        />

        {/* Subtle pulse animation for active focus */}
        {(focus === 'left' || focus === 'bilateral') && (
          <circle
            cx="70"
            cy="110"
            r="38"
            fill="none"
            stroke={colors.left}
            strokeWidth="0.8"
            opacity="0.2"
            style={{
              animation: 'pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            }}
          />
        )}

        {(focus === 'right' || focus === 'bilateral') && (
          <circle
            cx="130"
            cy="110"
            r="38"
            fill="none"
            stroke={colors.right}
            strokeWidth="0.8"
            opacity="0.2"
            style={{
              animation: 'pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite',
            }}
          />
        )}

        <style>{`
          @keyframes pulse {
            0%, 100% { r: 38px; opacity: 0.2; }
            50% { r: 42px; opacity: 0.4; }
          }
        `}</style>
      </svg>

      {/* Clean label and confidence */}
      <div className="text-center space-y-2">
        <div className="text-sm font-medium text-foreground">
          {focus === 'left' && 'Foyer Gauche / Left Focus'}
          {focus === 'right' && 'Foyer Droit / Right Focus'}
          {focus === 'bilateral' && 'Foyer Bilatéral / Bilateral Focus'}
        </div>
        {confidence && (
          <div className="text-xs text-muted-foreground">
            Confiance: {(confidence * 100).toFixed(1)}%
          </div>
        )}
      </div>
    </div>
  )
}
