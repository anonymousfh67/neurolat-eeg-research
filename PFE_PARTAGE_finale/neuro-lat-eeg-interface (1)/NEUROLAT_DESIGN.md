# NeuroLat – EEG Seizure Focus Detection Interface

## Design Vision

NeuroLat breaks away from typical dark-dashboard medical tool aesthetics with an **editorial, scientific minimalism** approach inspired by elegant research papers.

### Visual Direction
- **Palette**: Cream/warm white background (#fdfbf8) with mauve/purple primary (#8b5cf6) and soft pink accent (#ec4899)
- **Typography**: Serif headings (editorial feel) + clean sans-serif body text
- **Layout**: Asymmetric, organic—not rigid grid-based dashboards
- **Visual Metaphor**: Flowing waves and organic circles instead of clichéd glowing neurons

## Interface States

### 1. **Empty State** – Engaging & Inviting
- Large serif headline: "Analyze EEG in moments"
- Asymmetric two-column layout with upload zone on left
- Feature cards on right explaining pipeline stages
- Decorative SVG wavy background (organic, subtle)
- Footer statistics highlighting key metrics

**Key Elements:**
- Brain wave SVG icon in upload zone
- Three info cards: Two-Stage Pipeline, Confidence Scores, Detailed Breakdown
- Bottom stats: 847 windows analyzed, 3s processing time, 94% accuracy

### 2. **Loading State** – Animated & Transparent
- Headline: "Analyzing your brain"
- Centered animated visualization with concentric circles
- Orbiting particles and pulsing wave SVGs
- Two-stage progress bars showing detection vs classification
- Information cards explaining Stage 1 & Stage 2 work

**Key Elements:**
- Concentric rings with 3 rotating circles (orbiting particles)
- Central pulsing brain wave graphic
- Progress bar animation showing real-time progress (0-100%)
- Staggered reveal of Stage 1 & 2 cards

### 3. **Results State** – Clear & Data-Rich
- Primary result card with colored background (rose for right, blue for left, purple for bilateral)
- Large, prominent classification: "Right Focal Origin" / "Left Focal Origin" / "Bilateral"
- Circular confidence gauge (87% in the example)
- Classification breakdown with three cards (L/R/B) showing:
  - Count of windows classified
  - Model confidence score with progress bar
- Analysis summary table + Clinical notes
- "Analyze Another File" and "Export Results" buttons

**Key Elements:**
- Colored container background shifts based on result (rose #ef5350, blue #2196f3, purple #8b5cf6)
- Staggered animation reveal (titles → breakdown → summary → buttons)
- Three-color coded classification cards with badges (L/R/B initials)
- Confidence scores as circular progress indicators
- Professional clinical notes section

## Component Architecture

```
app/page.tsx (main state manager)
├── Header (branding + demo controls)
├── EmptyState
│   ├── FileUpload
│   ├── Feature cards
│   └── Statistics
├── AnalysisState (loading animation)
│   ├── Concentric circles (animated)
│   ├── Progress bars
│   └── Stage info cards
└── ResultsPanel
    ├── Result headline + confidence gauge
    ├── Classification breakdown (3 cards)
    ├── Analysis summary
    ├── Clinical notes
    └── Action buttons
```

## Design Tokens

### Colors
- **Background**: #fdfbf8 (warm cream)
- **Surface**: #ffffff
- **Primary**: #8b5cf6 (mauve/purple)
- **Accent**: #ec4899 (soft pink)
- **Text**: #1a1a1a
- **Text Secondary**: #666666
- **Border**: #e5e1d9

### Typography
- **Headings**: Cambria/Georgia serif (editorial style)
- **Body**: Inter/system sans-serif
- **Line heights**: 1.4-1.6 for body text

### Spacing & Radius
- Uses Tailwind spacing scale (gap-4, p-8, etc.)
- Rounded corners: 16px–32px (rounded-2xl, rounded-3xl)

## Responsive Behavior

- **Mobile-first design**
- Upload zone adapts from side-by-side to stacked
- Feature cards stack vertically on mobile
- Classification cards remain 3-column on desktop, stack on mobile
- All text remains readable at 375px viewport
- Header demo button accessible throughout

## Animation Strategy

### Empty State
- Subtle wavy SVG background (no animation, decorative)

### Loading State
- **Concentric rings**: Outer pulsing (3s), middle rotating (8s), inner pulsing (2s)
- **Orbiting particles**: Three circles rotating at different speeds (6s, 7s, 8s)
- **Wave icon**: Pulsing in center
- **Progress bars**: Smooth width transitions (500ms duration)
- Staggered delays between stage cards

### Results State
- **Cards**: Fade-in + translate-y with delays (0, 200ms, 300ms, 400ms)
- **Confidence circle**: Animated SVG stroke-dashoffset (1000ms)
- **Progress bars**: Animated width transitions

## Demo Controls

Click "Demo" button in top-right header to access demo state switcher:
- **Empty State**: Reset to initial upload screen
- **Loading State**: See animation and progress bars
- **Results State**: View analysis results with confidence breakdowns

This allows easy exploration of all interface states without needing actual EEG file uploads.

## Key Features

✨ **Original Design** – No dark dashboards, no clichés
🎨 **Editorial Aesthetic** – Serif typography, asymmetric layout
📊 **Clear Data Visualization** – Confidence gauges, progress bars, classification breakdown
🚀 **Smooth Animations** – Organic loading visualization, staggered reveals
📱 **Fully Responsive** – Mobile to desktop tested
♿ **Accessible** – Semantic HTML, ARIA labels, readable contrast

## Next Steps for Real Integration

1. Connect to actual EEG API endpoint
2. Replace mock data with real classification results
3. Add file upload validation (.edf format)
4. Implement actual progress tracking from backend
5. Add result export/download functionality
6. Integrate with hospital EHR systems
