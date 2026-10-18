import { NavigationClient } from '@/components/NavigationClient'
import Link from 'next/link'
import { ArrowRight, Brain, Zap } from 'lucide-react'
import { SHORT_DISCLAIMER } from '@/lib/performance'

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <NavigationClient currentPage="home" />

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="space-y-4">
              <p className="text-sm font-medium text-primary uppercase tracking-wide">Prototype de recherche</p>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground leading-tight">
                Latéralisation EEG assistée par IA
              </h1>
              <p className="text-lg text-muted-foreground leading-relaxed">
                NeuroLat détecte les crises dans un enregistrement EEG du corpus TUSZ et prédit, pour chaque détection, un côté : gauche, droit ou bilatéral.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-4">
              <Link 
                href="/analyze"
                className="inline-flex items-center justify-center gap-2 px-6 py-3 text-white font-semibold rounded-lg transition-colors"
                style={{ backgroundColor: '#8b5cf6' }}
              >
                Lancer une analyse
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>

            {/* Trust Badges */}
            <div className="grid grid-cols-2 gap-6 pt-8 border-t border-border">
              <div>
                <p className="text-2xl font-bold text-primary">M1</p>
                <p className="text-sm text-muted-foreground">Détection des crises (3 réseaux)</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-primary">M2</p>
                <p className="text-sm text-muted-foreground">Côté prédit : gauche, droit ou bilatéral (3 réseaux)</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-primary">TUSZ</p>
                <p className="text-sm text-muted-foreground">Corpus d&apos;entraînement et d&apos;évaluation (v2.0.3)</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-primary">EEGConformer</p>
                <p className="text-sm text-muted-foreground">Architecture de modèle</p>
              </div>
            </div>
          </div>

          {/* Hero Visual - EEG waves line art */}
          <div className="hidden md:flex items-center justify-center">
            <div className="w-full aspect-square bg-secondary rounded-2xl flex items-center justify-center p-8">
              <svg viewBox="0 0 400 200" className="w-full h-full text-primary">
                <path
                  d="M 10 100 L 30 100 L 40 70 L 50 130 L 60 80 L 70 120 L 80 90 L 90 110 L 100 95 L 110 105 L 120 100 L 130 100 L 140 100 L 150 100"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  opacity="0.3"
                />
                <path
                  d="M 10 120 L 50 120 L 60 90 L 70 150 L 80 100 L 90 140 L 100 110 L 110 130 L 120 115 L 130 125 L 140 120 L 150 120"
                  stroke="currentColor"
                  strokeWidth="2"
                  fill="none"
                  opacity="0.5"
                />
                <path
                  d="M 10 140 L 70 140 L 80 100 L 90 160 L 100 110 L 110 150 L 120 120 L 130 140 L 140 130 L 150 140"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  fill="none"
                />
                <circle cx="20" cy="100" r="3" fill="currentColor" />
                <circle cx="150" cy="140" r="3" fill="currentColor" />
              </svg>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="bg-secondary py-16 md:py-24">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Comment ça marche</h2>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Un pipeline en deux étapes : détection, puis prédiction du côté
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="bg-background rounded-lg p-8 border border-border">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">1</span>
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Upload EEG</h3>
              <p className="text-muted-foreground">
                Déposez un fichier .edf du corpus TUSZ. Le signal est filtré, converti en 18 dérivations bipolaires et découpé en fenêtres de 4 s.
              </p>
            </div>

            {/* Step 2 */}
            <div className="bg-background rounded-lg p-8 border border-border">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">2</span>
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Détection IA</h3>
              <p className="text-muted-foreground">
                Trois réseaux (M1) estiment la probabilité de crise de chaque fenêtre ; une règle figée sur les patients de validation regroupe les fenêtres en détections.
              </p>
            </div>

            {/* Step 3 */}
            <div className="bg-background rounded-lg p-8 border border-border">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">3</span>
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Classification</h3>
              <p className="text-muted-foreground">
                Pour chaque détection, trois réseaux (M2) donnent des probabilités pour gauche, droit et bilatéral ; le côté prédit est la plus forte probabilité moyenne.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Pipeline Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 md:py-24">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Architecture technique</h2>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Le pipeline évalué dans l&apos;article, appliqué tel quel
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-12">
          {/* M1: Detection */}
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-primary rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                <Zap className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-foreground">Étape 1 : Détection de crise</h3>
                <p className="text-muted-foreground mt-2">
                  Le modèle M1 analyse l&apos;enregistrement par fenêtres de 4 s (pas de 2 s) et estime la probabilité de crise de chacune.
                </p>
                <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Entrée : 4 s d&apos;EEG, 18 dérivations bipolaires</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Sortie : Probabilité [0, 1] de présence de crise</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Architecture : EEGConformer, moyenne de 3 graines</li>
                </ul>
              </div>
            </div>
          </div>

          {/* M2: Lateralization */}
          <div className="space-y-4">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 bg-accent rounded-lg flex items-center justify-center flex-shrink-0 mt-1">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-foreground">Étape 2 : Classification latéralisation</h3>
                <p className="text-muted-foreground mt-2">
                  Le modèle M2 reçoit les fenêtres de chaque détection et prédit gauche, droit ou bilatéral.
                </p>
                <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Entrée : fenêtres de 4 s d&apos;une détection</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Sortie : probabilités gauche / droit / bilatéral</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Côté : plus forte probabilité moyenne sur les fenêtres</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid md:grid-cols-4 gap-6 mt-16 pt-16 border-t border-border">
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Corpus</p>
            <p className="text-2xl font-bold text-primary">TUSZ v2.0.3</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Fenêtre</p>
            <p className="text-2xl font-bold text-primary">4 s / 2 s</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Fréquence</p>
            <p className="text-2xl font-bold text-primary">250 Hz</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Dérivations</p>
            <p className="text-2xl font-bold text-primary">18</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-secondary border-t border-border py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-3">
          <p className="font-semibold text-foreground">NeuroLat</p>
          <p className="text-sm text-muted-foreground">{SHORT_DISCLAIMER}</p>
        </div>
      </footer>
    </main>
  )
}
