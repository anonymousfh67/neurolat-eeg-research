import { NavigationClient } from '@/components/NavigationClient'
import Link from 'next/link'
import { ArrowRight, Brain, Zap, BarChart3, Shield } from 'lucide-react'

export default function Home() {
  return (
    <main className="min-h-screen bg-background">
      <NavigationClient currentPage="home" />

      {/* Hero Section */}
      <section className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-32">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="space-y-8">
            <div className="space-y-4">
              <p className="text-sm font-medium text-primary uppercase tracking-wide">Diagnostic assisté par IA</p>
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground leading-tight">
                Latéralisation EEG assistée par IA
              </h1>
              <p className="text-lg text-muted-foreground leading-relaxed">
                NeuroLat analyse automatiquement les enregistrements EEG pour détecter et localiser les foyers de crise d&apos;épilepsie avec une précision clinique. Un outil d&apos;aide à la décision pour les neurologues et chercheurs.
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
              <button 
                className="inline-flex items-center justify-center px-6 py-3 border border-border text-foreground font-semibold rounded-lg hover:bg-secondary transition-colors"
              >
                Demander une démo
              </button>
            </div>

            {/* Trust Badges */}
            <div className="grid grid-cols-2 gap-6 pt-8 border-t border-border">
              <div>
                <p className="text-2xl font-bold text-primary">97%</p>
                <p className="text-sm text-muted-foreground">Sensibilité détection</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-primary">96.6%</p>
                <p className="text-sm text-muted-foreground">Spécificité latéralisation</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-primary">TUSZ</p>
                <p className="text-sm text-muted-foreground">Dataset d&apos;entraînement</p>
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
              Un pipeline en deux étapes pour une détection précise et une latéralisation fiable
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
                Déposez votre fichier EEG au format .edf. Le système accepte les multi-canaux et normalise automatiquement les données.
              </p>
            </div>

            {/* Step 2 */}
            <div className="bg-background rounded-lg p-8 border border-border">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">2</span>
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Détection IA</h3>
              <p className="text-muted-foreground">
                Modèle d&apos;apprentissage profond détecte automatiquement les fenêtres temporelles contenant des crises épileptiques.
              </p>
            </div>

            {/* Step 3 */}
            <div className="bg-background rounded-lg p-8 border border-border">
              <div className="w-12 h-12 bg-primary rounded-lg flex items-center justify-center mb-4">
                <span className="text-white font-bold">3</span>
              </div>
              <h3 className="text-lg font-semibold text-foreground mb-2">Classification</h3>
              <p className="text-muted-foreground">
                Pour chaque crise détectée, le système classifie : foyer gauche, droit, ou bilatéral, avec score de confiance.
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
            Pipeline à deux étapes pour une détection fiable et une classification précise
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
                  Le modèle M1 analyse l&apos;enregistrement par fenêtres temporelles (4s) et classifie chaque segment comme "crise" ou "non-crise".
                </p>
                <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Entrée : 4s d&apos;EEG brut multi-canaux</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Sortie : Probabilité [0, 1] de présence de crise</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Architecture : EEGConformer v2.0</li>
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
                  Le modèle M2 reçoit les segments détectés comme "crise" et les classifie en trois catégories de latéralisation.
                </p>
                <ul className="mt-3 space-y-2 text-sm text-muted-foreground">
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Entrée : 4s de segment classé "crise"</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Sortie : Gauche / Droit / Bilatéral + scores</li>
                  <li className="flex items-center gap-2"><span className="text-primary">•</span> Poids du vote : tous les segments classifiés</li>
                </ul>
              </div>
            </div>
          </div>
        </div>

        {/* Key Metrics */}
        <div className="grid md:grid-cols-4 gap-6 mt-16 pt-16 border-t border-border">
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Dataset</p>
            <p className="text-2xl font-bold text-primary">TUH EEG</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Échantillons</p>
            <p className="text-2xl font-bold text-primary">23.4k</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Fréquence</p>
            <p className="text-2xl font-bold text-primary">250 Hz</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Canaux</p>
            <p className="text-2xl font-bold text-primary">19–21</p>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-secondary border-t border-border py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid md:grid-cols-4 gap-8 mb-12">
            <div>
              <p className="font-semibold text-foreground mb-4">NeuroLat</p>
              <p className="text-sm text-muted-foreground">Aide au diagnostic EEG assistée par IA</p>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-4">Produit</p>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><Link href="#" className="hover:text-foreground">Analyse</Link></li>
                <li><Link href="#" className="hover:text-foreground">Documentation</Link></li>
                <li><Link href="#" className="hover:text-foreground">API</Link></li>
              </ul>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-4">Entreprise</p>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><Link href="#" className="hover:text-foreground">À propos</Link></li>
                <li><Link href="#" className="hover:text-foreground">Contact</Link></li>
                <li><Link href="#" className="hover:text-foreground">Carrières</Link></li>
              </ul>
            </div>
            <div>
              <p className="font-semibold text-foreground mb-4">Légal</p>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><Link href="#" className="hover:text-foreground">Conditions</Link></li>
                <li><Link href="#" className="hover:text-foreground">Confidentialité</Link></li>
                <li><Link href="#" className="hover:text-foreground">Conformité</Link></li>
              </ul>
            </div>
          </div>

          <div className="border-t border-border pt-8">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <p className="text-sm text-muted-foreground">
                &copy; 2025 NeuroLat. Tous droits réservés. Dispositif médical en développement.
              </p>
              <p className="text-xs text-muted-foreground">
                <Shield className="w-4 h-4 inline mr-1" />
                Conforme à la directive IVDR
              </p>
            </div>
          </div>
        </div>
      </footer>
    </main>
  )
}
