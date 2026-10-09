'use client'

import Navigation from '@/components/Navigation'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-background">
      <Navigation currentPage="about" />
      
      <div className="max-w-4xl mx-auto px-4 py-12 space-y-8">
        {/* Header */}
        <div className="space-y-4">
          <Link href="/" className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors mb-4">
            <ArrowLeft className="w-4 h-4" />
            Retour
          </Link>
          <h1 className="text-4xl font-bold text-foreground">À propos de NeuroLat</h1>
          <p className="text-lg text-muted-foreground">Comprendre notre mission et notre vision</p>
        </div>

        {/* Mission Section */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Notre Mission</h2>
          <p className="text-foreground leading-relaxed">
            NeuroLat est une plateforme d'analyse EEG avancée utilisant l'intelligence artificielle pour détecter et localiser les foyers d'épilepsie avec une précision clinique. Notre mission est de fournir aux neurologues et aux cliniciens un outil fiable et intuitif pour améliorer le diagnostic et le traitement de l'épilepsie.
          </p>
        </div>

        {/* Vision Section */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Notre Vision</h2>
          <p className="text-foreground leading-relaxed">
            Nous aspirons à transformer la neurologie clinique en mettant la technologie IA au service du diagnostic médical. En développant des outils plus précis et accessibles, nous contribuons à améliorer les résultats pour les patients atteints d'épilepsie dans le monde entier.
          </p>
        </div>

        {/* Key Features Section */}
        <div className="space-y-6">
          <h2 className="text-2xl font-semibold text-foreground">Caractéristiques Principales</h2>
          <div className="grid md:grid-cols-2 gap-6">
            {/* Feature 1 */}
            <div className="rounded-lg bg-secondary p-6 space-y-3">
              <div className="text-2xl font-bold text-primary">🧠</div>
              <h3 className="text-lg font-semibold text-foreground">Analyse IA Avancée</h3>
              <p className="text-sm text-muted-foreground">
                Notre modèle EEGConformer utilise le deep learning pour identifier précisément les patterns épileptiformes.
              </p>
            </div>

            {/* Feature 2 */}
            <div className="rounded-lg bg-secondary p-6 space-y-3">
              <div className="text-2xl font-bold text-primary">📊</div>
              <h3 className="text-lg font-semibold text-foreground">Visualisation Intuitive</h3>
              <p className="text-sm text-muted-foreground">
                Visualisez les foyers d'épilepsie directement sur le cerveau en 3D pour une meilleure compréhension.
              </p>
            </div>

            {/* Feature 3 */}
            <div className="rounded-lg bg-secondary p-6 space-y-3">
              <div className="text-2xl font-bold text-primary">📈</div>
              <h3 className="text-lg font-semibold text-foreground">Rapports Détaillés</h3>
              <p className="text-sm text-muted-foreground">
                Générez des rapports cliniques complets avec statistiques et confiance du modèle.
              </p>
            </div>

            {/* Feature 4 */}
            <div className="rounded-lg bg-secondary p-6 space-y-3">
              <div className="text-2xl font-bold text-primary">🔒</div>
              <h3 className="text-lg font-semibold text-foreground">Sécurité des Données</h3>
              <p className="text-sm text-muted-foreground">
                Vos données EEG sont chiffrées et traitées de manière confidentielle en conformité avec les normes médicales.
              </p>
            </div>
          </div>
        </div>

        {/* Technology Section */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Technologie</h2>
          <p className="text-foreground leading-relaxed mb-4">
            NeuroLat utilise le modèle EEGConformer v2.0, une architecture de deep learning spécialisée dans l'analyse de signaux EEG. Le système a été entraîné sur des milliers d'enregistrements EEG annotés par des experts neurologues pour assurer une précision clinique optimale.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">250 Hz</div>
              <div className="text-xs text-muted-foreground">Fréquence d'échantillonnage</div>
            </div>
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">19</div>
              <div className="text-xs text-muted-foreground">Canaux EEG</div>
            </div>
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">87%</div>
              <div className="text-xs text-muted-foreground">Précision moyenne</div>
            </div>
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">&lt;2s</div>
              <div className="text-xs text-muted-foreground">Temps d'analyse</div>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center space-y-4 py-8">
          <h3 className="text-xl font-semibold text-foreground">Prêt à tester NeuroLat?</h3>
          <Link 
            href="/analyze"
            className="inline-flex items-center justify-center px-6 py-3 text-white font-semibold rounded-lg transition-colors"
            style={{ backgroundColor: '#a64afa' }}
          >
            Lancer une analyse
          </Link>
        </div>
      </div>
    </main>
  )
}
