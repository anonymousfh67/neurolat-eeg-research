'use client'

import Navigation from '@/components/Navigation'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'

export default function DocsPage() {
  return (
    <main className="min-h-screen bg-background">
      <Navigation currentPage="docs" />
      
      <div className="max-w-4xl mx-auto px-4 py-12 space-y-8">
        {/* Header */}
        <div className="space-y-4">
          <Link href="/" className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors mb-4">
            <ArrowLeft className="w-4 h-4" />
            Retour
          </Link>
          <h1 className="text-4xl font-bold text-foreground">Documentation</h1>
          <p className="text-lg text-muted-foreground">Guide d'utilisation de NeuroLat</p>
        </div>

        {/* Quick Start */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Démarrage Rapide</h2>
          <ol className="space-y-4 text-foreground">
            <li className="flex gap-4">
              <span className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-semibold text-sm">1</span>
              <div>
                <p className="font-semibold mb-1">Accédez à la page d'analyse</p>
                <p className="text-muted-foreground text-sm">Cliquez sur "Analyser" dans la navigation ou sur le bouton "Lancer une analyse" de la page d'accueil.</p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-semibold text-sm">2</span>
              <div>
                <p className="font-semibold mb-1">Téléchargez votre fichier EEG</p>
                <p className="text-muted-foreground text-sm">Sélectionnez un fichier au format EDF (European Data Format) depuis votre ordinateur.</p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-semibold text-sm">3</span>
              <div>
                <p className="font-semibold mb-1">Analysez les résultats</p>
                <p className="text-muted-foreground text-sm">Visualisez les résultats avec la localisation du foyer et la confiance du modèle.</p>
              </div>
            </li>
            <li className="flex gap-4">
              <span className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-semibold text-sm">4</span>
              <div>
                <p className="font-semibold mb-1">Exportez le rapport</p>
                <p className="text-muted-foreground text-sm">Cliquez sur "Voir le rapport détaillé" pour consulter l'analyse complète.</p>
              </div>
            </li>
          </ol>
        </div>

        {/* Understanding Results */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Comprendre les Résultats</h2>
          
          <div className="space-y-4">
            <div className="border-l-4 border-primary pl-4 py-2">
              <h3 className="font-semibold text-foreground mb-2">Foyer Détecté</h3>
              <p className="text-sm text-muted-foreground">
                Indique la localisation de l'activité épileptiforme détectée: foyer gauche, foyer droit, ou bilatéral.
              </p>
            </div>

            <div className="border-l-4 border-primary pl-4 py-2">
              <h3 className="font-semibold text-foreground mb-2">Confiance du Modèle</h3>
              <p className="text-sm text-muted-foreground">
                Pourcentage de confiance du modèle dans sa prédiction. Une valeur plus élevée indique une plus grande certitude.
              </p>
            </div>

            <div className="border-l-4 border-primary pl-4 py-2">
              <h3 className="font-semibold text-foreground mb-2">Crises Détectées</h3>
              <p className="text-sm text-muted-foreground">
                Nombre total de fenêtres d'analyse où une activité épileptiforme a été détectée.
              </p>
            </div>

            <div className="border-l-4 border-primary pl-4 py-2">
              <h3 className="font-semibold text-foreground mb-2">Répartition par Classe</h3>
              <p className="text-sm text-muted-foreground">
                Ventilation des crises détectées par localisation (gauche, droite, bilatéral) en nombre de segments.
              </p>
            </div>

            <div className="border-l-4 border-primary pl-4 py-2">
              <h3 className="font-semibold text-foreground mb-2">Visualisation 3D</h3>
              <p className="text-sm text-muted-foreground">
                Représentation du foyer d'épilepsie sur le cerveau. Le côté coloré en bleu/violet indique la zone affectée.
              </p>
            </div>
          </div>
        </div>

        {/* Technical Specifications */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Spécifications Techniques</h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-semibold text-foreground mb-3">Formats Acceptés</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>✓ EDF (European Data Format)</li>
                <li>✓ EDF+ (avec annotations)</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Paramètres EEG</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Fréquence d'échantillonnage: 250 Hz</li>
                <li>• Canaux: Configuration 19-20 standard</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Durée d'Analyse</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Analyse: &lt;2 secondes</li>
                <li>• Génération rapport: &lt;5 secondes</li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Sécurité</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>• Chiffrement SSL/TLS</li>
                <li>• Suppression automatique après traitement</li>
              </ul>
            </div>
          </div>
        </div>

        {/* FAQ */}
        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Questions Fréquentes</h2>
          
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold text-foreground mb-2">Quels formats EEG sont acceptés?</h3>
              <p className="text-sm text-muted-foreground">
                NeuroLat accepte actuellement les fichiers au format EDF et EDF+. Assurez-vous que vos fichiers EEG ont une fréquence d'échantillonnage de 250 Hz.
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-2">Mes données sont-elles sécurisées?</h3>
              <p className="text-sm text-muted-foreground">
                Oui, toutes les données sont chiffrées en transit et au repos. Les fichiers EEG sont supprimés automatiquement après traitement et ne sont jamais stockés à long terme.
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-2">Quelle est la précision du modèle?</h3>
              <p className="text-sm text-muted-foreground">
                Notre modèle EEGConformer v2.0 a une précision moyenne de 87% sur les ensembles de test. La confiance varie selon la qualité de l'enregistrement EEG.
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-2">Puis-je exporter les résultats?</h3>
              <p className="text-sm text-muted-foreground">
                Oui, vous pouvez consulter et exporter le rapport détaillé en cliquant sur "Voir le rapport détaillé" après l'analyse.
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-2">Qui peut utiliser NeuroLat?</h3>
              <p className="text-sm text-muted-foreground">
                NeuroLat est conçu pour les professionnels de la santé (neurologues, techniciens EEG). Les résultats doivent toujours être interprétés par des cliniciens qualifiés.
              </p>
            </div>
          </div>
        </div>

        {/* Support */}
        <div className="text-center space-y-4 py-8">
          <h3 className="text-xl font-semibold text-foreground">Besoin d'aide supplémentaire?</h3>
          <p className="text-muted-foreground">Consultez notre page À propos ou essayez NeuroLat maintenant.</p>
          <div className="flex justify-center gap-4">
            <Link 
              href="/about"
              className="px-6 py-3 text-primary font-semibold rounded-lg border border-primary hover:bg-primary/10 transition-colors"
            >
              À propos
            </Link>
            <Link 
              href="/analyze"
              className="px-6 py-3 text-white font-semibold rounded-lg transition-colors"
              style={{ backgroundColor: '#a64afa' }}
            >
              Analyser maintenant
            </Link>
          </div>
        </div>
      </div>
    </main>
  )
}
