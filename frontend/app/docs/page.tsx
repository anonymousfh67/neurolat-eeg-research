'use client'

import Navigation from '@/components/Navigation'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import { PERFORMANCE, SHORT_DISCLAIMER } from '@/lib/performance'

const steps = [
  ['Démarrer le serveur', 'Dans un terminal : « python app/server.py ». La pastille en haut à droite passe au vert quand le serveur répond.'],
  ['Déposer un fichier EDF', 'Page Analyse : fichier du corpus TUSZ (canaux « EEG … -REF » ou « -LE »).'],
  ['Lire les résultats', 'Chronologie de la probabilité de crise, tableau des détections avec le côté prédit et les probabilités.'],
  ['Exporter', '« Voir le rapport détaillé » (impression ou PDF) ou téléchargement du rapport texte.'],
]

const faq = [
  ['Quels fichiers sont acceptés ?', 'Des fichiers EDF au format du corpus TUSZ. Le signal est rééchantillonné à 250 Hz. Un enregistrement est refusé si moins de 16 des 18 dérivations bipolaires sont exploitables (signal absent ou écart-type hors de 1–500 µV).'],
  ['Que signifie « côté prédit » ?', 'Pour chaque détection, la classe (gauche, droit, bilatéral) de plus forte probabilité moyenne de M2 sur les fenêtres de la détection. Le résumé en haut de la page d’analyse est la plus forte probabilité moyenne sur toutes les fenêtres détectées ; quand les détections diffèrent, le tableau fait foi.'],
  ['Les probabilités sont-elles des niveaux de confiance ?', 'Non. Ce sont des sorties de réseau de neurones, non calibrées.'],
  ['Quelle est la fiabilité ?', `Sur ${PERFORMANCE.corpus} : sensibilité de détection ${PERFORMANCE.sensitivity}, ${PERFORMANCE.falseAlarmsPerHour} fausses alarmes par heure ; latéralisation de bout en bout, exactitude équilibrée ${PERFORMANCE.lateralizationBalancedAccuracy} (hasard ${PERFORMANCE.lateralizationChance}).`],
  ['Que devient mon fichier ?', 'Il est écrit dans un fichier temporaire sur la machine du serveur, analysé, puis supprimé.'],
  ['Qui peut l’utiliser ?', 'Des chercheurs. Ce n’est pas un dispositif médical et il ne doit pas servir à un diagnostic ou à un traitement.'],
]

export default function DocsPage() {
  return (
    <main className="min-h-screen bg-background">
      <Navigation currentPage="docs" />
      <div className="max-w-4xl mx-auto px-4 py-12 space-y-8">
        <div className="space-y-4">
          <Link href="/" className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors mb-4">
            <ArrowLeft className="w-4 h-4" />
            Retour
          </Link>
          <h1 className="text-4xl font-bold text-foreground">Documentation</h1>
          <p className="text-lg text-muted-foreground">Guide d&apos;utilisation de NeuroLat</p>
        </div>

        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Démarrage rapide</h2>
          <ol className="space-y-4 text-foreground">
            {steps.map(([title, text], i) => (
              <li key={i} className="flex gap-4">
                <span className="flex-shrink-0 w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center font-semibold text-sm">{i + 1}</span>
                <div>
                  <p className="font-semibold mb-1">{title}</p>
                  <p className="text-muted-foreground text-sm">{text}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>

        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Questions fréquentes</h2>
          <div className="space-y-4">
            {faq.map(([q, a]) => (
              <div key={q}>
                <h3 className="font-semibold text-foreground mb-2">{q}</h3>
                <p className="text-sm text-muted-foreground">{a}</p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg bg-amber-50 border border-amber-200 p-6">
          <p className="font-semibold text-amber-900 mb-1">Prototype de recherche</p>
          <p className="text-sm text-amber-800">{SHORT_DISCLAIMER}</p>
        </div>

        <div className="text-center py-4">
          <Link href="/analyze" className="px-6 py-3 text-white font-semibold rounded-lg transition-colors" style={{ backgroundColor: '#a64afa' }}>
            Analyser maintenant
          </Link>
        </div>
      </div>
    </main>
  )
}
