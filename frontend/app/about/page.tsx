'use client'

import Navigation from '@/components/Navigation'
import Link from 'next/link'
import { ArrowLeft } from 'lucide-react'
import { PERFORMANCE, SHORT_DISCLAIMER } from '@/lib/performance'

export default function AboutPage() {
  return (
    <main className="min-h-screen bg-background">
      <Navigation currentPage="about" />
      <div className="max-w-4xl mx-auto px-4 py-12 space-y-8">
        <div className="space-y-4">
          <Link href="/" className="inline-flex items-center gap-2 text-primary hover:text-primary/80 transition-colors mb-4">
            <ArrowLeft className="w-4 h-4" />
            Retour
          </Link>
          <h1 className="text-4xl font-bold text-foreground">À propos de NeuroLat</h1>
          <p className="text-lg text-muted-foreground">Un prototype de recherche sur la latéralisation des crises en EEG</p>
        </div>

        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Objectif</h2>
          <p className="text-foreground leading-relaxed">
            NeuroLat accompagne un travail de recherche sur le corpus TUSZ v2.0.3 (Temple University Hospital EEG Seizure Corpus). Il détecte les crises dans un enregistrement EEG, puis prédit pour chaque détection un côté : gauche, droit ou bilatéral. Il applique exactement le pipeline évalué dans l&apos;article.
          </p>
        </div>

        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Ce que valent les résultats</h2>
          <p className="text-foreground leading-relaxed">
            Évaluation sur {PERFORMANCE.corpus}, avec une règle de décision choisie sur les patients de validation puis figée avant l&apos;évaluation. Intervalles à 95 % (bootstrap par patient).
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">{PERFORMANCE.sensitivity}</div>
              <div className="text-xs text-muted-foreground">Sensibilité de détection par crise</div>
            </div>
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">{PERFORMANCE.falseAlarmsPerHour}</div>
              <div className="text-xs text-muted-foreground">Fausses alarmes par heure</div>
            </div>
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">{PERFORMANCE.precision}</div>
              <div className="text-xs text-muted-foreground">Précision de détection</div>
            </div>
            <div className="text-center p-4 rounded bg-background">
              <div className="font-semibold text-primary text-lg">{PERFORMANCE.lateralizationBalancedAccuracy}</div>
              <div className="text-xs text-muted-foreground">Latéralisation de bout en bout, exactitude équilibrée (hasard {PERFORMANCE.lateralizationChance})</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            La détection est utilisable pour la recherche mais produit des fausses alarmes. La latéralisation de bout en bout reste faible : son exactitude simple ({PERFORMANCE.lateralizationAccuracy}) est inférieure au taux de la classe majoritaire ({PERFORMANCE.majorityRate}). Avec une détection parfaite, l&apos;exactitude équilibrée est {PERFORMANCE.oracleBalancedAccuracy}.
          </p>
        </div>

        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Limites</h2>
          <ul className="list-disc pl-5 space-y-2 text-foreground">
            <li>Les labels gauche / droit / bilatéral viennent d&apos;une règle algorithmique appliquée aux annotations par canal de TUSZ ; aucun clinicien ne les a relus.</li>
            <li>Une seule base de données pour l&apos;évaluation principale ; un contrôle externe exploratoire sur Siena (14 patients) donne des résultats mixtes.</li>
            <li>Le pipeline attend le montage des fichiers TUH (canaux « EEG … -REF » ou « -LE ») ; d&apos;autres montages ne sont pas validés.</li>
          </ul>
          <p className="text-sm text-muted-foreground">{SHORT_DISCLAIMER}</p>
        </div>

        <div className="space-y-4 rounded-lg bg-secondary p-8">
          <h2 className="text-2xl font-semibold text-foreground">Données</h2>
          <p className="text-foreground leading-relaxed">
            Le fichier EEG est écrit dans un fichier temporaire sur la machine qui fait tourner le serveur, analysé, puis supprimé. Rien n&apos;est conservé ni envoyé ailleurs. Le serveur n&apos;écoute que sur l&apos;ordinateur local.
          </p>
        </div>

        <div className="text-center py-4">
          <Link href="/analyze" className="inline-flex items-center justify-center px-6 py-3 text-white font-semibold rounded-lg transition-colors" style={{ backgroundColor: '#a64afa' }}>
            Lancer une analyse
          </Link>
        </div>
      </div>
    </main>
  )
}
