'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link'
import { Download, Printer, ChevronLeft, Loader2 } from 'lucide-react'
import { NavigationClient } from '@/components/NavigationClient'
import { Timeline } from '@/components/Timeline'
import { DetectionsTable } from '@/components/DetectionsTable'
import { FOCUS_FR, mmss, type AnalysisResult } from '@/lib/api'
import { PERFORMANCE, SHORT_DISCLAIMER } from '@/lib/performance'

interface Stored {
  result: AnalysisResult
  fileInfo: { name: string; sizeMb: string; seconds: string; at: string }
}

export default function ReportPage() {
  const [data, setData] = useState<Stored | null>(null)
  const [generatingPdf, setGeneratingPdf] = useState(false)

  useEffect(() => {
    try {
      const stored = localStorage.getItem('neurolat_report_data')
      if (stored) {
        const parsed = JSON.parse(stored)
        if (parsed?.result?.detections) setData(parsed)
      }
    } catch {
      setData(null)
    }
  }, [])

  const handleDownloadPdf = async () => {
    if (!data) return
    setGeneratingPdf(true)
    try {
      const { jsPDF } = await import('jspdf')
      const { result, fileInfo } = data
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const margin = 18
      let y = 20
      const line = (text: string, size = 10, bold = false) => {
        pdf.setFontSize(size)
        pdf.setFont('helvetica', bold ? 'bold' : 'normal')
        const parts = pdf.splitTextToSize(text, pageWidth - margin * 2)
        if (y + parts.length * (size * 0.5 + 1.5) > 285) { pdf.addPage(); y = 20 }
        pdf.text(parts, margin, y)
        y += parts.length * (size * 0.5 + 1.5) + 1
      }
      line("RAPPORT D'ANALYSE EEG - NeuroLat", 14, true)
      line(SHORT_DISCLAIMER, 9)
      y += 3
      line(`Fichier : ${fileInfo.name}`)
      line(`Date : ${fileInfo.at}`)
      line(`Durée : ${mmss(result.duration_s)} | fenêtres : ${result.n_fenetres} | fenêtres plates : ${result.pct_flat}%`)
      line(`Règle figée : seuil ${result.rule.tau}, au moins ${result.rule.n_min} fenêtre(s), fusion ${result.rule.merge_gap_s} s`)
      y += 3
      line(`Détections : ${result.n_detections}`, 12, true)
      result.detections.forEach((d, i) => {
        line(`${i + 1}. ${mmss(d.start_s)} - ${mmss(d.stop_s)} (${Math.round(d.duration_s)} s, ${d.n_windows} fenêtres) : ${FOCUS_FR[d.label]}  p(G/D/B) = ${d.p_left.toFixed(2)} / ${d.p_right.toFixed(2)} / ${d.p_bilateral.toFixed(2)}`)
      })
      if (!result.detections.length) line('Aucune détection.')
      y += 3
      line('Performances de référence', 12, true)
      line(`${PERFORMANCE.corpus}. Sensibilité de détection ${PERFORMANCE.sensitivity} ; fausses alarmes par heure ${PERFORMANCE.falseAlarmsPerHour} ; précision ${PERFORMANCE.precision}. Latéralisation de bout en bout : exactitude équilibrée ${PERFORMANCE.lateralizationBalancedAccuracy} (hasard ${PERFORMANCE.lateralizationChance}) ; exactitude ${PERFORMANCE.lateralizationAccuracy}, inférieure au taux de la classe majoritaire (${PERFORMANCE.majorityRate}).`, 9)
      pdf.save(`rapport_neurolat_${fileInfo.name.replace(/\.edf$/i, '')}.pdf`)
    } catch (err) {
      console.error('Erreur génération PDF :', err)
      alert('Erreur lors de la génération du PDF.')
    } finally {
      setGeneratingPdf(false)
    }
  }

  if (!data) {
    return (
      <main className="min-h-screen bg-background">
        <NavigationClient currentPage="analyze" />
        <div className="max-w-2xl mx-auto px-4 py-24 text-center space-y-6">
          <h1 className="text-2xl font-bold text-foreground">Aucun rapport disponible</h1>
          <p className="text-muted-foreground">Lancez une analyse depuis la page Analyse pour générer un rapport.</p>
          <Link href="/analyze" className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors">
            <ChevronLeft className="w-4 h-4" />
            Aller à l&apos;analyse
          </Link>
        </div>
      </main>
    )
  }

  const { result, fileInfo } = data
  return (
    <main className="min-h-screen bg-background">
      <NavigationClient currentPage="analyze" />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 print:py-8">
        <div className="flex gap-3 mb-8 print:hidden">
          <Link href="/analyze" className="inline-flex items-center justify-center gap-2 px-4 py-2 border border-border text-foreground font-medium rounded-lg hover:bg-secondary transition-colors">
            <ChevronLeft className="w-4 h-4" />
            Retour
          </Link>
          <button onClick={() => window.print()} className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-secondary text-foreground font-medium rounded-lg hover:bg-border transition-colors">
            <Printer className="w-4 h-4" />
            Imprimer
          </button>
          <button onClick={handleDownloadPdf} disabled={generatingPdf} className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary text-white font-medium rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50">
            {generatingPdf ? <Loader2 className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
            {generatingPdf ? 'Génération...' : 'Télécharger PDF'}
          </button>
        </div>

        <div id="report-content" className="space-y-8">
          <div className="border-b-2 border-border pb-6">
            <p className="text-xs text-muted-foreground uppercase tracking-widest font-semibold mb-2">Rapport d&apos;analyse EEG</p>
            <h1 className="text-3xl font-bold text-foreground">NeuroLat</h1>
            <div className="grid grid-cols-2 gap-4 text-sm mt-4">
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Fichier</p><p className="font-medium">{fileInfo.name}</p></div>
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Date</p><p className="font-medium">{fileInfo.at}</p></div>
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Durée de l&apos;enregistrement</p><p className="font-medium">{mmss(result.duration_s)}</p></div>
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Fenêtres (4 s, pas de 2 s)</p><p className="font-medium">{result.n_fenetres} dont {result.pct_flat}% plates</p></div>
            </div>
          </div>

          <div className="bg-amber-50 border-2 border-amber-200 rounded-lg p-5 print:bg-white print:border-gray-400">
            <p className="font-semibold text-amber-900 mb-1">Prototype de recherche</p>
            <p className="text-sm text-amber-800 leading-relaxed">{SHORT_DISCLAIMER}</p>
          </div>

          <section>
            <h2 className="text-2xl font-bold text-foreground mb-4">Résultat</h2>
            <p className="text-foreground">
              {result.n_detections === 0
                ? 'Aucune détection avec la règle appliquée.'
                : `${result.n_detections} détection(s), ${result.n_crises} fenêtre(s) détectée(s) sur ${result.n_fenetres}. La sortie évaluée est le côté prédit pour chaque détection (tableau).`}
            </p>
          </section>

          <section className="space-y-3">
            <h3 className="text-lg font-semibold text-foreground">Chronologie</h3>
            <Timeline result={result} />
          </section>

          <section className="space-y-3">
            <h3 className="text-lg font-semibold text-foreground">Liste des détections</h3>
            <DetectionsTable detections={result.detections} />
          </section>

          <section className="bg-secondary rounded-lg p-6 border border-border print:bg-white">
            <h2 className="text-lg font-semibold text-foreground mb-4">Méthode</h2>
            <ul className="text-sm text-foreground space-y-2 list-disc pl-5">
              <li>Prétraitement : 18 dérivations bipolaires, 250 Hz, filtre 0,5–70 Hz, coupe-bande 60 Hz ; enregistrement rejeté si moins de 16 dérivations valides.</li>
              <li>Détection : M1 (EEGConformer, moyenne de 3 graines) ; fenêtre détectée si p(crise) ≥ {result.rule.tau} et non plate ; fenêtres séparées de moins de {result.rule.merge_gap_s} s fusionnées ; détection gardée si elle compte au moins {result.rule.n_min} fenêtre(s).</li>
              <li>Côté : M2 (EEGConformer, moyenne de 3 graines) ; classe de plus forte probabilité moyenne sur les fenêtres de la détection.</li>
              <li>Règle de décision choisie sur les patients de validation, puis figée avant l&apos;évaluation.</li>
            </ul>
          </section>

          <section className="bg-secondary rounded-lg p-6 border border-border print:bg-white">
            <h2 className="text-lg font-semibold text-foreground mb-4">Performances de référence ({PERFORMANCE.corpus})</h2>
            <div className="grid md:grid-cols-2 gap-4 text-sm">
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Sensibilité de détection (par crise)</p><p>{PERFORMANCE.sensitivity}</p></div>
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Fausses alarmes par heure</p><p>{PERFORMANCE.falseAlarmsPerHour}</p></div>
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Précision de détection</p><p>{PERFORMANCE.precision}</p></div>
              <div><p className="text-xs text-muted-foreground uppercase font-semibold">Latéralisation de bout en bout : exactitude équilibrée</p><p>{PERFORMANCE.lateralizationBalancedAccuracy} (hasard {PERFORMANCE.lateralizationChance})</p></div>
            </div>
            <p className="text-xs text-muted-foreground mt-4">
              Intervalles à 95 % (bootstrap par patient). L&apos;exactitude simple de la latéralisation de bout en bout ({PERFORMANCE.lateralizationAccuracy}) est inférieure au taux de la classe majoritaire ({PERFORMANCE.majorityRate}) ; avec une détection parfaite, l&apos;exactitude équilibrée est {PERFORMANCE.oracleBalancedAccuracy}.
              Les labels de référence viennent d&apos;une règle algorithmique sur les annotations par canal et n&apos;ont pas été relus par un clinicien.
            </p>
          </section>
        </div>
      </div>
    </main>
  )
}
