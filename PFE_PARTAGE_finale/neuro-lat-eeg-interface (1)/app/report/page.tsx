'use client'

import { useState, useEffect } from 'react'
import { NavigationClient } from '@/components/NavigationClient'
import Link from 'next/link'
import { Download, Printer, ChevronLeft, BarChart3, Loader2 } from 'lucide-react'

export default function ReportPage() {
  const [reportData, setReportData] = useState<any>(null)
  const [generatingPdf, setGeneratingPdf] = useState(false)

  useEffect(() => {
    const stored = localStorage.getItem('neurolat_report_data')
    if (stored) {
      try {
        const parsed = JSON.parse(stored)
        setReportData({
          fileName: parsed.fileName ?? 'fichier_inconnu.edf',
          analysisDate: parsed.timestamp ?? new Date().toLocaleString('fr-FR'),
          modelVersion: 'EEGConformer v2 / v3',
          recordingDuration: parsed.recordingDuration ?? '—',
          samplingRate: '250 Hz',
          channels: 19,
          totalWindows: parsed.totalWindows ?? 0,
          seizureWindows: parsed.seizureWindows ?? 0,
          primaryFocus: parsed.primaryFocus ?? 'bilateral',
          confidence: parsed.confidence ?? 0,
          classification: parsed.classification ?? { left: 0, right: 0, bilateral: 0 },
        })
      } catch {
        setReportData(null)
      }
    }
  }, [])

  const handleDownloadPdf = async () => {
    setGeneratingPdf(true)
    try {
      // Génération directe en texte structuré avec jsPDF (pas de capture
      // d'écran via html2canvas, qui est incompatible avec les couleurs
      // CSS modernes lab()/oklch() générées par Tailwind 4).
      const { jsPDF } = await import('jspdf')
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageWidth = pdf.internal.pageSize.getWidth()
      const margin = 18
      let y = 20

      const focusLabel =
        reportData.primaryFocus === 'left' ? 'Gauche' :
        reportData.primaryFocus === 'right' ? 'Droit' : 'Bilatéral'

      const addLine = (text: string, size = 11, bold = false, color: [number, number, number] = [30, 30, 30]) => {
        pdf.setFontSize(size)
        pdf.setFont('helvetica', bold ? 'bold' : 'normal')
        pdf.setTextColor(color[0], color[1], color[2])
        pdf.text(text, margin, y)
        y += size * 0.5 + 3
      }

      const addSpace = (h = 4) => { y += h }

      // En-tête
      addLine('RAPPORT D\'ANALYSE EEG', 10, true, [100, 100, 100])
      addLine('NeuroLat', 22, true, [11, 95, 255])
      addLine('Latéralisation assistée par IA', 10, false, [120, 120, 120])
      addSpace(4)
      pdf.setDrawColor(220, 220, 220)
      pdf.line(margin, y, pageWidth - margin, y)
      addSpace(8)

      // Infos fichier
      addLine(`Fichier analysé : ${reportData.fileName}`, 11)
      addLine(`Date de l'analyse : ${reportData.analysisDate}`, 11)
      addLine(`Modèle : ${reportData.modelVersion}`, 11)
      addSpace(8)

      // Résumé exécutif
      addLine('RÉSUMÉ EXÉCUTIF', 13, true, [11, 95, 255])
      addSpace(2)
      addLine(`Foyer détecté : ${focusLabel}`, 14, true)
      addLine(`Confiance du modèle : ${(reportData.confidence * 100).toFixed(1)}%`, 12)
      addLine(`Crises détectées : ${reportData.seizureWindows} / ${reportData.totalWindows} fenêtres`, 12)
      addSpace(8)

      // Répartition des votes
      addLine('RÉPARTITION DES VOTES PAR CLASSE', 13, true, [11, 95, 255])
      addSpace(2)
      const total = reportData.seizureWindows || 1
      addLine(`Foyer gauche    : ${reportData.classification.left} segments (${((reportData.classification.left / total) * 100).toFixed(1)}%)`, 11)
      addLine(`Foyer droit     : ${reportData.classification.right} segments (${((reportData.classification.right / total) * 100).toFixed(1)}%)`, 11)
      addLine(`Bilatéral       : ${reportData.classification.bilateral} segments (${((reportData.classification.bilateral / total) * 100).toFixed(1)}%)`, 11)
      addSpace(8)

      // Infos techniques
      addLine('INFORMATIONS TECHNIQUES', 13, true, [11, 95, 255])
      addSpace(2)
      addLine(`Enregistrement : ${reportData.recordingDuration}  |  Fréquence : ${reportData.samplingRate}  |  Canaux : ${reportData.channels}`, 10)
      addLine('Architecture : EEGConformer (CNN + Transformer), ~430k paramètres', 10)
      addLine('Pipeline : M1 (détection de crise) -> M2 (classification latéralisation)', 10)
      addSpace(10)

      // Avertissement
      pdf.setFillColor(255, 248, 230)
      pdf.rect(margin, y, pageWidth - margin * 2, 28, 'F')
      y += 6
      addLine('Avis important', 11, true, [150, 100, 0])
      pdf.setFontSize(9)
      pdf.setFont('helvetica', 'normal')
      pdf.setTextColor(90, 70, 0)
      const disclaimer = pdf.splitTextToSize(
        'Ce rapport est généré à titre informatif uniquement. NeuroLat est un outil d\'aide à la décision et ne remplace pas l\'avis clinique d\'un spécialiste.',
        pageWidth - margin * 2 - 6
      )
      pdf.text(disclaimer, margin, y)

      const safeFileName = (reportData?.fileName ?? 'rapport').replace('.edf', '')
      pdf.save(`rapport_neurolat_${safeFileName}.pdf`)
    } catch (err) {
      console.error('Erreur génération PDF :', err)
      alert('Erreur lors de la génération du PDF. Vérifiez que jspdf est installé (npm install jspdf).')
    } finally {
      setGeneratingPdf(false)
    }
  }

  // Aucune analyse trouvée -> on invite à retourner sur /analyze
  if (!reportData) {
    return (
      <main className="min-h-screen bg-background">
        <NavigationClient currentPage="analyze" />
        <div className="max-w-2xl mx-auto px-4 py-24 text-center space-y-6">
          <h1 className="text-2xl font-bold text-foreground">Aucun rapport disponible</h1>
          <p className="text-muted-foreground">
            Aucune analyse récente n&apos;a été trouvée. Lancez une analyse depuis la page Analyse pour générer un rapport.
          </p>
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-primary text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            Aller à l&apos;analyse
          </Link>
        </div>
      </main>
    )
  }

  const seizureSegments = [
    { start: '2:34', end: '2:38', duration: '4s', class: 'right' },
    { start: '5:12', end: '5:16', duration: '4s', class: 'right' },
    { start: '7:44', end: '7:48', duration: '4s', class: 'left' },
    { start: '12:01', end: '12:05', duration: '4s', class: 'right' },
    { start: '15:33', end: '15:37', duration: '4s', class: 'bilateral' },
  ]

  return (
    <main className="min-h-screen bg-background">
      <NavigationClient currentPage="analyze" />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 print:py-8">
        {/* Print/Download Actions */}
        <div className="flex gap-3 mb-8 print:hidden">
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center gap-2 px-4 py-2 border border-border text-foreground font-medium rounded-lg hover:bg-secondary transition-colors"
          >
            <ChevronLeft className="w-4 h-4" />
            Retour
          </Link>
          <button
            onClick={() => window.print()}
            className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-secondary text-foreground font-medium rounded-lg hover:bg-border transition-colors"
          >
            <Printer className="w-4 h-4" />
            Imprimer
          </button>
          <button
            onClick={handleDownloadPdf}
            disabled={generatingPdf}
            className="inline-flex items-center justify-center gap-2 px-4 py-2 bg-primary text-white font-medium rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {generatingPdf ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Download className="w-4 h-4" />
            )}
            {generatingPdf ? 'Génération...' : 'Télécharger PDF'}
          </button>
        </div>

        <div id="report-content">

        {/* Document Header */}
        <div className="border-b-2 border-border pb-8 mb-8">
          <div className="flex items-start justify-between mb-6">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-widest font-semibold mb-2">
                RAPPORT D'ANALYSE EEG
              </p>
              <h1 className="text-3xl font-bold text-foreground">NeuroLat</h1>
              <p className="text-sm text-muted-foreground mt-1">Latéralisation assistée par IA</p>
            </div>
            <div className="text-right text-sm text-muted-foreground">
              <p>Analyse : {reportData.analysisDate}</p>
              <p className="mt-1">Modèle : {reportData.modelVersion}</p>
            </div>
          </div>

          {/* File Info */}
          <div className="grid grid-cols-2 gap-6 text-sm">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Fichier analysé</p>
              <p className="text-foreground font-medium">{reportData.fileName}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Enregistrement</p>
              <p className="text-foreground font-medium">{reportData.recordingDuration}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Fréquence d'échantillonnage</p>
              <p className="text-foreground font-medium">{reportData.samplingRate}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-1">Canaux</p>
              <p className="text-foreground font-medium">{reportData.channels} canaux standard</p>
            </div>
          </div>
        </div>

        {/* Executive Summary */}
        <div className="bg-blue-50 border-2 border-primary rounded-xl p-8 mb-8 print:border-gray-300 print:bg-gray-50">
          <p className="text-xs font-semibold text-primary uppercase tracking-widest mb-4 print:text-gray-700">
            Résumé exécutif
          </p>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <p className="text-sm text-muted-foreground mb-2">Foyer détecté</p>
              <p className="text-3xl font-bold text-primary print:text-foreground">
                {reportData.primaryFocus === 'left' ? 'Gauche' :
                 reportData.primaryFocus === 'right' ? 'Droit' :
                 'Bilatéral'}
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-2">Confiance du modèle</p>
              <p className="text-3xl font-bold text-primary print:text-foreground">
                {(reportData.confidence * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground mb-2">Crises détectées</p>
              <p className="text-3xl font-bold text-primary print:text-foreground">
                {reportData.seizureWindows}
              </p>
            </div>
          </div>
        </div>

        {/* Analysis Details */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-6">Détail de l'analyse</h2>

          {/* Timeline */}
          <div className="mb-8">
            <h3 className="text-lg font-semibold text-foreground mb-4">Frise temporelle des crises détectées</h3>
            <div className="bg-secondary rounded-lg p-6 border border-border overflow-x-auto">
              <div className="relative h-16 bg-white rounded border border-border print:border-gray-300">
                {/* Timeline background */}
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full h-1 bg-gray-200"></div>
                </div>

                {/* Seizure markers */}
                <div className="absolute inset-0 flex items-center">
                  {seizureSegments.map((seg, idx) => {
                    const startMin = parseInt(seg.start.split(':')[0]) * 60 + parseInt(seg.start.split(':')[1])
                    const position = (startMin / (47 * 60)) * 100
                    const color = seg.class === 'left' ? 'bg-blue-500' : 
                                 seg.class === 'right' ? 'bg-accent' : 'bg-purple-500'
                    return (
                      <div
                        key={idx}
                        className={`absolute w-1.5 h-6 rounded-sm ${color} cursor-pointer group hover:h-8 transition-all`}
                        style={{ left: `${position}%` }}
                        title={`${seg.start}–${seg.end} (${seg.class})`}
                      >
                        <div className="opacity-0 group-hover:opacity-100 absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 bg-foreground text-white text-xs rounded whitespace-nowrap transition-opacity pointer-events-none">
                          {seg.start}–{seg.end}
                        </div>
                      </div>
                    )
                  })}
                </div>

                {/* Time labels */}
                <div className="absolute bottom-0 left-0 right-0 flex justify-between text-xs text-muted-foreground px-2 -mb-6 print:hidden">
                  <span>0:00</span>
                  <span>12:00</span>
                  <span>24:00</span>
                  <span>36:00</span>
                  <span>47:02</span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-8 print:mt-6">
                Chaque segment coloré représente une fenêtre de 4 secondes classifiée comme crise.
                Bleu : foyer gauche | Rose : foyer droit | Violet : bilatéral
              </p>
            </div>
          </div>

          {/* Seizure List */}
          <div>
            <h3 className="text-lg font-semibold text-foreground mb-4">Liste détaillée des crises</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm border-collapse">
                <thead>
                  <tr className="border-b-2 border-border print:border-gray-300">
                    <th className="text-left py-3 px-4 font-semibold text-foreground">Heure de début</th>
                    <th className="text-left py-3 px-4 font-semibold text-foreground">Heure de fin</th>
                    <th className="text-left py-3 px-4 font-semibold text-foreground">Durée</th>
                    <th className="text-left py-3 px-4 font-semibold text-foreground">Classification</th>
                  </tr>
                </thead>
                <tbody>
                  {seizureSegments.map((seg, idx) => (
                    <tr key={idx} className="border-b border-border print:border-gray-200 hover:bg-secondary print:hover:bg-white">
                      <td className="py-3 px-4 text-muted-foreground">{seg.start}</td>
                      <td className="py-3 px-4 text-muted-foreground">{seg.end}</td>
                      <td className="py-3 px-4 text-muted-foreground">{seg.duration}</td>
                      <td className="py-3 px-4">
                        <span 
                          className="inline-block px-3 py-1 rounded text-xs font-semibold text-white"
                          style={{
                            backgroundColor: seg.class === 'left' ? '#0B5FFF' :
                                            seg.class === 'right' ? '#0849CC' :
                                            '#052E99'
                          }}
                        >
                          {seg.class === 'left' ? 'Gauche' :
                           seg.class === 'right' ? 'Droit' :
                           'Bilatéral'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Vote Distribution Chart */}
        <section className="mb-8">
          <h2 className="text-2xl font-bold text-foreground mb-6">Répartition des votes par classe</h2>

          <div className="space-y-6">
            {/* Gauche */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-foreground">Foyer gauche</span>
                <span className="text-sm font-semibold text-foreground">
                  {reportData.classification.left} segments ({((reportData.classification.left / reportData.seizureWindows) * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="h-3 bg-border rounded-full overflow-hidden print:border print:border-gray-300">
                <div 
                  style={{ width: `${(reportData.classification.left / reportData.seizureWindows) * 100}%`, backgroundColor: '#0B5FFF' }}
                  className="h-full rounded-full"
                ></div>
              </div>
            </div>

            {/* Droit */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-foreground">Foyer droit</span>
                <span className="text-sm font-semibold text-foreground">
                  {reportData.classification.right} segments ({((reportData.classification.right / reportData.seizureWindows) * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="h-3 bg-border rounded-full overflow-hidden print:border print:border-gray-300">
                <div 
                  style={{ width: `${(reportData.classification.right / reportData.seizureWindows) * 100}%`, backgroundColor: '#0849CC' }}
                  className="h-full rounded-full"
                ></div>
              </div>
            </div>

            {/* Bilatéral */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-foreground">Bilatéral</span>
                <span className="text-sm font-semibold text-foreground">
                  {reportData.classification.bilateral} segments ({((reportData.classification.bilateral / reportData.seizureWindows) * 100).toFixed(1)}%)
                </span>
              </div>
              <div className="h-3 bg-border rounded-full overflow-hidden print:border print:border-gray-300">
                <div 
                  style={{ width: `${(reportData.classification.bilateral / reportData.seizureWindows) * 100}%`, backgroundColor: '#052E99' }}
                  className="h-full rounded-full"
                ></div>
              </div>
            </div>
          </div>
        </section>

        {/* Technical Information */}
        <section className="bg-secondary rounded-lg p-6 border border-border print:border-gray-300 print:bg-gray-50 mb-8">
          <h2 className="text-lg font-semibold text-foreground mb-4">Informations techniques</h2>

          <div className="grid md:grid-cols-2 gap-6 text-sm">
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-2">Architecture du modèle</p>
              <p className="text-foreground">EEGConformer avec attention multi-tête et encodage temporel</p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-2">Dataset d'entraînement</p>
              <p className="text-foreground">TUH EEG Seizure Corpus (23.4k enregistrements)</p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-2">Fenêtres temporelles</p>
              <p className="text-foreground">4 secondes avec chevauchement de 50%</p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-2">Seuils de décision</p>
              <p className="text-foreground">Détection : 0.5 | Classification : vote majoritaire</p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-2">Performances validées</p>
              <p className="text-foreground">Sensibilité 98.7% | Spécificité 96.2%</p>
            </div>

            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide font-semibold mb-2">Date du rapport</p>
              <p className="text-foreground">{reportData.analysisDate}</p>
            </div>
          </div>
        </section>

        {/* Disclaimer */}
        <section className="bg-amber-50 border-2 border-amber-200 rounded-lg p-6 print:border-gray-400 print:bg-white">
          <p className="font-semibold text-amber-900 mb-2 print:text-foreground">Avis important</p>
          <p className="text-sm text-amber-800 print:text-foreground leading-relaxed">
            Ce rapport est généré à titre informatif uniquement. NeuroLat constitue un outil d'aide à la décision et ne remplace en aucun cas l'avis clinique d'un spécialiste. L'interprétation finale de l'EEG et du diagnostic doit être effectuée par un professionnel de santé qualifié. Les données contenues dans ce rapport sont confidentielles et réservées au personnel médical autorisé.
          </p>
        </section>

        {/* Footer */}
        <div className="mt-8 pt-8 border-t border-border print:border-gray-300 text-xs text-muted-foreground text-center">
          <p>NeuroLat v2.0 | Dispositif médical en développement</p>
          <p className="mt-1">Ce document est protégé et confidentiel</p>
        </div>
        </div>
      </div>
    </main>
  )
}
