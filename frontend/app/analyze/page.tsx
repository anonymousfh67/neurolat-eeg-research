'use client'

import { useState } from 'react'
import Link from 'next/link'
import { AlertCircle, FileText, BarChart3, Download, Loader2 } from 'lucide-react'
import { NavigationClient } from '@/components/NavigationClient'
import { BrainVisualization } from '@/components/BrainVisualization'
import { Timeline } from '@/components/Timeline'
import { DetectionsTable } from '@/components/DetectionsTable'
import { API_URL, FOCUS_FR, mmss, type AnalysisResult } from '@/lib/api'
import { SHORT_DISCLAIMER } from '@/lib/performance'

type State = 'empty' | 'loading' | 'complete' | 'error'
const BRAIN = { gauche: 'left', droite: 'right', bilateral: 'bilateral' } as const

export default function AnalyzePage() {
  const [state, setState] = useState<State>('empty')
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [fileInfo, setFileInfo] = useState<{ name: string; sizeMb: string; seconds: string; at: string } | null>(null)
  const [errorMessage, setErrorMessage] = useState('')

  const handleFileSelect = async (file: File) => {
    setState('loading')
    setErrorMessage('')
    const t0 = performance.now()
    const form = new FormData()
    form.append('file', file)
    try {
      const res = await fetch(`${API_URL}/analyser`, { method: 'POST', body: form })
      const data = await res.json()
      if (data.error) {
        setErrorMessage(data.error)
        setState('error')
        return
      }
      setResult(data as AnalysisResult)
      setFileInfo({
        name: file.name,
        sizeMb: (file.size / 1024 / 1024).toFixed(1),
        seconds: ((performance.now() - t0) / 1000).toFixed(1),
        at: new Date().toLocaleString('fr-FR'),
      })
      setState('complete')
    } catch (err) {
      console.error('Erreur appel API NeuroLat :', err)
      setErrorMessage(`Impossible de contacter le serveur d'analyse sur ${API_URL}. Vérifiez que « python app/server.py » tourne dans un terminal.`)
      setState('error')
    }
  }

  const reset = () => {
    setState('empty')
    setResult(null)
    setErrorMessage('')
  }

  const reportText = () => {
    if (!result || !fileInfo) return ''
    const rows = result.detections.map((d, i) =>
      `${String(i + 1).padStart(2)}  ${mmss(d.start_s)}  ${mmss(d.stop_s)}  ${String(Math.round(d.duration_s)).padStart(5)} s  ${FOCUS_FR[d.label].padEnd(9)}  p(G/D/B) = ${d.p_left.toFixed(2)} / ${d.p_right.toFixed(2)} / ${d.p_bilateral.toFixed(2)}`)
    return `RAPPORT D'ANALYSE NEUROLAT
========================================
${SHORT_DISCLAIMER}

Fichier : ${fileInfo.name}
Date : ${fileInfo.at}
Durée de l'enregistrement : ${mmss(result.duration_s)}
Règle de décision (figée sur la validation) : seuil ${result.rule.tau}, au moins ${result.rule.n_min} fenêtre(s), fusion ${result.rule.merge_gap_s} s

DÉTECTIONS (${result.n_detections})
--------------------------------------
${rows.length ? rows.join('\n') : 'Aucune détection.'}
`
  }

  const downloadReport = () => {
    const blob = new Blob([reportText()], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `rapport_neurolat_${(fileInfo?.name ?? 'analyse').replace(/\.edf$/i, '')}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <main className="min-h-screen bg-background">
      <NavigationClient currentPage="analyze" />
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 md:py-24">
        {state === 'empty' && (
          <div className="space-y-8">
            <div className="text-center mb-12">
              <h1 className="text-4xl font-bold text-foreground mb-4">Nouvelle analyse</h1>
              <p className="text-lg text-muted-foreground">Déposez un enregistrement EEG (.edf) pour détecter les crises et prédire leur côté.</p>
            </div>
            <div className="relative">
              <input type="file" accept=".edf" onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])} className="sr-only" id="file-upload" />
              <label htmlFor="file-upload" className="block border-2 border-dashed border-border rounded-xl p-12 text-center cursor-pointer hover:border-primary hover:bg-secondary transition-colors group">
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 bg-secondary rounded-lg flex items-center justify-center"><FileText className="w-8 h-8 text-primary" /></div>
                  <div>
                    <p className="text-lg font-semibold text-foreground">Déposez votre fichier EEG</p>
                    <p className="text-sm text-muted-foreground mt-1">ou cliquez pour sélectionner un fichier .edf</p>
                  </div>
                </div>
              </label>
            </div>
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-sm font-medium text-primary mb-2 uppercase tracking-wide">Format</p>
                <p className="text-foreground font-semibold">EDF du corpus TUSZ</p>
                <p className="text-sm text-muted-foreground mt-2">Canaux nommés « EEG … -REF » ou « -LE » (montage TUH). Au moins 16 des 18 dérivations doivent être exploitables.</p>
              </div>
              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-sm font-medium text-primary mb-2 uppercase tracking-wide">Prétraitement</p>
                <p className="text-foreground font-semibold">250 Hz · 0,5–70 Hz</p>
                <p className="text-sm text-muted-foreground mt-2">Filtre coupe-bande 60 Hz, 18 dérivations bipolaires, fenêtres de 4 s toutes les 2 s.</p>
              </div>
              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-sm font-medium text-primary mb-2 uppercase tracking-wide">Durée</p>
                <p className="text-foreground font-semibold">Selon l&apos;enregistrement</p>
                <p className="text-sm text-muted-foreground mt-2">Elle dépend de la longueur de l&apos;enregistrement et du matériel. Le fichier est supprimé du serveur après l&apos;analyse.</p>
              </div>
            </div>
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-6 flex gap-4">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-amber-900">Prototype de recherche</p>
                <p className="text-sm text-amber-800 mt-1">{SHORT_DISCLAIMER}</p>
              </div>
            </div>
          </div>
        )}

        {state === 'loading' && (
          <div className="text-center py-24 space-y-6">
            <Loader2 className="w-12 h-12 mx-auto text-primary animate-spin" />
            <h2 className="text-2xl font-bold text-foreground">Analyse en cours</h2>
            <p className="text-muted-foreground max-w-md mx-auto">
              Filtrage, découpage en fenêtres, détection (M1, 3 réseaux), puis latéralisation (M2, 3 réseaux). Le temps dépend de la durée de l&apos;enregistrement.
            </p>
          </div>
        )}

        {state === 'complete' && result && fileInfo && (
          <div className="space-y-8">
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-foreground mb-2">Résultats d&apos;analyse</h2>
              <p className="text-muted-foreground">Fichier : {fileInfo.name} ({fileInfo.sizeMb} Mo) | {fileInfo.at}</p>
            </div>

            <div className="rounded-xl p-8" style={{ backgroundColor: '#faf0fb', color: '#0f1419' }}>
              <div className="grid md:grid-cols-2 gap-8 items-center">
                <div>
                  <p className="text-sm font-medium uppercase tracking-wide mb-2" style={{ color: '#2b0336' }}>Côté prédit (résumé)</p>
                  {result.prediction === 'non_detecte' ? (
                    <h3 className="text-3xl font-bold mb-4">Aucune crise détectée</h3>
                  ) : (
                    <h3 className="text-4xl font-bold mb-4">{result.label}</h3>
                  )}
                  {result.labels_differ && (
                    <p className="text-sm mb-4" style={{ color: '#6b7280' }}>Les détections ne prédisent pas toutes le même côté : voir le tableau ci-dessous, qui est la sortie évaluée.</p>
                  )}
                  <div className="grid grid-cols-2 gap-6 mt-6 pt-6" style={{ borderTop: '1px solid #c4b5d0' }}>
                    <div>
                      <p className="text-sm" style={{ color: '#6b7280' }}>Détections</p>
                      <p className="text-3xl font-bold mt-1">{result.n_detections}</p>
                    </div>
                    <div>
                      <p className="text-sm" style={{ color: '#6b7280' }}>Fenêtres détectées</p>
                      <p className="text-3xl font-bold mt-1">{result.n_crises} <span className="text-base font-normal">/ {result.n_fenetres}</span></p>
                    </div>
                  </div>
                </div>
                {result.prediction !== 'non_detecte' && (
                  <div className="flex justify-center">
                    <div className="rounded-xl p-6" style={{ backgroundColor: '#faf0fb' }}>
                      <BrainVisualization focus={BRAIN[result.prediction]} confidence={Math.max(...result.probs)} size="md" />
                    </div>
                  </div>
                )}
              </div>
            </div>

            <section className="space-y-3">
              <h3 className="text-xl font-semibold text-foreground">Chronologie</h3>
              <Timeline result={result} />
              <p className="text-xs text-muted-foreground">{result.pct_flat}% des fenêtres sont plates (signal quasi nul) et ne sont jamais détectées.</p>
            </section>

            <section className="space-y-3">
              <h3 className="text-xl font-semibold text-foreground">Détections</h3>
              <DetectionsTable detections={result.detections} />
              <p className="text-xs text-muted-foreground">
                Une détection regroupe les fenêtres de 4 s dont p(crise) ≥ {result.rule.tau} séparées d&apos;au plus {result.rule.merge_gap_s} s. « Fenêtres ≥ seuil » compte ces fenêtres ; la durée va du début de la première à la fin de la dernière et peut donc inclure des fenêtres intermédiaires sous le seuil. Le côté prédit est la classe de plus forte probabilité moyenne de M2 sur les fenêtres de la détection.
                Les probabilités sont des sorties de réseau, pas des mesures de confiance calibrées.
              </p>
            </section>

            <div className="bg-amber-50 border border-amber-200 rounded-lg p-6 flex gap-4">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div className="space-y-2">
                <p className="font-semibold text-amber-900">Lecture des résultats</p>
                <p className="text-sm text-amber-800">{SHORT_DISCLAIMER}</p>
              </div>
            </div>

            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Link
                href="/report"
                onClick={() => localStorage.setItem('neurolat_report_data', JSON.stringify({ result, fileInfo }))}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 text-white font-semibold rounded-lg transition-colors"
                style={{ backgroundColor: '#a64afa' }}
              >
                <BarChart3 className="w-5 h-5" />
                Voir le rapport détaillé
              </Link>
              <button onClick={downloadReport} className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-secondary text-foreground font-semibold rounded-lg hover:bg-border transition-colors">
                <Download className="w-5 h-5" />
                Télécharger le rapport (texte)
              </button>
              <button onClick={reset} className="inline-flex items-center justify-center px-6 py-3 border border-border text-foreground font-semibold rounded-lg hover:bg-secondary transition-colors">
                Nouvelle analyse
              </button>
            </div>
          </div>
        )}

        {state === 'error' && (
          <div className="max-w-xl mx-auto text-center space-y-6 py-20">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-50">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <h2 className="text-2xl font-bold text-foreground">Analyse impossible</h2>
            <p className="text-muted-foreground">{errorMessage}</p>
            <button onClick={reset} className="inline-flex items-center justify-center px-6 py-3 bg-primary text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors">
              Réessayer
            </button>
          </div>
        )}
      </div>
    </main>
  )
}
