'use client'

import { useState } from 'react'
import { NavigationClient } from '@/components/NavigationClient'
import { BrainVisualization } from '@/components/BrainVisualization'
import Link from 'next/link'
import { AlertCircle, CheckCircle2, FileText, BarChart3, Download } from 'lucide-react'

// URL du backend Flask local. Pas besoin de la changer entre les
// sessions puisque tout tourne en local (contrairement à ngrok).
const API_URL = 'http://localhost:5000'

export default function AnalyzePage() {
  const [analysisState, setAnalysisState] = useState<'empty' | 'loading' | 'complete' | 'error'>('empty')
  const [results, setResults] = useState<any>(null)
  const [errorMessage, setErrorMessage] = useState<string>('')

  const handleFileSelect = async (file: File) => {
    setAnalysisState('loading')
    setErrorMessage('')

    const startTime = performance.now()
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_URL}/analyser`, {
        method: 'POST',
        body: formData,
      })

      const data = await res.json()

      if (data.error) {
        setErrorMessage(data.error)
        setAnalysisState('error')
        return
      }

      const elapsedSeconds = ((performance.now() - startTime) / 1000).toFixed(1)

      // Le backend renvoie des clés FR (gauche/droite/bilateral) et un
      // format différent de l'UI (left/right/bilateral) — on traduit ici.
      const [pGauche, pDroite, pBilateral] =
        data.probs && data.probs.length === 3 ? data.probs : [0, 0, 0]

      const focusMap: Record<string, string> = {
        gauche: 'left',
        droite: 'right',
        bilateral: 'bilateral',
        non_detecte: 'bilateral',
      }

      const realResults = {
        fileName: file.name,
        fileSize: (file.size / 1024 / 1024).toFixed(2),
        analysisTime: `${elapsedSeconds}s`,
        totalWindows: data.n_fenetres,
        seizureWindows: data.n_crises,
        recordingDuration: '—', // non fourni par le backend actuel
        classification: {
          left: data.votes?.gauche ?? 0,
          right: data.votes?.droite ?? 0,
          bilateral: data.votes?.bilateral ?? 0,
        },
        primaryFocus: focusMap[data.prediction] ?? 'bilateral',
        confidence: Math.max(pGauche, pDroite, pBilateral) || 0,
        leftConfidence: pGauche,
        rightConfidence: pDroite,
        bilateralConfidence: pBilateral,
        timestamp: new Date().toLocaleString('fr-FR'),
      }

      setResults(realResults)
      setAnalysisState('complete')
    } catch (err) {
      console.error('Erreur appel API NeuroLat :', err)
      setErrorMessage(
        'Impossible de contacter le serveur d\'analyse sur ' + API_URL +
        '. Vérifiez que "python app.py" tourne bien dans un terminal.'
      )
      setAnalysisState('error')
    }
  }

  const handleReset = () => {
    setAnalysisState('empty')
    setResults(null)
    setErrorMessage('')
  }

  return (
    <main className="min-h-screen bg-background">
      <NavigationClient currentPage="analyze" />

      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-16 md:py-24">
        {/* Empty State */}
        {analysisState === 'empty' && (
          <div className="space-y-8">
            <div className="text-center mb-12">
              <h1 className="text-4xl font-bold text-foreground mb-4">Nouvelle analyse</h1>
              <p className="text-lg text-muted-foreground">
                Déposez un fichier EEG pour détecter et localiser les foyers de crises d&apos;épilepsie
              </p>
            </div>

            {/* Upload Zone */}
            <div className="relative">
              <input
                type="file"
                accept=".edf"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
                className="sr-only"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="block border-2 border-dashed border-border rounded-xl p-12 text-center cursor-pointer hover:border-primary hover:bg-secondary transition-colors group"
              >
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 bg-secondary rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                    <FileText className="w-8 h-8 text-primary" />
                  </div>
                  <div>
                    <p className="text-lg font-semibold text-foreground">
                      Déposez votre fichier EEG
                    </p>
                    <p className="text-sm text-muted-foreground mt-1">
                      ou cliquez pour sélectionner un fichier .edf
                    </p>
                  </div>
                </div>
              </label>
            </div>

            {/* Info Cards */}
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-sm font-medium text-primary mb-2 uppercase tracking-wide">Format supporté</p>
                <p className="text-foreground font-semibold">Fichier EEG (.edf)</p>
                <p className="text-sm text-muted-foreground mt-2">Format standard pour les enregistrements EEG multi-canaux</p>
              </div>

              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-sm font-medium text-primary mb-2 uppercase tracking-wide">Spécifications</p>
                <p className="text-foreground font-semibold">250 Hz – 19–21 canaux</p>
                <p className="text-sm text-muted-foreground mt-2">Fréquence d&apos;échantillonnage standard pour analyses cliniques</p>
              </div>

              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-sm font-medium text-primary mb-2 uppercase tracking-wide">Durée de l&apos;analyse</p>
                <p className="text-foreground font-semibold">&lt; 5 secondes</p>
                <p className="text-sm text-muted-foreground mt-2">Traitement rapide et résultats immédiats</p>
              </div>
            </div>

            {/* Disclaimer */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 flex gap-4">
              <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-blue-900">Avis important</p>
                <p className="text-sm text-blue-800 mt-1">
                  NeuroLat est un outil d&apos;aide à la décision et ne remplace pas l&apos;avis clinique d&apos;un spécialiste. Tous les résultats doivent être validés par un professionnel de santé qualifié.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {analysisState === 'loading' && (
          <div className="space-y-12">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-foreground mb-2">Analyse en cours</h2>
              <p className="text-muted-foreground">Votre enregistrement EEG est traité...</p>
            </div>

            {/* Pipeline Steps */}
            <div className="space-y-6">
              {/* Step 1: Detection */}
              <div className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
                    <CheckCircle2 className="w-6 h-6 text-white" />
                  </div>
                  <div className="w-0.5 h-16 bg-border my-2"></div>
                </div>
                <div className="flex-1 pb-8">
                  <p className="font-semibold text-foreground">Étape 1 : Détection de crise</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Segmentation et analyse par fenêtres temporelles
                  </p>
                  <div className="mt-3 flex items-center gap-2">
                    <div className="flex-1 h-1 bg-border rounded-full overflow-hidden">
                      <div className="h-full w-full bg-primary rounded-full"></div>
                    </div>
                    <span className="text-xs text-muted-foreground font-medium">100%</span>
                  </div>
                </div>
              </div>

              {/* Step 2: Classification */}
              <div className="flex gap-4">
                <div className="flex flex-col items-center">
                  <div className="w-10 h-10 bg-accent rounded-full flex items-center justify-center flex-shrink-0 animate-pulse">
                    <div className="w-4 h-4 bg-white rounded-full"></div>
                  </div>
                  <div className="w-0.5 h-16 bg-border my-2"></div>
                </div>
                <div className="flex-1 pb-8">
                  <p className="font-semibold text-foreground">Étape 2 : Classification latéralisation</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Détermination du foyer (gauche, droit, bilatéral)
                  </p>
                  <div className="mt-3 flex items-center gap-2">
                    <div className="flex-1 h-1 bg-border rounded-full overflow-hidden">
                      <div className="h-full w-2/3 bg-accent rounded-full animate-pulse"></div>
                    </div>
                    <span className="text-xs text-muted-foreground font-medium">67%</span>
                  </div>
                </div>
              </div>

              {/* Step 3: Report */}
              <div className="flex gap-4">
                <div className="w-10 h-10 bg-border rounded-full flex items-center justify-center flex-shrink-0">
                  <span className="text-xs text-muted-foreground font-medium">3</span>
                </div>
                <div className="flex-1">
                  <p className="font-semibold text-muted-foreground">Génération du rapport</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    Compilation des résultats et métriques
                  </p>
                </div>
              </div>
            </div>

            {/* Wave Animation */}
            <div className="flex justify-center gap-2 py-8">
              <div className="w-1 h-8 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0s' }}></div>
              <div className="w-1 h-12 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-1 h-8 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
            </div>
          </div>
        )}

        {/* Results State */}
        {analysisState === 'complete' && results && (
          <div className="space-y-8">
            {/* Header */}
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-foreground mb-2">Résultats d&apos;analyse</h2>
              <p className="text-muted-foreground">
                Fichier : {results.fileName} | {results.timestamp}
              </p>
            </div>

            {/* Main Result Card with Brain Visualization */}
            <div 
              className="rounded-xl p-8"
              style={{
                backgroundColor: '#faf0fb',
                color: '#0f1419'
              }}
            >
              <div className="grid md:grid-cols-2 gap-8 items-center">
                {/* Text content */}
                <div>
                  <p className="text-sm font-medium uppercase tracking-wide mb-2" style={{ color: '#2b0336' }}>Foyer détecté / Detected Focus</p>
                  <h3 className="text-4xl font-bold mb-4" style={{ color: '#0f1419' }}>
                    {results.primaryFocus === 'left' ? 'Foyer gauche / Left Focus' :
                     results.primaryFocus === 'right' ? 'Foyer droit / Right Focus' :
                     'Bilatéral / Bilateral'}
                  </h3>
                  
                  <div className="grid grid-cols-2 gap-6 mt-6 pt-6" style={{ borderTopColor: '#0B5FFF', borderTopWidth: '1px' }}>
                    <div>
                      <p className="text-sm" style={{ color: '#6b7280' }}>Confiance du modèle</p>
                      <p className="text-3xl font-bold mt-1" style={{ color: '#0f1419' }}>{(results.confidence * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-sm" style={{ color: '#6b7280' }}>Crises détectées</p>
                      <p className="text-3xl font-bold mt-1" style={{ color: '#0f1419' }}>{results.seizureWindows}</p>
                    </div>
                  </div>
                </div>

                {/* Brain Visualization */}
                <div className="flex justify-center">
                  <div className="rounded-xl p-6" style={{ backgroundColor: '#faf0fb' }}>
                    <BrainVisualization 
                      focus={results.primaryFocus}
                      confidence={results.confidence}
                      size="md"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Metrics Grid */}
            <div className="grid md:grid-cols-3 gap-6">
              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-xs text-muted-foreground uppercase tracking-wide font-medium mb-2">Fenêtres analysées</p>
                <p className="text-3xl font-bold text-foreground">{results.totalWindows}</p>
                <p className="text-sm text-muted-foreground mt-2">Segments de 4s chacun</p>
              </div>

              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-xs text-muted-foreground uppercase tracking-wide font-medium mb-2">Durée enregistrement</p>
                <p className="text-3xl font-bold text-foreground">{results.recordingDuration}</p>
                <p className="text-sm text-muted-foreground mt-2">Temps total d&apos;analyse</p>
              </div>

              <div className="bg-secondary rounded-lg p-6 border border-border">
                <p className="text-xs text-muted-foreground uppercase tracking-wide font-medium mb-2">Temps de traitement</p>
                <p className="text-3xl font-bold text-foreground">{results.analysisTime}</p>
                <p className="text-sm text-muted-foreground mt-2">Pipeline complet</p>
              </div>
            </div>

            {/* Vote Distribution */}
            <div className="bg-secondary rounded-lg p-8 border border-border">
              <h4 className="font-semibold text-foreground mb-6">Répartition des votes par classe</h4>
              
              <div className="space-y-4">
                {/* Left */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-foreground">Foyer gauche</span>
                    <span className="text-sm font-semibold text-foreground">{results.classification.left} segments</span>
                  </div>
                  <div className="w-full h-2 bg-border rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${(results.classification.left / results.seizureWindows) * 100}%`, backgroundColor: '#630bff' }}
                    ></div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{(results.leftConfidence * 100).toFixed(1)}% confiance</p>
                </div>

                {/* Right */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-foreground">Foyer droit</span>
                    <span className="text-sm font-semibold text-foreground">{results.classification.right} segments</span>
                  </div>
                  <div className="w-full h-2 bg-border rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${(results.classification.right / results.seizureWindows) * 100}%`, backgroundColor: '#cc0839' }}
                    ></div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{(results.rightConfidence * 100).toFixed(1)}% confiance</p>
                </div>

                {/* Bilateral */}
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-foreground">Bilatéral</span>
                    <span className="text-sm font-semibold text-foreground">{results.classification.bilateral} segments</span>
                  </div>
                  <div className="w-full h-2 bg-border rounded-full overflow-hidden">
                    <div 
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${(results.classification.bilateral / results.seizureWindows) * 100}%`, backgroundColor: '#052E99' }}
                    ></div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{(results.bilateralConfidence * 100).toFixed(1)}% confiance</p>
                </div>
              </div>
            </div>

            {/* Warning */}
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-6 flex gap-4">
              <AlertCircle className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold text-amber-900">Remarque clinique</p>
                <p className="text-sm text-amber-800 mt-1">
                  Cet outil est une aide à la décision. Les résultats doivent être interprétés par un professionnel de santé qualifié en contexte clinique complet.
                </p>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col sm:flex-row gap-4 pt-4">
              <Link
                href="/report"
                onClick={() => {
                  // On transmet le vrai résultat à la page /report via
                  // localStorage, car Next.js charge /report comme une
                  // page neuve qui n'a pas accès aux variables de /analyze.
                  localStorage.setItem('neurolat_report_data', JSON.stringify(results))
                }}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 text-white font-semibold rounded-lg transition-colors"
                style={{ backgroundColor: '#a64afa' }}
              >
                <BarChart3 className="w-5 h-5" />
                Voir le rapport détaillé
              </Link>
              <button
                onClick={() => {
                  const reportText = `RAPPORT D'ANALYSE NEUROLAT
========================================
Fichier analysé : ${results.fileName}
Date de l'analyse : ${results.timestamp}

RÉSULTAT
--------
Foyer détecté : ${results.primaryFocus === 'left' ? 'Gauche' : results.primaryFocus === 'right' ? 'Droit' : 'Bilatéral'}
Confiance du modèle : ${(results.confidence * 100).toFixed(1)}%

DÉTAILS
-------
Fenêtres analysées : ${results.totalWindows}
Crises détectées : ${results.seizureWindows}
Temps de traitement : ${results.analysisTime}

RÉPARTITION DES VOTES
----------------------
Gauche    : ${results.classification.left} segments (${(results.leftConfidence * 100).toFixed(1)}% confiance)
Droite    : ${results.classification.right} segments (${(results.rightConfidence * 100).toFixed(1)}% confiance)
Bilatéral : ${results.classification.bilateral} segments (${(results.bilateralConfidence * 100).toFixed(1)}% confiance)

----------------------------------------
NeuroLat est un outil d'aide à la décision.
Ce résultat doit être validé par un professionnel de santé qualifié.
`
                  const blob = new Blob([reportText], { type: 'text/plain;charset=utf-8' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `rapport_neurolat_${results.fileName.replace('.edf', '')}.txt`
                  document.body.appendChild(a)
                  a.click()
                  document.body.removeChild(a)
                  URL.revokeObjectURL(url)
                }}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 bg-secondary text-foreground font-semibold rounded-lg hover:bg-border transition-colors"
              >
                <Download className="w-5 h-5" />
                Télécharger le rapport
              </button>
              <button
                onClick={handleReset}
                className="inline-flex items-center justify-center px-6 py-3 border border-border text-foreground font-semibold rounded-lg hover:bg-secondary transition-colors"
              >
                Nouvelle analyse
              </button>
            </div>
          </div>
        )}

        {/* Error State */}
        {analysisState === 'error' && (
          <div className="max-w-xl mx-auto text-center space-y-6 py-20">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-50">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <h2 className="text-2xl font-bold text-foreground">Erreur d&apos;analyse</h2>
            <p className="text-muted-foreground">{errorMessage}</p>
            <button
              onClick={handleReset}
              className="inline-flex items-center justify-center px-6 py-3 bg-primary text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors"
            >
              Réessayer
            </button>
          </div>
        )}
      </div>
    </main>
  )
}
