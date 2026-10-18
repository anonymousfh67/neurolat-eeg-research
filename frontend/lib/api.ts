// URL of the local Flask server (python app/server.py). Override with NEXT_PUBLIC_API_URL in .env.local if needed.
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:5000'

export type Focus = 'gauche' | 'droite' | 'bilateral'

export interface Detection {
  start_s: number
  stop_s: number
  duration_s: number
  n_windows: number
  p1_max: number
  p_left: number
  p_right: number
  p_bilateral: number
  label: Focus
}

export interface AnalysisResult {
  recording: string
  duration_s: number
  n_fenetres: number
  n_crises: number
  pct_crises: number
  pct_flat: number
  n_detections: number
  detections: Detection[]
  votes: Record<Focus, number>
  prediction: Focus | 'non_detecte'
  label: string
  probs: number[]
  labels_differ: boolean
  rule: { tau: number; n_min: number; merge_gap_s: number }
  timeline: { step_s: number; win_s: number; p1: number[]; flat: number[] }
  disclaimer: string
}

export const FOCUS_FR: Record<Focus, string> = { gauche: 'Gauche', droite: 'Droit', bilateral: 'Bilatéral' }
export const FOCUS_COLOR: Record<Focus, string> = { gauche: '#2563eb', droite: '#dc2626', bilateral: '#7c3aed' }

export function mmss(t: number): string {
  const s = Math.round(t)
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  return `${h}:${String(m).padStart(2, '0')}:${String(sec).padStart(2, '0')}`
}
