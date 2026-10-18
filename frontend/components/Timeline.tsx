'use client'

import { FOCUS_COLOR, mmss, type AnalysisResult } from '@/lib/api'

// Probability of seizure (M1) over the whole recording, decision threshold, flat windows, and the detections coloured by predicted side.
export function Timeline({ result }: { result: AnalysisResult }) {
  const W = 900, H = 170, padL = 36, padR = 8, padT = 8, padB = 24
  const total = Math.max(result.duration_s, 1)
  const x = (t: number) => padL + (t / total) * (W - padL - padR)
  const y = (p: number) => padT + (1 - p) * (H - padT - padB)
  const { p1, flat, step_s } = result.timeline
  const pts = p1.map((p, i) => `${x(i * step_s + 2).toFixed(1)},${y(p).toFixed(1)}`).join(' ')
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * total)
  return (
    <div className="bg-secondary rounded-lg p-4 border border-border overflow-x-auto">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full min-w-[640px]" role="img" aria-label="Probabilité de crise au cours du temps">
        {flat.map((f, i) => f ? <rect key={i} x={x(i * step_s)} y={padT} width={Math.max(1, x(step_s) - x(0))} height={H - padT - padB} fill="#d1d5db" opacity={0.6} /> : null)}
        {result.detections.map((d, i) => (
          <rect key={i} x={x(d.start_s)} y={padT} width={Math.max(2, x(d.stop_s) - x(d.start_s))} height={H - padT - padB} fill={FOCUS_COLOR[d.label]} opacity={0.28} />
        ))}
        <line x1={padL} x2={W - padR} y1={y(result.rule.tau)} y2={y(result.rule.tau)} stroke="#6b7280" strokeDasharray="4 3" />
        <text x={W - padR} y={y(result.rule.tau) - 3} textAnchor="end" fontSize="10" fill="#6b7280">seuil {result.rule.tau}</text>
        <polyline points={pts} fill="none" stroke="#111827" strokeWidth="1" />
        <line x1={padL} x2={padL} y1={padT} y2={H - padB} stroke="#9ca3af" />
        <line x1={padL} x2={W - padR} y1={H - padB} y2={H - padB} stroke="#9ca3af" />
        {[0, 0.5, 1].map((p) => <text key={p} x={padL - 4} y={y(p) + 3} textAnchor="end" fontSize="10" fill="#6b7280">{p}</text>)}
        {ticks.map((t, i) => <text key={i} x={x(t)} y={H - 8} textAnchor={i === 0 ? 'start' : i === 4 ? 'end' : 'middle'} fontSize="10" fill="#6b7280">{mmss(t)}</text>)}
      </svg>
      <div className="flex flex-wrap gap-4 text-xs text-muted-foreground mt-2">
        <span>Courbe noire : probabilité de crise (M1)</span>
        <span>Gris : fenêtres plates, jamais détectées</span>
        {(['gauche', 'droite', 'bilateral'] as const).map((f) => (
          <span key={f} className="inline-flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-sm" style={{ backgroundColor: FOCUS_COLOR[f], opacity: 0.5 }} />{f === 'gauche' ? 'Gauche' : f === 'droite' ? 'Droit' : 'Bilatéral'}</span>
        ))}
      </div>
    </div>
  )
}
