'use client'

import { FOCUS_COLOR, FOCUS_FR, mmss, type Detection } from '@/lib/api'

export function DetectionsTable({ detections }: { detections: Detection[] }) {
  if (detections.length === 0) {
    return <p className="text-sm text-muted-foreground">Aucune détection avec la règle appliquée.</p>
  }
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b-2 border-border text-left">
            <th className="py-2 px-3 font-semibold">#</th>
            <th className="py-2 px-3 font-semibold">Début</th>
            <th className="py-2 px-3 font-semibold">Fin</th>
            <th className="py-2 px-3 font-semibold">Durée (s)</th>
            <th className="py-2 px-3 font-semibold">Fenêtres ≥ seuil</th>
            <th className="py-2 px-3 font-semibold">p(crise) max</th>
            <th className="py-2 px-3 font-semibold">Côté prédit</th>
            <th className="py-2 px-3 font-semibold">p(G) / p(D) / p(B)</th>
          </tr>
        </thead>
        <tbody>
          {detections.map((d, i) => (
            <tr key={i} className="border-b border-border">
              <td className="py-2 px-3 text-muted-foreground">{i + 1}</td>
              <td className="py-2 px-3">{mmss(d.start_s)}</td>
              <td className="py-2 px-3">{mmss(d.stop_s)}</td>
              <td className="py-2 px-3">{Math.round(d.duration_s)}</td>
              <td className="py-2 px-3">{d.n_windows}</td>
              <td className="py-2 px-3">{d.p1_max.toFixed(2)}</td>
              <td className="py-2 px-3">
                <span className="inline-block px-3 py-1 rounded text-xs font-semibold text-white" style={{ backgroundColor: FOCUS_COLOR[d.label] }}>{FOCUS_FR[d.label]}</span>
              </td>
              <td className="py-2 px-3 text-muted-foreground">{d.p_left.toFixed(2)} / {d.p_right.toFixed(2)} / {d.p_bilateral.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
