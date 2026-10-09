# NeuroLat - Plateforme de Diagnostic EEG Assistée par IA

Suite complète medtech professionnelle pour la détection et latéralisation des crises d'épilepsie via deep learning.

## 🚀 Démarrage Rapide

```bash
# Installation
pnpm install

# Démarrage du serveur de développement
pnpm dev

# Accès
http://localhost:3000
```

---

## 📋 Structure du Projet

```
/vercel/share/v0-project/
├── app/
│   ├── page.tsx              # Landing page (/)
│   ├── analyze/
│   │   └── page.tsx          # Page d'analyse (/analyze)
│   ├── report/
│   │   └── page.tsx          # Rapport détaillé (/report)
│   ├── layout.tsx            # Métadonnées globales
│   └── globals.css           # Design system Tailwind (medtech)
├── components/
│   └── Navigation.tsx        # Header réutilisable
├── MEDTECH_DOCUMENTATION.md  # Documentation complète du design
├── NEUROLAT_DESIGN.md        # Spécifications visuelles (v1)
└── package.json
```

---

## 🎨 Design System

### Palette Medtech Professionnelle
| Élément | Couleur | Hex |
|---------|---------|-----|
| Primaire (Bleu médical) | Confiance, CTAs | `#0B5FFF` |
| Accent (Bleu ciel) | Données, accents | `#00a8e8` |
| Foyer Droit | Rose/Magenta | `#ec4899` |
| Foyer Bilatéral | Violet | `#a855f7` |
| Fond | Blanc pur | `#ffffff` |
| Texte primaire | Bleu-noir | `#0f1419` |
| Texte secondaire | Gris doux | `#6b7280` |
| Border | Gris très léger | `#e5e7eb` |

### Typographie
- **Sans-serif**: Inter (moderne, clinique)
- **Hiérarchie**: Titres gras + espaces généreux
- **Espacement**: Design rassurant et épuré

---

## 📄 Pages et Fonctionnalités

### 1. Landing Page (/)
**Objectif**: Présenter NeuroLat comme solution credible

✅ **Sections**:
- Hero section avec illustration EEG minimaliste
- 4 trust badges (98.7% sensibilité, 96.2% spécificité, TUSZ, EEGConformer)
- "Comment ça marche" - pipeline 3 étapes
- Architecture technique détaillée (M1 + M2)
- Métriques (23.4k échantillons, 250 Hz, 19-21 canaux)
- Footer institutionnel

✅ **Responsive**: Desktop 2-colonnes → Mobile 1-colonne

---

### 2. Page d'Analyse (/analyze)
**Objectif**: Interface professionnelle pour upload EEG et résultats

✅ **État Vide** (par défaut):
- Zone d'upload élégante (drag & drop + clic)
- Accepte: `.edf` uniquement
- 3 cartes d'info (format, spécifications, durée)
- Avertissement discrète (AlertCircle bleu)

✅ **État Chargement** (3-4s):
- "Analyse en cours"
- Pipeline visuel à 2 étapes:
  1. ✅ Détection (100%, coche verte)
  2. 🔄 Classification (animée, pulse)
  3. ⏳ Rapport (en attente)
- Animations fluides (barres + vagues rebondissantes)

✅ **État Résultats**:
- Carte résultat colorée (rose pour droit, bleu pour gauche, violet pour bilatéral)
- Affiche: Foyer détecté + Confiance + Crises
- 3 métriques (fenêtres, durée, temps)
- Distribution des votes (barres horizontales fines)
- 3 boutons d'action:
  - Voir le rapport détaillé
  - Télécharger le rapport
  - Nouvelle analyse

---

### 3. Rapport Détaillé (/report)
**Objectif**: Document clinique consultable et imprimable

✅ **Structure**:
- Actions: Retour | Imprimer | Télécharger PDF
- En-tête document (fichier, date, modèle)
- Résumé exécutif (encadré bleu)
- Frise temporelle interactive (crises colorées)
- Tableau détaillé des crises
- Répartition des votes par classe
- Infos techniques (architecture, dataset, seuils)
- Avertissement clinique final

✅ **Optimisé pour**:
- Lecture écran
- Impression A4 (styles `print:`)
- Consultabilité du rapport (document clinique)
- Confidentialité et conformité

---

## 🔧 Données Mock

Tous les résultats utilisent des données mockées pour la démo:

```javascript
{
  fileName: 'patient_001.edf',
  totalWindows: 847,
  seizureWindows: 34,
  primaryFocus: 'right', // 'left' | 'right' | 'bilateral'
  confidence: 0.87,
  classification: { left: 12, right: 18, bilateral: 4 },
  recordingDuration: '47:02 min',
  analysisTime: '3.4s'
}
```

---

## 🎯 Flux Utilisateur Complet

### Scénario: Neurologue analyse un patient

1. **Accueil** (`/`) - Comprend la technologie
2. **CTA** - Clique "Lancer une analyse"
3. **Upload** (`/analyze`) - Dépose fichier EEG
4. **Chargement** - Voit pipeline 3 étapes
5. **Résultats** - Consulte classification (foyer droit, 87% confiance)
6. **Rapport** - Clique "Voir rapport détaillé"
7. **Impression** - Imprime ou exporte PDF

---

## ✨ Points Forts du Design

✅ **Crédibilité Médicale**
- Palette medtech professionnelle (bleu #0B5FFF)
- Typographie institutionnelle
- Ton sérieux et rassurant
- Espaces blancs généreux

✅ **Clarté et Hiérarchie**
- Données mises en avant
- CTA primaires évidentes
- Avertissements visibles
- Pipeline visuellement intuitif

✅ **Responsive & Accessible**
- Mobile-first design
- Breakpoints: sm, md, lg
- ARIA labels + sémantique HTML
- Contraste WCAG AA

✅ **Animations Subtiles**
- Pulse sur étapes actives
- Progress bars fluides
- Vagues rebondissantes (wave animation)
- Transitions CSS optimisées

✅ **Impression-Friendly**
- Styles `print:` pour A4
- Rapport sans clutter
- Lisibilité en B&W
- Métadonnées complètes

---

## 📱 Responsive Breakpoints

| Device | Viewport | Layout |
|--------|----------|--------|
| Mobile | 375px | 1 colonne, stack vertical |
| Tablet | 768px | 2 colonnes, grids |
| Desktop | 1920px | 2-3 colonnes, spacieux |

---

## 🔐 Sécurité & Conformité

- ✅ **Avertissement clinique** - "Outil d'aide à la décision"
- ✅ **Disclaimer** - "Ne remplace pas l'avis clinique"
- ✅ **Confidentialité** - "Document confidentiel"
- ✅ **Conformité IVDR** - Mention au footer
- ✅ **Responsabilité** - Clairement affichée

---

## 🚀 Déploiement

```bash
# Build de production
pnpm build

# Export statique (optionnel)
pnpm export

# Déploiement sur Vercel
vercel deploy
```

Cliquez sur **"Publish"** en haut à droite pour déployer instantanément sur Vercel.

---

## 🔮 Améliorations Futures

- [ ] Backend réel (Python + FastAPI / Node.js)
- [ ] Base de données (Neon PostgreSQL / Supabase)
- [ ] Authentification (Better Auth / Auth.js)
- [ ] Historique analyses (gestion utilisateur)
- [ ] Export PDF réel (pdfkit / puppeteer)
- [ ] WebSocket pour progress real-time
- [ ] Dark mode toggle
- [ ] Multilingue (FR/EN)
- [ ] Monitoring & Analytics (Sentry, PostHog)
- [ ] Tests E2E (Playwright / Cypress)

---

## 📚 Documentation

Pour plus de détails:
- **Design complet**: `MEDTECH_DOCUMENTATION.md`
- **Architecture antérieure**: `NEUROLAT_DESIGN.md`

---

## 👨‍💼 Support & Contact

- 📧 contact@neurolat.com (placeholder)
- 🔗 Documentation: `Documentation` (menu)
- 💬 Support: voir footer

---

## 📄 License

Dispositif médical en développement - NeuroLat v2.0 (2026)

---

**Built with**: React 19 + Next.js 16 + Tailwind CSS v4 + Lucide Icons

**Last Updated**: 24 June 2026
