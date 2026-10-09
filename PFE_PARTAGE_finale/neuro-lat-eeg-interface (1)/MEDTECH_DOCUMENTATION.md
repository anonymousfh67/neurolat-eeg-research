# NeuroLat - Plateforme Medtech Professionnelle

Système complet d'aide au diagnostic EEG pour la détection et latéralisation de crises d'épilepsie via deep learning.

## Architecture et Design System

### Palette de Couleurs (Medtech Profesionelle)
- **Bleu Médical Primaire**: `#0B5FFF` - Confiance, crédibilité clinique
- **Bleu Ciel Accent**: `#00a8e8` - Données secondaires, accents
- **Fond**: `#ffffff` - Blanc pur pour clarté maximale
- **Texte Primaire**: `#0f1419` - Noir très légèrement bleuté
- **Texte Secondaire**: `#6b7280` - Gris doux pour hiérarchie
- **Border/Séparations**: `#e5e7eb` - Gris très léger

### Typographie
- **Sans-serif institutionnel**: Inter / IBM Plex Sans (moderne, clinique)
- **Hiérarchie claire**: Titres gras et spacieux, corps léger
- **Espaces blancs généreux**: Design rassurant et épuré

---

## Structure des Pages

### 1. Landing Page (/)
**Objectif**: Présenter NeuroLat comme solution medtech crédible

#### Sections:
- **Hero**: Titre "Latéralisation EEG assistée par IA" + sous-texte explicatif
  - CTA primaire: "Lancer une analyse"
  - CTA secondaire: "Demander une démo"
  - Illustration: Courbes EEG minimalistes (line-art bleu)
  - Trust badges: 98.7% sensibilité, 96.2% spécificité, TUSZ dataset, EEGConformer

- **Comment ça marche**: 3 étapes visuelles
  1. Upload EEG
  2. Détection IA (modèle M1)
  3. Classification latéralisation (modèle M2)

- **Architecture technique**: Pipeline détaillé
  - Étape 1 : Détection de crise (M1)
  - Étape 2 : Classification latéralisation (M2)
  - Métriques: Dataset, architecture, performances

- **Footer**: Navigation complète, informations légales, conformité IVDR

#### Responsive:
- Desktop: 2 colonnes (texte + visuel)
- Mobile: 1 colonne, typographie scalée

---

### 2. Page d'Analyse (/analyze)
**Objectif**: Interface professionnelle pour télécharger EEG et consulter résultats

#### États:

##### État Vide (par défaut)
- **Zone d'upload élégante**:
  - Icône de dossier médical (FileText)
  - Texte: "Déposez votre fichier EEG" + sous-texte
  - Hover state: Border bleu, fond léger
  - Accepte: `.edf` uniquement

- **Cartes d'info**: 3 badges informatifs
  1. Format supporté: EEG (.edf)
  2. Spécifications: 250 Hz – 19–21 canaux
  3. Durée: < 5 secondes

- **Avertissement**: Bleu léger (AlertCircle) - "Cet outil est une aide à la décision, ne remplace pas l'avis clinique d'un spécialiste"

##### État Chargement (3-4s)
- **Titre**: "Analyse en cours"
- **Pipeline visuel à 2 étapes**:
  1. ✅ Étape 1 : Détection (100% complète, coche verte)
     - Barre de progression (100%)
  2. 🔄 Étape 2 : Classification (en cours, pulse animation)
     - Barre de progression animée (~67%)
  3. ⏳ Étape 3 : Rapport (en attente, numéro gris)
     - Pas encore en cours

- **Animations**: 
  - Pulse sur l'étape active
  - Barres de progression fluides
  - Vagues animées en bas (3 barres rebondissantes)

##### État Résultats (après analyse)
- **En-tête**: Nom fichier + timestamp
- **Carte résultat principale** (couleur dynamique):
  - Fond: Rose accent (#ec4899) pour foyer droit
  - Texte: Blanc
  - Titre grand: "Foyer droit" (ou gauche/bilatéral)
  - 2 valeurs principales:
    - Confiance du modèle: 87%
    - Nombre de crises: 34

- **Métriques grid** (3 colonnes):
  - Fenêtres analysées: 847
  - Durée enregistrement: 47:02 min
  - Temps de traitement: 3.4s

- **Distribution des votes** (barres horizontales fines):
  - Foyer gauche: 12 segments (23.0% confiance)
  - Foyer droit: 18 segments (87.0% confiance) ← plus épais
  - Bilatéral: 4 segments (15.0% confiance)
  - Code couleur: Bleu/Rose/Violet

- **Avertissement clinique** (Amber): "Remarque clinique - cet outil est une aide..."

- **Actions** (3 boutons):
  - Primaire: "Voir le rapport détaillé" (bleu)
  - Secondaire: "Télécharger le rapport" (gris)
  - Tertiaire: "Nouvelle analyse" (border)

---

### 3. Page de Rapport (/report)
**Objectif**: Document clinique consultable et imprimable (A4-friendly)

#### Structure:
- **Actions en haut** (print:hidden):
  - Retour
  - Imprimer
  - Télécharger PDF

- **En-tête document**:
  - "RAPPORT D'ANALYSE EEG" (petit texte gris)
  - Logo + "NeuroLat"
  - Sous-texte: "Latéralisation assistée par IA"
  - Infos fichier (grid 2x2):
    - Fichier analysé
    - Enregistrement
    - Fréquence d'échantillonnage
    - Canaux

- **Résumé exécutif** (encadré bleu):
  - Foyer détecté (grande police, bleu)
  - Confiance du modèle (%)
  - Crises détectées (count)

- **Détail de l'analyse**:
  1. **Frise temporelle** (interactive/hover):
     - Barre de temps (0:00 → 47:02)
     - Marqueurs colorés des crises
     - Code couleur: Bleu/Rose/Violet
     - Tooltip au hover: heure + durée + classe

  2. **Liste détaillée des crises** (table):
     - Colonnes: Heure début | Heure fin | Durée | Classification
     - Classe affichée comme badge coloré
     - Exemple: 5 crises détectées

- **Répartition des votes** (barres horizontales):
  - Même que page analyse mais plus sobre
  - Formato rapport: texte gris
  - Optimisé pour impression B&W

- **Informations techniques** (grid petit texte):
  - Architecture du modèle
  - Dataset d'entraînement
  - Fenêtres temporelles
  - Seuils de décision
  - Performances validées
  - Date du rapport

- **Avertissement clinique final** (Amber, important):
  - Texte complet sur confidentialité et usage clinique

- **Footer**: Copyright + "Document confidentiel"

#### Impression:
- Design prévu pour A4
- Print styles: Borders gris clair, bg supprimés
- Classes `print:` partout
- Breakpoints à l'impression

---

## Composants Réutilisables

### Navigation.tsx
- Logo + marque
- Menu: Accueil | Analyse | À propos | Documentation
- Statut système: "Modèles chargés" + indicateur vert
- Responsive: Menu caché sur mobile

### Palette de Couleurs par Foyer
- **Gauche**: Bleu (#0B5FFF)
- **Droit**: Rose/Accent (#ec4899)
- **Bilatéral**: Violet/Purple (#a855f7)

### Animations
- Pulse: `animate-pulse` sur éléments actifs
- Wave: Barres qui rebondissent (`animate-bounce`)
- Progress bars: `transition-all duration-500`

---

## Données Mock

```javascript
{
  fileName: 'patient_001.edf',
  fileSize: '1.45 MB',
  analysisTime: '3.4s',
  timestamp: '24/06/2026 10:00:41',
  totalWindows: 847,
  seizureWindows: 34,
  recordingDuration: '47:02 min',
  samplingRate: '250 Hz',
  channels: 19,
  primaryFocus: 'right', // 'left' | 'right' | 'bilateral'
  confidence: 0.87, // 0-1
  classification: {
    left: 12,
    right: 18,
    bilateral: 4
  },
  leftConfidence: 0.23,
  rightConfidence: 0.87,
  bilateralConfidence: 0.15
}
```

---

## Fonctionnalités Clés

### Upload
- Drag & drop ou clic
- Format: `.edf` uniquement
- Validation automatique

### Analysis Flow
1. Upload file → Set state to "loading"
2. Simulate 3-4s processing
3. Generate mock results
4. Display results with animations

### Report Generation
- PDF export (placeholder button)
- Print-friendly layout
- All metadata included

### Navigation
- Sticky header
- Active page highlight
- System health indicator

---

## Accessibilité

- **Sémantique HTML**: `<main>`, `<header>`, `<section>`, `<footer>`
- **ARIA labels**: Utiles pour icônes
- **Contraste**: Conforme WCAG AA
- **Focus states**: Visibles et logiques
- **Textes alternatifs**: Illustrations SVG descriptives

---

## Performance & SEO

### Metadata (layout.tsx)
```
Title: NeuroLat – EEG Seizure Focus Detection
Description: Analyze EEG recordings to detect seizure foci...
Theme Color: #0B5FFF
```

### Optimisations
- Images SVG inline pour hero
- Pas d'images externes
- CSS-in-JS Tailwind (optimisé)
- Client-side state management (React hooks)

---

## Cas d'Usage Complets

### Scenario 1: Neurologue analyse un patient
1. Accès landing page → comprend la technologie
2. Clic "Lancer une analyse"
3. Upload fichier EEG patient
4. Attends ~3-4s (voir loading state)
5. Consulte résultats (foyer détecté + confiance)
6. Clique "Voir rapport détaillé"
7. Consulte/imprime/exporte rapport

### Scenario 2: Chercheur explore les métriques
1. Analyze page → upload dataset
2. Vérifie résultats granulaires
3. Consulte distribution votes par classe
4. Compare avec session précédente

---

## Prochaines Étapes Possibles

- [ ] Backend API réelle (Python + FastAPI)
- [ ] Authentification (OAuth/Better Auth)
- [ ] Historique analyses (base de données)
- [ ] Export PDF réel (pdfkit/puppeteer)
- [ ] WebSocket pour progress real-time
- [ ] Dark mode toggle
- [ ] Multilingue (FR/EN)
- [ ] Métriques/Analytics
- [ ] Tests E2E (Playwright)
- [ ] Performance monitoring

---

## Notes de Design

- **Ton**: Professionnel, rassurant, clinique
- **Pas d'émojis**: Environnement médical
- **Espaces généreux**: Lisibilité et crédibilité
- **Couleurs institutionnelles**: Bleu médical + blanc
- **Pas d'animations flashy**: Subtiles et utiles
- **Typographie claire**: Font-sizes progressives

---

**Date de création**: 24 juin 2026
**Version**: 1.0 - Production ready (UI/UX)
**Stack**: React 19 + Next.js 16 + Tailwind CSS v4
