export type Language = 'fr' | 'en'

// Only the navigation bar is translated; the pages are in French.
export const translations = {
  fr: {
    nav: { home: 'Accueil', analyze: 'Analyse', about: 'À propos', docs: 'Documentation' },
    analyze: { serverOnline: 'Serveur connecté', serverOffline: 'Serveur hors ligne' },
  },
  en: {
    nav: { home: 'Home', analyze: 'Analyze', about: 'About', docs: 'Documentation' },
    analyze: { serverOnline: 'Server connected', serverOffline: 'Server offline' },
  },
}

export function getTranslation(lang: Language, path: string) {
  const keys = path.split('.')
  let value: any = translations[lang]
  for (const key of keys) {
    if (value && typeof value === 'object' && key in value) value = value[key]
    else return path
  }
  return value
}
