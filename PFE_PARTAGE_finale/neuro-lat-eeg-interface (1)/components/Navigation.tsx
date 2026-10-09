'use client'

import Link from 'next/link'
import { useState, useEffect } from 'react'
import type { Language } from '@/lib/translations'
import { getTranslation } from '@/lib/translations'
import { Globe } from 'lucide-react'

interface NavigationProps {
  currentPage?: string
}

export default function Navigation({ currentPage }: NavigationProps) {
  const [language, setLanguageState] = useState<Language>('fr')
  const [langMenuOpen, setLangMenuOpen] = useState(false)

  useEffect(() => {
    const stored = (localStorage.getItem('neurolat-language') as Language) || 'fr'
    setLanguageState(stored)
  }, [])

  const setLanguage = (lang: Language) => {
    setLanguageState(lang)
    localStorage.setItem('neurolat-language', lang)
  }

  const t = (path: string) => getTranslation(language, path)
  return (
    <header className="border-b border-border bg-background sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 flex-shrink-0">
            <div className="w-10 h-10 bg-primary rounded flex items-center justify-center">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <span className="font-semibold text-lg text-foreground hidden sm:inline">NeuroLat</span>
          </Link>

          {/* Navigation Menu */}
          <nav className="hidden md:flex items-center gap-8">
            <Link 
              href="/" 
              className={`text-sm font-medium transition-colors ${
                currentPage === 'home' 
                  ? 'text-primary' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t('nav.home')}
            </Link>
            <Link 
              href="/analyze" 
              className={`text-sm font-medium transition-colors ${
                currentPage === 'analyze' 
                  ? 'text-primary' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t('nav.analyze')}
            </Link>
            <Link 
              href="/about" 
              className={`text-sm font-medium transition-colors ${
                currentPage === 'about' 
                  ? 'text-primary' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t('nav.about')}
            </Link>
            <Link 
              href="/docs" 
              className={`text-sm font-medium transition-colors ${
                currentPage === 'docs' 
                  ? 'text-primary' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              {t('nav.docs')}
            </Link>
          </nav>

          {/* System Status & Language Switcher */}
          <div className="flex items-center gap-4">
            {/* Language Switcher */}
            <div className="relative">
              <button
                onClick={() => setLangMenuOpen(!langMenuOpen)}
                className="flex items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors rounded hover:bg-secondary"
                title="Change language"
              >
                <Globe className="w-4 h-4" />
                <span>{language.toUpperCase()}</span>
              </button>
              
              {langMenuOpen && (
                <div className="absolute top-full right-0 mt-2 bg-background border border-border rounded-lg shadow-lg py-2 z-50 min-w-28">
                  <button
                    onClick={() => {
                      setLanguage('fr')
                      setLangMenuOpen(false)
                    }}
                    className={`w-full text-left px-4 py-2 text-sm ${
                      language === 'fr'
                        ? 'bg-primary/10 text-primary font-semibold'
                        : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                    }`}
                  >
                    Français
                  </button>
                  <button
                    onClick={() => {
                      setLanguage('en')
                      setLangMenuOpen(false)
                    }}
                    className={`w-full text-left px-4 py-2 text-sm ${
                      language === 'en'
                        ? 'bg-primary/10 text-primary font-semibold'
                        : 'text-muted-foreground hover:text-foreground hover:bg-secondary'
                    }`}
                  >
                    English
                  </button>
                </div>
              )}
            </div>

            {/* System Status */}
            <div className="flex items-center gap-2 text-xs text-muted-foreground border-l border-border pl-4">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <span className="hidden sm:inline">{t('analyze.status')}</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
