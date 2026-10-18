'use client'

import Navigation from './Navigation'

interface NavigationClientProps {
  currentPage?: string
}

export function NavigationClient({ currentPage }: NavigationClientProps) {
  return <Navigation currentPage={currentPage} />
}
