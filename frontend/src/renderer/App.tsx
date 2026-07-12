import React from 'react'
import { HashRouter, Routes, Route } from 'react-router-dom'
import { AppShell } from './components/layout/AppShell'
import { ErrorBoundary } from './components/ErrorBoundary'

export const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <HashRouter>
        <Routes>
          <Route path="/*" element={<AppShell />} />
        </Routes>
      </HashRouter>
    </ErrorBoundary>
  )
}
