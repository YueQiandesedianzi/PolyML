import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { TitleBar } from './TitleBar'
import { WelcomePage } from '../../pages/WelcomePage'
import { ProjectPage } from '../../pages/ProjectPage'
import { DataImportPage } from '../../pages/DataImportPage'
import { DOEPage } from '../../pages/DOEPage'
import { FeaturesPage } from '../../pages/FeaturesPage'
import { AutoMLPage } from '../../pages/AutoMLPage'
import { ResultsPage } from '../../pages/ResultsPage'
import { PredictPage } from '../../pages/PredictPage'
import { PolymerDBPage } from '../../pages/PolymerDBPage'
import { ActiveLearningPage } from '../../pages/ActiveLearningPage'
import { CodeExportPage } from '../../pages/CodeExportPage'
import { ChatWidget } from '../ChatWidget'

export const AppShell: React.FC = () => {
  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <TitleBar />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<WelcomePage />} />
            <Route path="/project" element={<ProjectPage />} />
            <Route path="/data" element={<DataImportPage />} />
            <Route path="/doe" element={<DOEPage />} />
            <Route path="/features" element={<FeaturesPage />} />
            <Route path="/automl" element={<AutoMLPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/predict" element={<PredictPage />} />
            <Route path="/active-learning" element={<ActiveLearningPage />} />
            <Route path="/code-export" element={<CodeExportPage />} />
            <Route path="/polymer-db" element={<PolymerDBPage />} />
          </Routes>
        </main>
      </div>
      <ChatWidget />
    </div>
  )
}
