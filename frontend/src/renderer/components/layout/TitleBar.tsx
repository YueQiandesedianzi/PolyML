import React from 'react'
import { useProjectStore } from '@/renderer/store/projectStore'
import { useNavigate } from 'react-router-dom'

export const TitleBar: React.FC = () => {
  const { currentProject } = useProjectStore()
  const navigate = useNavigate()

  return (
    <header className="h-12 bg-white border-b border-gray-200 flex items-center px-6 flex-shrink-0">
      <h1
        className="text-lg font-semibold text-gray-800 select-none cursor-pointer hover:text-primary-600 transition"
        onClick={() => navigate('/')}
      >
        PolyML
      </h1>
      <span className="ml-3 text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded">
        聚合物材料机器学习平台
      </span>

      {/* Current project badge */}
      {currentProject && (
        <div className="ml-auto flex items-center gap-2">
          <div className="text-xs text-gray-500">项目:</div>
          <button
            onClick={() => navigate('/project')}
            className="text-xs font-medium text-primary-600 bg-primary-50 hover:bg-primary-100 px-2.5 py-1 rounded-md transition"
          >
            {currentProject.name}
          </button>
          <span className="text-[10px] text-gray-400">
            {currentProject.dataRowCount > 0 ? `${currentProject.dataRowCount} 行数据` : '无数据'}
          </span>
        </div>
      )}
    </header>
  )
}
