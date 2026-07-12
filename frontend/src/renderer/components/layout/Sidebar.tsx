import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { API_BASE_URL } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'

interface NavItem {
  to: string
  label: string
  icon: string
}

const navItems: NavItem[] = [
  { to: '/', label: '首页', icon: '🏠' },
  { to: '/project', label: '项目管理', icon: '📁' },
  { to: '/data', label: '数据导入', icon: '📊' },
  { to: '/doe', label: '实验设计', icon: '🧪' },
  { to: '/features', label: '特征工程', icon: '⚙️' },
  { to: '/automl', label: '自动建模', icon: '🤖' },
  { to: '/results', label: '结果分析', icon: '📈' },
  { to: '/predict', label: '预测', icon: '🎯' },
  { to: '/active-learning', label: '主动学习', icon: '🔬' },
  { to: '/code-export', label: '代码导出', icon: '📋' },
  { to: '/polymer-db', label: '聚合物库', icon: '🗄️' },
]

const NAV_ROUTES_WITHOUT_PROJECT = new Set(['/', '/project', '/polymer-db'])

export const Sidebar: React.FC = () => {
  const { currentProject, projects, setCurrentProject } = useProjectStore()
  const [showSwitcher, setShowSwitcher] = useState(false)

  return (
    <aside className="w-56 bg-white border-r border-gray-200 flex flex-col flex-shrink-0">
      {/* Current project indicator */}
      <div className="px-4 pt-4 pb-3 border-b border-gray-100">
        {currentProject ? (
          <button
            onClick={() => setShowSwitcher(!showSwitcher)}
            className="w-full text-left group"
          >
            <div className="text-[10px] text-gray-400 uppercase tracking-wide mb-1">当前项目</div>
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-800 truncate group-hover:text-primary-600 transition">
                {currentProject.name}
              </span>
              <svg className="w-3 h-3 text-gray-400 shrink-0 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </button>
        ) : (
          <NavLink to="/project" className="block group">
            <div className="text-[10px] text-gray-400 uppercase tracking-wide mb-1">当前项目</div>
            <div className="text-sm text-primary-500 group-hover:text-primary-700 transition">
              + 选择或创建项目
            </div>
          </NavLink>
        )}

        {/* Project switcher dropdown */}
        {showSwitcher && projects.length > 1 && (
          <div className="mt-2 bg-gray-50 rounded-md border border-gray-200 max-h-40 overflow-auto">
            {projects
              .filter((p) => p.id !== currentProject?.id)
              .map((p) => (
                <button
                  key={p.id}
                  onClick={() => { setCurrentProject(p); setShowSwitcher(false) }}
                  className="w-full text-left px-3 py-2 text-xs text-gray-600 hover:bg-white hover:text-primary-600 transition"
                >
                  {p.name}
                </button>
              ))}
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 space-y-0.5 overflow-auto">
        {navItems.map((item) => {
          const needsProject = !NAV_ROUTES_WITHOUT_PROJECT.has(item.to)
          const isDisabled = needsProject && !currentProject
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `flex items-center gap-2 px-4 py-1.5 text-sm mx-2 rounded-md transition-colors ${
                  isActive
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                } ${isDisabled ? 'opacity-40 pointer-events-none' : ''}`
              }
            >
              <span className="text-sm">{item.icon}</span>
              {item.label}
            </NavLink>
          )
        })}
      </nav>

      {/* Backend status */}
      <div className="p-4 border-t border-gray-200">
        <BackendStatusDot />
      </div>
    </aside>
  )
}

const BackendStatusDot: React.FC = () => {
  const [online, setOnline] = React.useState(false)

  React.useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/health`)
        setOnline(res.ok)
      } catch {
        setOnline(false)
      }
    }
    check()
    const interval = setInterval(check, 3000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="flex items-center gap-2 text-xs text-gray-400">
      <span className={`inline-block w-2 h-2 rounded-full ${online ? 'bg-green-500' : 'bg-red-400'}`} />
      {online ? '后端运行中' : '后端离线'}
    </div>
  )
}
