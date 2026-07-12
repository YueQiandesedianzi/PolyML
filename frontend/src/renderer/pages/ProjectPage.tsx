import React, { useState, useEffect } from 'react'
import { listProjects, createProject, deleteProject } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'
import type { ProjectMeta } from '@/renderer/types/api'

export const ProjectPage: React.FC = () => {
  const { projects, setProjects, setCurrentProject } = useProjectStore()
  const [newName, setNewName] = useState('')
  const [newDesc, setNewDesc] = useState('')
  const [creating, setCreating] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadProjects()
  }, [])

  const loadProjects = async () => {
    try {
      const list = await listProjects()
      setProjects(list)
    } catch {
      setProjects([])
    }
  }

  const handleCreate = async () => {
    if (!newName.trim()) return
    setCreating(true)
    setError(null)
    try {
      const project = await createProject(newName.trim(), newDesc.trim())
      setNewName('')
      setNewDesc('')
      setCurrentProject(project)
      await loadProjects()
    } catch (e: any) {
      setError(e.message || '项目创建失败')
    } finally {
      setCreating(false)
    }
  }

  const handleDelete = async (id: string) => {
    if (!confirm('确定删除此项目？')) return
    try {
      await deleteProject(id)
      await loadProjects()
    } catch (e: any) {
      setError(e.message)
    }
  }

  const handleSelect = (project: ProjectMeta) => {
    setCurrentProject(project)
  }

  return (
    <div className="max-w-3xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-6">项目管理</h2>

      {/* Create new project */}
      <div className="bg-white rounded-lg border border-gray-200 p-5 mb-6">
        <h3 className="text-sm font-medium text-gray-700 mb-3">创建新项目</h3>
        <div className="flex flex-col gap-3">
          <input
            type="text"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="项目名称（如：PET 玻璃化转变温度）"
            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <input
            type="text"
            value={newDesc}
            onChange={(e) => setNewDesc(e.target.value)}
            placeholder="描述（可选）"
            className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          {error && <p className="text-red-500 text-xs">{error}</p>}
          <button
            onClick={handleCreate}
            disabled={creating || !newName.trim()}
            className="self-start bg-primary-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
          >
            {creating ? '创建中...' : '创建项目'}
          </button>
        </div>
      </div>

      {/* Existing projects */}
      <div className="bg-white rounded-lg border border-gray-200">
        <div className="px-5 py-3 border-b border-gray-200">
          <h3 className="text-sm font-medium text-gray-700">已有项目（{projects.length}）</h3>
        </div>
        {projects.length === 0 ? (
          <div className="p-8 text-center text-sm text-gray-400">
            暂无项目，请在上方创建。
          </div>
        ) : (
          <ul className="divide-y divide-gray-100">
            {projects.map((p) => (
              <li key={p.id} className="px-5 py-3 flex items-center justify-between hover:bg-gray-50">
                <button onClick={() => handleSelect(p)} className="text-left flex-1">
                  <div className="text-sm font-medium text-gray-800">{p.name}</div>
                  {p.description && (
                    <div className="text-xs text-gray-500 mt-0.5">{p.description}</div>
                  )}
                  <div className="text-[10px] text-gray-400 mt-1">
                    {p.dataRowCount} 行 · 创建于 {new Date(p.createdAt).toLocaleDateString()}
                  </div>
                </button>
                <button
                  onClick={() => handleDelete(p.id)}
                  className="text-xs text-red-400 hover:text-red-600 ml-4"
                >
                  删除
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}
