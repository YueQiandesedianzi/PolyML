import { create } from 'zustand'
import type { ProjectMeta } from '@/renderer/types/api'

const STORAGE_KEY = 'polyml_project_id'

function loadPersistedProjectId(): string | null {
  try { return localStorage.getItem(STORAGE_KEY) } catch { return null }
}

function savePersistedProjectId(id: string | null) {
  try {
    if (id) localStorage.setItem(STORAGE_KEY, id)
    else localStorage.removeItem(STORAGE_KEY)
  } catch { /* ignore */ }
}

interface ProjectState {
  projects: ProjectMeta[]
  currentProject: ProjectMeta | null
  backendOnline: boolean

  setProjects: (projects: ProjectMeta[]) => void
  setCurrentProject: (project: ProjectMeta | null) => void
  setBackendOnline: (online: boolean) => void
  addProject: (project: ProjectMeta) => void
  removeProject: (id: string) => void
}

export const useProjectStore = create<ProjectState>((set) => ({
  projects: [],
  currentProject: null,
  backendOnline: false,

  setProjects: (projects) => set((state) => {
    // Restore persisted project selection if available
    const persistedId = loadPersistedProjectId()
    if (persistedId && !state.currentProject) {
      const found = projects.find((p) => p.id === persistedId)
      if (found) return { projects, currentProject: found }
    }
    return { projects }
  }),

  setCurrentProject: (project) => {
    savePersistedProjectId(project?.id ?? null)
    set({ currentProject: project })
  },

  setBackendOnline: (online) => set({ backendOnline: online }),
  addProject: (project) =>
    set((state) => ({ projects: [...state.projects, project] })),
  removeProject: (id) =>
    set((state) => {
      const next = state.projects.filter((p) => p.id !== id)
      const newCurrent = state.currentProject?.id === id ? null : state.currentProject
      savePersistedProjectId(newCurrent?.id ?? null)
      return { projects: next, currentProject: newCurrent }
    }),
}))
