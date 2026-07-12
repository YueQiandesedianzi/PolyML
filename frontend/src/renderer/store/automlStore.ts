import { create } from 'zustand'

interface ModelResult {
  r2?: number
  rmse?: number
  mae?: number
  duration_sec?: number
  cv_score?: number
  selection_metric?: string
}

interface AutoMLState {
  modelResults: Record<string, ModelResult>
  bestModel: string | null
  setModelResults: (results: Record<string, ModelResult>) => void
  setBestModel: (model: string | null) => void
  reset: () => void
}

export const useAutoMLStore = create<AutoMLState>((set) => ({
  modelResults: {},
  bestModel: null,
  setModelResults: (results) =>
    set((state) => ({
      modelResults: { ...state.modelResults, ...results },
    })),
  setBestModel: (model) => set({ bestModel: model }),
  reset: () => set({ modelResults: {}, bestModel: null }),
}))
