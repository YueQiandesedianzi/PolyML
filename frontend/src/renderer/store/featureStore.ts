import { create } from 'zustand'
import type { CustomFeatureRule } from '@/renderer/types/api'

interface FeatureState {
  customRules: CustomFeatureRule[]
  addRule: (rule: CustomFeatureRule) => void
  updateRule: (index: number, rule: CustomFeatureRule) => void
  removeRule: (index: number) => void
  clearRules: () => void
  setRules: (rules: CustomFeatureRule[]) => void
}

export const useFeatureStore = create<FeatureState>((set) => ({
  customRules: [],

  addRule: (rule) => set((s) => ({ customRules: [...s.customRules, rule] })),

  updateRule: (index, rule) => set((s) => ({
    customRules: s.customRules.map((r, i) => (i === index ? rule : r)),
  })),

  removeRule: (index) => set((s) => ({
    customRules: s.customRules.filter((_, i) => i !== index),
  })),

  clearRules: () => set({ customRules: [] }),

  setRules: (rules) => set({ customRules: rules }),
}))
