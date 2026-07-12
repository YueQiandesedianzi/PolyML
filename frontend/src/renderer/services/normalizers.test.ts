import { describe, expect, it } from 'vitest'
import { normalizeDOEDesignResponse, normalizePredictionResponse } from './normalizers'

describe('API response normalizers', () => {
  it('normalizes versioned prediction metadata', () => {
    expect(normalizePredictionResponse({
      prediction: 12.3,
      uncertainty: 0.4,
      uncertainty_kind: 'validation_rmse',
      units: 'MPa',
      target_name: 'strength',
      target_unit: 'MPa',
      model_id: 'ridge-v2',
      model_used: 'ridge',
      warnings: ['imputed'],
    })).toMatchObject({
      modelId: 'ridge-v2', targetName: 'strength', uncertaintyKind: 'validation_rmse',
    })
  })

  it('keeps the DOE candidate-set identifier', () => {
    expect(normalizeDOEDesignResponse({
      method: 'lhs', n_experiments: 4, design_matrix: [], factor_names: [], levels: {},
      candidate_set_id: 'abc123',
    }).candidateSetId).toBe('abc123')
  })
})
