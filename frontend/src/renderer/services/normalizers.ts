import type { DOEDesignResponse, PredictionResponse } from '@/renderer/types/api'

export function normalizePredictionResponse(data: any): PredictionResponse {
  return {
    prediction: data.prediction,
    uncertainty: data.uncertainty,
    units: data.units,
    modelUsed: data.model_used ?? data.modelUsed,
    modelId: data.model_id ?? data.modelId,
    targetName: data.target_name ?? data.targetName,
    targetUnit: data.target_unit ?? data.targetUnit ?? '',
    uncertaintyKind: data.uncertainty_kind ?? data.uncertaintyKind,
    warnings: data.warnings ?? [],
    applicabilityDomain: data.applicability_domain ?? data.applicabilityDomain,
  }
}

export function normalizeDOEDesignResponse(data: any): DOEDesignResponse {
  return {
    method: data.method,
    nExperiments: data.n_experiments ?? data.nExperiments,
    nBeforeConstraints: data.n_before_constraints ?? data.nBeforeConstraints,
    designMatrix: data.design_matrix ?? data.designMatrix ?? [],
    factorNames: data.factor_names ?? data.factorNames ?? [],
    levels: data.levels,
    constraintsApplied: data.constraints_applied ?? data.constraintsApplied,
    candidateSetId: data.candidate_set_id ?? data.candidateSetId ?? data.design_id,
  }
}
