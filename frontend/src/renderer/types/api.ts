// Shared TypeScript types (mirrors Python Pydantic schemas)

export interface ProjectMeta {
  id: string
  name: string
  description: string
  createdAt: string
  updatedAt: string
  dataFilename?: string
  dataRowCount: number
  targetColumn?: string
  smilesColumn?: string
}

export interface ColumnMapping {
  [columnName: string]: 'smiles' | 'numeric' | 'target' | 'ignore'
}

export interface DataImportResult {
  columns: string[]
  preview: Record<string, any>[]
  detectedTypes: Record<string, string>
  rowCount: number
}

export interface AutoMLConfig {
  models: string[]
  cvFolds: number
  cvMethod: string
  nTrials: number
  testSize: number
}

export interface SSEEvent {
  type: 'trial_update' | 'model_complete' | 'model_start' | 'all_complete' | 'error' | 'cancelled'
  data: Record<string, any>
}

export interface ModelResult {
  modelName: string
  bestParams: Record<string, any>
  testRmse: number
  testR2: number
  testMae: number
}

export interface TrainingResult {
  runId: string
  bestModel: string
  results: Record<string, ModelResult>
  totalDurationSec: number
}

export interface PredictionRequest {
  smiles: string
  processingParams: Record<string, number>
  modelName?: string
}

export interface PredictionResponse {
  prediction: number
  uncertainty: number
  units: string
  modelUsed: string
}

// DOE types
export interface DOEFactor {
  name: string
  type: string
  min: number
  max: number
  mean: number
  currentLow: number
  currentHigh: number
}

export interface DOEConstraint {
  type: string
  factorNames: string[]
  value: number
  relation: string
}

export interface DOEDesignRequest {
  method: string
  factors: Array<{ name: string; low: number; high: number; center?: number }>
  nSamples?: number
  resolution?: number
  seed?: number
  constraints?: DOEConstraint[]
}

export interface DOEDesignResponse {
  method: string
  nExperiments: number
  nBeforeConstraints?: number
  designMatrix: Record<string, number>[]
  factorNames: string[]
  levels: Record<string, number[]>
  constraintsApplied?: boolean
}

export interface DOEApplyRequest {
  mode: 'append' | 'predict'
  designMatrix: Record<string, number>[]
  smilesTemplate?: string
  fillValues?: Record<string, any>
}

// Custom feature types
export interface CustomFeatureRule {
  name: string
  ruleType: 'formula' | 'substructure' | 'bin' | 'interaction' | 'domain'
  expression: string
  params: Record<string, any>
}

export interface FeatureEngineeringConfig {
  includeDescriptors: boolean
  includeVanKrevelen: boolean
  include3d: boolean
  customRules?: CustomFeatureRule[]
}

export interface PolymerEntry {
  id: number
  commonName: string
  abbreviation?: string
  smiles: string
  source: 'builtin' | 'user'
  tags: string[]
}
