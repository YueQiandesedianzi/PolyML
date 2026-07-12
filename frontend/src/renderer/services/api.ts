import axios from 'axios'
import type {
  ProjectMeta,
  DataImportResult,
  ColumnMapping,
  FeatureEngineeringConfig,
  AutoMLConfig,
  TrainingResult,
  PredictionRequest,
  PredictionResponse,
  PolymerEntry,
  DOEFactor,
  DOEDesignRequest,
  DOEDesignResponse,
  DOEApplyRequest,
  CustomFeatureRule,
} from '@/renderer/types/api'
import { normalizeDOEDesignResponse, normalizePredictionResponse } from './normalizers'

export const API_BASE_URL = 'http://127.0.0.1:18921'

const backendConfigPromise: Promise<{ baseUrl: string; sessionToken: string }> =
  window.electronAPI?.getBackendConfig?.() ?? Promise.resolve({ baseUrl: API_BASE_URL, sessionToken: '' })

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  timeout: 30000,
})

api.interceptors.request.use(async (config) => {
  const backend = await backendConfigPromise
  config.baseURL = `${backend.baseUrl}/api`
  if (backend.sessionToken) config.headers.set('X-PolyML-Token', backend.sessionToken)
  return config
})

export { api }

// Health
export async function healthCheck(): Promise<{ status: string; rdkit: boolean }> {
  const { data } = await api.get('/health')
  return data
}

// Projects
export async function createProject(name: string, description: string): Promise<ProjectMeta> {
  const { data } = await api.post('/projects', { name, description })
  return data
}

export async function listProjects(): Promise<ProjectMeta[]> {
  const { data } = await api.get('/projects')
  return data
}

export async function getProject(id: string): Promise<ProjectMeta> {
  const { data } = await api.get(`/projects/${id}`)
  return data
}

export async function deleteProject(id: string): Promise<void> {
  await api.delete(`/projects/${id}`)
}

// Data
export async function importData(
  projectId: string,
  file: File
): Promise<DataImportResult> {
  const formData = new FormData()
  formData.append('file', file)
  const { data } = await api.post(`/projects/${projectId}/data/import`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 60000,
  })
  // Normalize snake_case backend keys to camelCase
  return {
    ...data,
    detectedTypes: data.detected_types || data.detectedTypes || {},
  }
}

export async function mapColumns(
  projectId: string,
  mapping: ColumnMapping
): Promise<void> {
  await api.post(`/projects/${projectId}/data/map-columns`, { mapping })
}

export interface ReloadedData {
  hasData: boolean
  columns: string[]
  preview: Record<string, any>[]
  detectedTypes: Record<string, string>
  mapping: ColumnMapping
  mappingDone: boolean
  rowCount: number
}

export async function reloadProjectData(projectId: string): Promise<ReloadedData> {
  const { data } = await api.get(`/projects/${projectId}/data/reload`)
  return {
    hasData: data.has_data ?? data.hasData ?? false,
    columns: data.columns ?? [],
    preview: data.preview ?? [],
    detectedTypes: data.detected_types ?? data.detectedTypes ?? {},
    mapping: data.mapping ?? {},
    mappingDone: data.mapping_done ?? data.mappingDone ?? false,
    rowCount: data.row_count ?? data.rowCount ?? 0,
  }
}

// Features
export async function engineerFeatures(
  projectId: string,
  config: FeatureEngineeringConfig
): Promise<{ X_shape: [number, number]; feature_names: string[] }> {
  const { data } = await api.post(`/projects/${projectId}/features/engineer`, config)
  return {
    X_shape: data.X_shape ?? data.xShape,
    feature_names: data.feature_names ?? data.featureNames ?? [],
  }
}

// AutoML
export function trainAutoML(
  projectId: string,
  config: AutoMLConfig,
  onEvent: (event: { type: string; data: any }) => void,
  onError: (error: Error) => void,
  onComplete: () => void
): AbortController {
  const controller = new AbortController()

  backendConfigPromise
    .then((backend) => fetch(`${backend.baseUrl}/api/projects/${projectId}/automl/train`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(backend.sessionToken ? { 'X-PolyML-Token': backend.sessionToken } : {}),
      },
      body: JSON.stringify(config),
      signal: controller.signal,
    }))
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`Training failed: ${response.statusText}`)
      }
      const reader = response.body?.getReader()
      if (!reader) throw new Error('No response body')

      const decoder = new TextDecoder()
      let buffer = ''
      let currentEventType = 'message'

      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          // Flush remaining buffer
          if (buffer.trim()) {
            for (const line of buffer.split('\n')) {
              if (line.startsWith('event: ')) {
                currentEventType = line.slice(7).trim()
              } else if (line.startsWith('data: ')) {
                try {
                  const eventData = JSON.parse(line.slice(6))
                  onEvent({ type: currentEventType, data: eventData })
                  currentEventType = 'message'
                } catch { /* skip */ }
              }
            }
          }
          break
        }

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEventType = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(line.slice(6))
              onEvent({ type: currentEventType, data: eventData })
              currentEventType = 'message'
            } catch {
              // skip malformed lines
            }
          }
        }
      }
      onComplete()
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        onError(err)
      }
    })

  return controller
}

export async function cancelTraining(projectId: string): Promise<void> {
  await api.post(`/projects/${projectId}/automl/cancel`)
}

export async function getTrainingResults(projectId: string): Promise<TrainingResult[]> {
  const { data } = await api.get(`/projects/${projectId}/automl/results`)
  // Normalize snake_case keys from backend
  return (data || []).map((r: any) => ({
    runId: r.run_id ?? r.runId,
    bestModel: r.best_model ?? r.bestModel,
    results: r.results,
    totalDurationSec: r.total_duration_sec ?? r.totalDurationSec,
    config: r.config,
    timestamp: r.timestamp,
  }))
}

export async function deleteRun(projectId: string, runId: string): Promise<void> {
  await api.delete(`/projects/${projectId}/automl/runs/${runId}`)
}

export async function getParityData(
  projectId: string,
  runId: string
): Promise<{ y_test: number[]; y_pred: number[]; r2: number; rmse: number }> {
  const { data } = await api.get(`/projects/${projectId}/automl/parity-data`, {
    params: { run_id: runId },
  })
  return data
}

export async function getFeatureImportance(
  projectId: string,
  runId: string
): Promise<{ features: string[]; importance: number[] }> {
  const { data } = await api.get(`/projects/${projectId}/automl/feature-importance`, {
    params: { run_id: runId },
  })
  return data
}

// Prediction
export async function predict(
  projectId: string,
  request: PredictionRequest
): Promise<PredictionResponse> {
  const { data } = await api.post(`/projects/${projectId}/predict`, request)
  return normalizePredictionResponse(data)
}

// Models
export async function saveModel(
  projectId: string,
  name: string,
  runId: string
): Promise<void> {
  await api.post(`/projects/${projectId}/models/save`, { name, run_id: runId })
}

export async function listModels(
  projectId: string
): Promise<Array<{ name: string; modelType?: string; metrics: Record<string, number>; legacy: boolean }>> {
  const { data } = await api.get(`/projects/${projectId}/models`)
  return (data || []).map((model: any) => ({
    ...model,
    modelType: model.model_type ?? model.modelType,
    metrics: model.metrics ?? {},
    legacy: model.legacy ?? false,
  }))
}

// Polymer DB
export async function searchPolymers(query: string): Promise<PolymerEntry[]> {
  const { data } = await api.get('/polymer-db/search', { params: { q: query } })
  return data
}

export async function listAllPolymers(
  page: number = 1,
  limit: number = 50
): Promise<{ items: PolymerEntry[]; total: number }> {
  const { data } = await api.get('/polymer-db', { params: { page, limit } })
  return data
}

export async function addPolymer(entry: Partial<PolymerEntry>): Promise<PolymerEntry> {
  const { data } = await api.post('/polymer-db', entry)
  return data
}

export async function deletePolymer(id: number): Promise<void> {
  await api.delete(`/polymer-db/${id}`)
}

// DOE
export async function getDOEFactors(projectId: string): Promise<DOEFactor[]> {
  const { data } = await api.get(`/projects/${projectId}/doe/factors`)
  const raw = data.factors || []
  // Normalize snake_case backend keys to camelCase
  return raw.map((f: any) => ({
    name: f.name,
    type: f.type,
    min: f.min,
    max: f.max,
    mean: f.mean,
    currentLow: f.current_low ?? f.currentLow ?? f.min,
    currentHigh: f.current_high ?? f.currentHigh ?? f.max,
  }))
}

export async function generateDOEDesign(
  projectId: string,
  request: DOEDesignRequest
): Promise<DOEDesignResponse> {
  const { data } = await api.post(`/projects/${projectId}/doe/generate`, request)
  return normalizeDOEDesignResponse(data)
}

export async function applyDOEDesign(
  projectId: string,
  request: DOEApplyRequest
): Promise<{ applied: boolean; rowsAdded: number; totalRows: number }> {
  const { data } = await api.post(`/projects/${projectId}/doe/apply`, request)
  return {
    applied: data.applied,
    rowsAdded: data.rows_added ?? data.rowsAdded,
    totalRows: data.total_rows ?? data.totalRows,
  }
}

// Custom Features
export async function validateCustomExpression(
  projectId: string,
  rule: { ruleType: string; expression: string; availableColumns: string[] }
): Promise<{ valid: boolean; error?: string }> {
  const { data } = await api.post(`/projects/${projectId}/custom-features/validate`, rule)
  return data
}

// Agent / Chat
export interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export async function sendChatMessage(
  messages: ChatMessage[],
  projectId?: string
): Promise<string> {
  const { data } = await api.post('/agent/chat', { messages, projectId })
  return data.reply ?? data
}

export default api
