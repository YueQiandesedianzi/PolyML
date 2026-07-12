import React, { useState, useEffect } from 'react'
import { api } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'

interface Suggestion {
  index: number
  features: number[]
  acquisition_score: number
  predicted_y: number
  predicted_std: number
}

interface FeatureImportance {
  name: string
  importance: number
}

interface CVResults {
  r2: number
  rmse: number
  mae: number
  n_samples: number
}

export const ActiveLearningPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([])
  const [cvResults, setCvResults] = useState<CVResults | null>(null)
  const [acquisition, setAcquisition] = useState('ei')
  const [nSuggestions, setNSuggestions] = useState(5)
  const [loading, setLoading] = useState(false)
  const [evalLoading, setEvalLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [status, setStatus] = useState<{ ready: boolean; n_samples?: number; n_features?: number } | null>(null)

  useEffect(() => {
    if (currentProject) {
      api.get(`/projects/${currentProject.id}/active-learning/status`)
        .then((r) => setStatus(r.data))
        .catch(() => setStatus({ ready: false }))
    }
  }, [currentProject?.id])

  const handleSuggest = async () => {
    if (!currentProject) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.post(`/projects/${currentProject.id}/active-learning/suggest`, {
        acquisition,
        nSuggestions: nSuggestions,
        xi: 0.01,
        beta: 2.0,
      })
      setSuggestions(res.data.suggestions)
      setFeatureImportance(res.data.feature_importance)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleEvaluate = async () => {
    if (!currentProject) return
    setEvalLoading(true)
    try {
      const res = await api.post(`/projects/${currentProject.id}/active-learning/evaluate-gpr`)
      setCvResults(res.data.cv_results)
      setFeatureImportance(res.data.feature_importance)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setEvalLoading(false)
    }
  }

  return (
    <div className="max-w-5xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">主动学习 / 贝叶斯优化</h2>
      <p className="text-sm text-gray-500 mb-6">使用高斯过程代理模型和采集函数，智能推荐下一个实验。</p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : !status?.ready ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先完成特征工程计算。
        </div>
      ) : (
        <>
          {/* Status bar */}
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 mb-5 flex gap-6 text-sm">
            <span className="text-blue-700">样本数: <strong>{status.n_samples}</strong></span>
            <span className="text-blue-700">特征数: <strong>{status.n_features}</strong></span>
          </div>

          {/* Controls */}
          <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
            <div className="flex gap-6 items-end flex-wrap">
              <div>
                <label className="text-xs text-gray-500">采集函数</label>
                <select
                  value={acquisition}
                  onChange={(e) => setAcquisition(e.target.value)}
                  className="ml-2 border border-gray-300 rounded px-2 py-1 text-sm"
                >
                  <option value="ei">EI — 期望改进</option>
                  <option value="ucb">UCB — 上置信界</option>
                  <option value="pi">PI — 改进概率</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500">推荐数量</label>
                <input
                  type="number"
                  value={nSuggestions}
                  onChange={(e) => setNSuggestions(Number(e.target.value))}
                  min={1} max={20}
                  className="w-16 ml-2 border border-gray-300 rounded px-2 py-1 text-sm"
                />
              </div>
              <button
                onClick={handleSuggest}
                disabled={loading}
                className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
              >
                {loading ? '计算中...' : '推荐下一步实验'}
              </button>
              <button
                onClick={handleEvaluate}
                disabled={evalLoading}
                className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm hover:bg-gray-50 transition"
              >
                {evalLoading ? '评估中...' : '评估 GPR 质量'}
              </button>
            </div>

            {error && <p className="mt-3 text-red-500 text-xs">{error}</p>}
          </div>

          {/* GPR CV results */}
          {cvResults && (
            <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
              <h3 className="text-sm font-medium text-gray-700 mb-3">GPR 代理模型交叉验证</h3>
              <div className="flex gap-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-primary-600">{cvResults.r2?.toFixed(4)}</div>
                  <div className="text-xs text-gray-500">R²</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{cvResults.rmse?.toFixed(4)}</div>
                  <div className="text-xs text-gray-500">RMSE</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-orange-600">{cvResults.mae?.toFixed(4)}</div>
                  <div className="text-xs text-gray-500">MAE</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-gray-800">{cvResults.n_samples}</div>
                  <div className="text-xs text-gray-500">样本数</div>
                </div>
              </div>
            </div>
          )}

          {/* Suggestions table */}
          {suggestions.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden mb-5">
              <div className="px-5 py-3 border-b border-gray-200">
                <h3 className="text-sm font-medium text-gray-700">推荐的下一个实验</h3>
              </div>
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">#</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">采集分数</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">预测 Y</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">不确定度 (σ)</th>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">特征向量</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {suggestions.map((s, i) => (
                    <tr key={i} className="bg-blue-50/50">
                      <td className="px-4 py-2 font-medium">{i + 1}</td>
                      <td className="px-4 py-2 text-right font-mono text-primary-600">{s.acquisition_score.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right font-mono">{s.predicted_y.toFixed(4)}</td>
                      <td className="px-4 py-2 text-right font-mono text-orange-600">±{s.predicted_std.toFixed(4)}</td>
                      <td className="px-4 py-2 text-xs font-mono text-gray-500">
                        [{s.features.slice(0, 5).map((f) => f.toFixed(2)).join(', ')}{s.features.length > 5 ? ', ...' : ''}]
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Feature importance */}
          {featureImportance.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-gray-700 mb-3">特征重要性（基于 GPR 核长度尺度）</h3>
              <div className="space-y-2">
                {featureImportance.slice(0, 10).map((fi) => (
                  <div key={fi.name} className="flex items-center gap-3">
                    <span className="text-xs text-gray-600 w-20 truncate" title={fi.name}>{fi.name}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-3">
                      <div
                        className="bg-primary-500 rounded-full h-3 transition-all"
                        style={{ width: `${fi.importance * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-gray-500 w-12 text-right">{(fi.importance * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
