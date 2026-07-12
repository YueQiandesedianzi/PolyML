import React, { useState, useEffect } from 'react'
import { getTrainingResults, getParityData, getFeatureImportance, saveModel, deleteRun } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'

type Tab = 'parity' | 'residuals' | 'comparison' | 'heatmap' | 'features'

export const ResultsPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [runs, setRuns] = useState<any[]>([])
  const [selectedRun, setSelectedRun] = useState<any>(null)
  const [parity, setParity] = useState<any>(null)
  const [featureImp, setFeatureImp] = useState<any>(null)
  const [modelName, setModelName] = useState('')
  const [saved, setSaved] = useState(false)
  const [tab, setTab] = useState<Tab>('parity')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (currentProject) loadRuns()
  }, [currentProject?.id])

  const loadRuns = async () => {
    if (!currentProject) return
    try {
      const data = await getTrainingResults(currentProject.id)
      setRuns(data)
      if (data.length > 0) {
        selectRun(data[data.length - 1])
      }
    } catch { setRuns([]) }
  }

  const selectRun = async (run: any) => {
    if (!currentProject) return
    setSelectedRun(run)
    setSaved(false)
    try {
      const [p, fi] = await Promise.all([
        getParityData(currentProject.id, run.runId || run.run_id),
        getFeatureImportance(currentProject.id, run.runId || run.run_id).catch(() => null),
      ])
      setParity(p)
      setFeatureImp(fi)
    } catch { setParity(null); setFeatureImp(null) }
  }

  const handleSave = async () => {
    if (!currentProject || !selectedRun || !modelName.trim()) return
    try {
      await saveModel(currentProject.id, modelName.trim(), selectedRun.runId || selectedRun.run_id)
      setSaved(true)
    } catch (e: any) { setError(e.message || '保存失败') }
  }

  const handleDeleteRun = async (run: any) => {
    if (!currentProject) return
    const rid = run.runId || run.run_id
    const label = `运行 · ${run.bestModel || run.best_model || '?'}`
    if (!confirm(`确定删除 ${label}？此操作不可撤销。`)) return
    setError(null)
    try {
      await deleteRun(currentProject.id, rid)
      // Reload runs and select the last remaining one
      const data = await getTrainingResults(currentProject.id)
      setRuns(data)
      if (data.length > 0) {
        selectRun(data[data.length - 1])
      } else {
        setSelectedRun(null)
        setParity(null)
        setFeatureImp(null)
      }
    } catch (e: any) { setError(e.message || '删除失败') }
  }

  // Aggregate results across runs for model comparison
  const allModelResults = runs.flatMap((r) => Object.entries(r.results || {}))
  const uniqueModels = [...new Set(allModelResults.map(([k]) => k))]

  const tabs: { key: Tab; label: string }[] = [
    { key: 'parity', label: '预测 vs 实际' },
    { key: 'residuals', label: '残差分析' },
    { key: 'comparison', label: '模型对比' },
    { key: 'heatmap', label: '相关性热图' },
    { key: 'features', label: '特征重要性' },
  ]

  return (
    <div className="max-w-5xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">结果分析</h2>
      <p className="text-sm text-gray-500 mb-5">可视化训练结果并保存最佳模型。</p>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-sm text-red-700 flex justify-between items-center">
          <span>{error}</span>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 ml-2">&times;</button>
        </div>
      )}

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : runs.length === 0 ? (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-sm text-gray-500">
          暂无训练记录。请先到「自动建模」训练模型。
        </div>
      ) : (
        <div className="space-y-5">
          {/* Run selector */}
          <div className="flex gap-2 flex-wrap">
            {runs.map((run, i) => {
              const rid = run.runId || run.run_id
              const isSelected = selectedRun?.runId === rid || selectedRun?.run_id === rid
              return (
                <div key={rid} className={`flex items-center text-xs rounded-md border transition ${
                  isSelected
                    ? 'bg-primary-50 border-primary-300 text-primary-700'
                    : 'border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}>
                  <button
                    onClick={() => selectRun(run)}
                    className="px-3 py-1.5"
                  >
                    运行 {i + 1} · {run.bestModel || run.best_model || '?'}
                  </button>
                  <button
                    onClick={(e) => { e.stopPropagation(); handleDeleteRun(run) }}
                    className="px-1.5 py-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-r-md border-l border-gray-200 transition"
                    title="删除此运行"
                  >
                    ×
                  </button>
                </div>
              )
            })}
          </div>

          {/* Tab bar */}
          <div className="flex gap-1 border-b border-gray-200">
            {tabs.map((t) => (
              <button
                key={t.key}
                onClick={() => setTab(t.key)}
                className={`px-4 py-2 text-sm font-medium transition border-b-2 ${
                  tab === t.key
                    ? 'border-primary-500 text-primary-700'
                    : 'border-transparent text-gray-500 hover:text-gray-700'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {tab === 'parity' && (
            parity ? (
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-sm font-medium text-gray-700 mb-3">预测值 vs 实际值</h3>
                <ParityPlot data={parity} />
              </div>
            ) : (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-sm text-gray-500">
                暂无预测数据
              </div>
            )
          )}

          {tab === 'residuals' && (
            parity ? (
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-sm font-medium text-gray-700 mb-3">残差分析</h3>
                <ResidualPlot data={parity} />
              </div>
            ) : (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-sm text-gray-500">
                暂无残差数据
              </div>
            )
          )}

          {tab === 'comparison' && (
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-gray-700 mb-3">模型性能对比</h3>
              <ModelComparisonChart runs={runs} />
            </div>
          )}

          {tab === 'heatmap' && (
            parity ? (
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-sm font-medium text-gray-700 mb-3">预测-实际相关性</h3>
                <CorrelationCard data={parity} />
              </div>
            ) : (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-sm text-gray-500">
                暂无相关性数据
              </div>
            )
          )}

          {tab === 'features' && (
            featureImp ? (
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-sm font-medium text-gray-700 mb-3">特征重要性（前 20）</h3>
                <FeatureBarChart data={featureImp} />
              </div>
            ) : (
              <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-sm text-gray-500">
                暂无特征重要性数据
              </div>
            )
          )}

          {/* Save model */}
          <div className="bg-white border border-gray-200 rounded-lg p-5 flex items-center gap-3">
            <input
              type="text"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="模型名称（如：my_pet_model）"
              className="flex-1 border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
            <button
              onClick={handleSave}
              disabled={!modelName.trim() || saved}
              className="bg-primary-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
            >
              {saved ? '已保存' : '保存模型'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/* ---- SVG Parity Plot ---- */
const ParityPlot: React.FC<{ data: any }> = ({ data }) => {
  const { y_test, y_pred, r2, rmse, mae } = data
  const size = 400
  const pad = 50
  const allVals = [...y_test, ...y_pred]
  const min = Math.min(...allVals)
  const max = Math.max(...allVals)
  const range = max - min || 1
  const lo = min - range * 0.05
  const hi = max + range * 0.05

  const sx = (v: number) => pad + ((v - lo) / (hi - lo)) * (size - 2 * pad)
  const sy = (v: number) => size - pad - ((v - lo) / (hi - lo)) * (size - 2 * pad)

  return (
    <div className="flex items-start gap-6">
      <svg width={size} height={size} className="border border-gray-100 rounded">
        {/* Diagonal */}
        <line x1={sx(lo)} y1={sy(lo)} x2={sx(hi)} y2={sy(hi)} stroke="#e5e7eb" strokeDasharray="4" />
        {/* Points */}
        {y_test.map((v: number, i: number) => (
          <circle key={i} cx={sx(v)} cy={sy(y_pred[i])} r={3} fill="#3b82f6" opacity={0.6} />
        ))}
        <text x={size / 2} y={size - 8} textAnchor="middle" fontSize="11" fill="#6b7280">实际值</text>
        <text x={12} y={size / 2} textAnchor="middle" fontSize="11" fill="#6b7280" transform={`rotate(-90,12,${size / 2})`}>预测值</text>
      </svg>
      <div className="text-sm space-y-2 mt-2">
        <div className="font-medium text-gray-800">评估指标</div>
        <div className="text-xs text-gray-600">R² = <span className="font-mono font-medium">{r2?.toFixed(4)}</span></div>
        <div className="text-xs text-gray-600">RMSE = <span className="font-mono">{rmse?.toFixed(4)}</span></div>
        <div className="text-xs text-gray-600">MAE = <span className="font-mono">{mae?.toFixed(4)}</span></div>
        <div className="text-xs text-gray-400 mt-2">n = {y_test.length}</div>
      </div>
    </div>
  )
}

/* ---- Residual Plot ---- */
const ResidualPlot: React.FC<{ data: any }> = ({ data }) => {
  const { y_test, y_pred } = data
  const residuals = y_test.map((v: number, i: number) => v - y_pred[i])
  const size = 400
  const pad = 50

  const xMin = Math.min(...y_pred)
  const xMax = Math.max(...y_pred)
  const xRange = xMax - xMin || 1
  const xLo = xMin - xRange * 0.05
  const xHi = xMax + xRange * 0.05

  const yAbsMax = Math.max(...residuals.map(Math.abs)) || 1
  const yLo = -yAbsMax * 1.2
  const yHi = yAbsMax * 1.2

  const sx = (v: number) => pad + ((v - xLo) / (xHi - xLo)) * (size - 2 * pad)
  const sy = (v: number) => size / 2 - (v / yAbsMax) * (size / 2 - pad)

  return (
    <div className="flex items-start gap-6">
      <svg width={size} height={size} className="border border-gray-100 rounded">
        {/* Zero line */}
        <line x1={pad} y1={sy(0)} x2={size - pad} y2={sy(0)} stroke="#e5e7eb" strokeDasharray="4" />
        {/* ±2σ dashed lines if data exists */}
        {y_test.length > 3 && (() => {
          const mean = residuals.reduce((s: number, r: number) => s + r, 0) / residuals.length
          const std = Math.sqrt(residuals.reduce((s: number, r: number) => s + (r - mean) * (r - mean), 0) / residuals.length)
          return (
            <>
              <line x1={pad} y1={sy(2 * std)} x2={size - pad} y2={sy(2 * std)} stroke="#fca5a5" strokeDasharray="3" />
              <line x1={pad} y1={sy(-2 * std)} x2={size - pad} y2={sy(-2 * std)} stroke="#fca5a5" strokeDasharray="3" />
            </>
          )
        })()}
        {/* Points */}
        {residuals.map((r: number, i: number) => (
          <circle key={i} cx={sx(y_pred[i])} cy={sy(r)} r={3} fill="#f59e0b" opacity={0.7} />
        ))}
        <text x={size / 2} y={size - 8} textAnchor="middle" fontSize="11" fill="#6b7280">预测值</text>
        <text x={12} y={size / 2} textAnchor="middle" fontSize="11" fill="#6b7280" transform={`rotate(-90,12,${size / 2})`}>残差</text>
      </svg>
      <div className="text-sm space-y-2 mt-2">
        <div className="font-medium text-gray-800">残差统计</div>
        {(() => {
          const mean = residuals.reduce((s: number, v: number) => s + v, 0) / residuals.length
          const std = Math.sqrt(residuals.reduce((s: number, r: number) => s + (r - mean) * (r - mean), 0) / residuals.length)
          return (
            <>
              <div className="text-xs text-gray-600">均值 = <span className="font-mono">{mean.toFixed(4)}</span></div>
              <div className="text-xs text-gray-600">标准差 = <span className="font-mono">{std.toFixed(4)}</span></div>
            </>
          )
        })()}
        <div className="text-xs text-gray-600">最大|残差| = <span className="font-mono">{Math.max(...residuals.map(Math.abs)).toFixed(4)}</span></div>
        <div className="text-xs text-gray-400 mt-2">红虚线 = ±2σ</div>
      </div>
    </div>
  )
}

/* ---- Model Comparison Bar Chart ---- */
const ModelComparisonChart: React.FC<{ runs: any[] }> = ({ runs }) => {
  // Aggregate best result per model across runs
  const modelBest: Record<string, { r2: number; rmse: number; count: number }> = {}
  for (const run of runs) {
    for (const [key, val] of Object.entries(run.results || {})) {
      const v = val as any
      if (!modelBest[key] || v.test_r2 > modelBest[key].r2) {
        modelBest[key] = { r2: v.test_r2, rmse: v.test_rmse, count: 1 }
      }
    }
  }

  const entries = Object.entries(modelBest).sort(([, a], [, b]) => b.r2 - a.r2)
  if (entries.length === 0) return <p className="text-sm text-gray-500">无对比数据</p>

  const maxR2 = Math.max(...entries.map(([, v]) => v.r2))
  const barColors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']

  return (
    <div>
      <div className="flex gap-4 mb-4">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          <div className="w-3 h-3 rounded" style={{ background: barColors[0] }} /> R²
        </div>
      </div>
      <div className="space-y-2">
        {entries.map(([key, val], i) => (
          <div key={key} className="flex items-center gap-3">
            <span className="w-36 text-xs text-gray-700 font-medium truncate">{key}</span>
            <div className="flex-1 flex items-center gap-2">
              <div className="flex-1 bg-gray-100 rounded-full h-5 overflow-hidden">
                <div
                  className="h-full rounded-full transition-all"
                  style={{ width: `${(val.r2 / (maxR2 || 1)) * 100}%`, background: barColors[i % barColors.length] }}
                />
              </div>
              <span className="text-xs font-mono text-gray-600 w-14 text-right">{val.r2?.toFixed(4)}</span>
            </div>
            <span className="text-[10px] text-gray-400 w-20 text-right">RMSE {val.rmse?.toFixed(3)}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ---- Correlation Card ---- */
const CorrelationCard: React.FC<{ data: any }> = ({ data }) => {
  const { y_test, y_pred, r2 } = data
  // Pearson correlation
  const n = y_test.length
  const meanY = y_test.reduce((s: number, v: number) => s + v, 0) / n
  const meanP = y_pred.reduce((s: number, v: number) => s + v, 0) / n
  let num = 0, denY = 0, denP = 0
  for (let i = 0; i < n; i++) {
    const dy = y_test[i] - meanY
    const dp = y_pred[i] - meanP
    num += dy * dp
    denY += dy * dy
    denP += dp * dp
  }
  const r = num / (Math.sqrt(denY) * Math.sqrt(denP))

  return (
    <div className="grid grid-cols-3 gap-6 max-w-md">
      <div className="text-center p-4 bg-blue-50 rounded-lg">
        <div className="text-3xl font-bold text-blue-600">{r2?.toFixed(4)}</div>
        <div className="text-xs text-gray-500 mt-1">R²</div>
      </div>
      <div className="text-center p-4 bg-green-50 rounded-lg">
        <div className="text-3xl font-bold text-green-600">{r?.toFixed(4)}</div>
        <div className="text-xs text-gray-500 mt-1">Pearson r</div>
      </div>
      <div className="text-center p-4 bg-orange-50 rounded-lg">
        <div className="text-3xl font-bold text-orange-600">{n}</div>
        <div className="text-xs text-gray-500 mt-1">样本数</div>
      </div>
    </div>
  )
}

/* ---- Feature Importance Bar Chart ---- */
const FeatureBarChart: React.FC<{ data: any }> = ({ data }) => {
  const { features, importance } = data
  const top = features.slice(0, 20)
  const topImp = importance.slice(0, 20)
  const maxImp = Math.max(...topImp) || 1

  return (
    <div className="space-y-1">
      {top.map((feat: string, i: number) => (
        <div key={i} className="flex items-center gap-2 text-xs">
          <span className="w-44 truncate text-right text-gray-600 font-mono text-[10px]">{feat}</span>
          <div className="flex-1 h-4 bg-gray-100 rounded overflow-hidden">
            <div
              className="h-full bg-primary-500 rounded"
              style={{ width: `${(topImp[i] / maxImp) * 100}%` }}
            />
          </div>
          <span className="w-14 text-right text-gray-500 font-mono text-[10px]">
            {topImp[i]?.toFixed(4)}
          </span>
        </div>
      ))}
    </div>
  )
}
