import React, { useState, useRef, useEffect } from 'react'
import { trainAutoML, cancelTraining } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'
import { useAutoMLStore } from '@/renderer/store/automlStore'

const ALL_MODELS = [
  { key: 'ridge', name: 'Ridge', desc: '线性基线', tags: ['线性', '快速'], checked: true },
  { key: 'lasso', name: 'LASSO', desc: '特征选择', tags: ['线性'], checked: true },
  { key: 'elasticnet', name: 'ElasticNet', desc: '混合正则化', tags: ['线性'], checked: false },
  { key: 'pls', name: 'PLS', desc: '偏最小二乘', tags: ['小样本', '高维'], checked: false },
  { key: 'knn', name: 'KNN', desc: 'K近邻', tags: ['非参数'], checked: false },
  { key: 'kernel_ridge', name: 'Kernel Ridge', desc: '核岭回归', tags: ['核方法'], checked: false },
  { key: 'random_forest', name: 'Random Forest', desc: '随机森林', tags: ['集成', '鲁棒'], checked: true },
  { key: 'gradient_boosting', name: 'Gradient Boosting', desc: '梯度提升', tags: ['集成', '强学习器'], checked: false },
  { key: 'xgboost', name: 'XGBoost', desc: '极端梯度提升', tags: ['集成', '强学习器'], checked: true },
  { key: 'svm', name: 'SVM (RBF)', desc: '支持向量机', tags: ['核方法', '非线性'], checked: true },
  { key: 'gaussian_process', name: 'Gaussian Process', desc: '高斯过程', tags: ['小样本', '不确定性'], checked: true },
  { key: 'mlp', name: 'MLP', desc: '神经网络', tags: ['非线性'], checked: false },
]

const CV_METHODS = [
  { key: 'kfold', name: 'K-Fold', desc: '标准交叉验证' },
  { key: 'loocv', name: 'LOOCV', desc: '留一法，适合小样本(<30)' },
  { key: 'repeated_kfold', name: 'Repeated K-Fold', desc: '重复多次，更稳定' },
]

export const AutoMLPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const { modelResults, setModelResults, setBestModel } = useAutoMLStore()
  const [selectedModels, setSelectedModels] = useState<string[]>(
    ALL_MODELS.filter((m) => m.checked).map((m) => m.key)
  )
  const [cvFolds, setCvFolds] = useState(5)
  const [cvMethod, setCvMethod] = useState('kfold')
  const [nTrials, setNTrials] = useState(50)
  const [splitStrategy, setSplitStrategy] = useState<'random' | 'group' | 'time'>('random')
  const [groupColumn, setGroupColumn] = useState('')
  const [selectionMetric, setSelectionMetric] = useState<'rmse' | 'mae' | 'r2'>('rmse')
  const [training, setTraining] = useState(false)
  const [progress, setProgress] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const controllerRef = useRef<AbortController | null>(null)

  const toggleModel = (key: string) => {
    setSelectedModels((prev) =>
      prev.includes(key) ? prev.filter((m) => m !== key) : [...prev, key]
    )
  }

  const handleTrain = async () => {
    if (!currentProject || selectedModels.length === 0) return
    setTraining(true)
    setProgress([])
    setError(null)
    setModelResults({})

    controllerRef.current = trainAutoML(
      currentProject.id,
      {
        models: selectedModels, cvFolds, cvMethod, nTrials, testSize: 0.2,
        splitStrategy,
        groupColumn: groupColumn.trim() || undefined,
        selectionMetric,
        randomSeed: 42,
      },
      (event) => {
        if (event.type === 'model_start') {
          const d = event.data
          setProgress((p) => [...p, `[${d.current_model}/${d.total_models}] 正在训练 ${d.model}...`])
        } else if (event.type === 'model_complete') {
          const d = event.data
          setProgress((p) => [...p, `✓ ${d.model}: CV ${d.selection_metric}=${Number(d.cv_score).toFixed(4)} (${d.duration_sec}s)`])
          setModelResults({ [d.model]: d })
        } else if (event.type === 'all_complete') {
          const d = event.data
          setProgress((p) => [...p, `\n🏆 最佳模型: ${d.best_model}`])
          setBestModel(d.best_model)
        } else if (event.type === 'error') {
          setProgress((p) => [...p, `⚠ 错误: ${event.data.message}`])
        }
      },
      (err) => setError(err.message),
      () => setTraining(false),
    )
  }

  const handleCancel = async () => {
    if (!currentProject) return
    controllerRef.current?.abort()
    await cancelTraining(currentProject.id)
    setTraining(false)
  }

  return (
    <div className="max-w-4xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">AutoML 训练</h2>
      <p className="text-sm text-gray-500 mb-6">选择模型并进行自动超参数优化训练。</p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : (
        <>
          {/* Config panel */}
          <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
            <h3 className="text-sm font-medium text-gray-700 mb-3">模型选择</h3>
            <div className="grid grid-cols-2 gap-2 mb-4">
              {ALL_MODELS.map((m) => (
                <label key={m.key} className={`flex items-start gap-2 p-2 rounded-md cursor-pointer transition text-sm ${
                  selectedModels.includes(m.key) ? 'bg-primary-50 border border-primary-200' : 'border border-gray-100 hover:bg-gray-50'
                }`}>
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(m.key)}
                    onChange={() => toggleModel(m.key)}
                    className="mt-0.5"
                  />
                  <div>
                    <div className="font-medium text-gray-800">{m.name}</div>
                    <div className="text-[10px] text-gray-400">{m.desc}</div>
                    <div className="flex gap-1 mt-0.5">
                      {m.tags.map((t) => (
                        <span key={t} className="text-[9px] px-1 py-0.5 bg-gray-100 text-gray-500 rounded">{t}</span>
                      ))}
                    </div>
                  </div>
                </label>
              ))}
            </div>

            <div className="flex gap-6 items-center flex-wrap">
              <div>
                <label className="text-xs text-gray-500">数据划分</label>
                <select value={splitStrategy} onChange={(e) => setSplitStrategy(e.target.value as any)} className="ml-2 border border-gray-300 rounded px-2 py-1 text-sm">
                  <option value="random">随机</option>
                  <option value="group">按组</option>
                  <option value="time">按时间</option>
                </select>
              </div>
              {splitStrategy !== 'random' && (
                <div>
                  <label className="text-xs text-gray-500">{splitStrategy === 'group' ? '分组列' : '时间列'}</label>
                  <input value={groupColumn} onChange={(e) => setGroupColumn(e.target.value)} className="w-32 ml-2 border border-gray-300 rounded px-2 py-1 text-sm" />
                </div>
              )}
              <div>
                <label className="text-xs text-gray-500">选型指标</label>
                <select value={selectionMetric} onChange={(e) => setSelectionMetric(e.target.value as any)} className="ml-2 border border-gray-300 rounded px-2 py-1 text-sm">
                  <option value="rmse">RMSE</option>
                  <option value="mae">MAE</option>
                  <option value="r2">R²</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-gray-500">交叉验证方法</label>
                <select
                  value={cvMethod}
                  onChange={(e) => setCvMethod(e.target.value)}
                  className="ml-2 border border-gray-300 rounded px-2 py-1 text-sm"
                >
                  {CV_METHODS.map((m) => (
                    <option key={m.key} value={m.key}>{m.name} — {m.desc}</option>
                  ))}
                </select>
              </div>
              {cvMethod !== 'loocv' && (
                <div>
                  <label className="text-xs text-gray-500">交叉验证折数</label>
                  <input
                    type="number"
                    value={cvFolds}
                    onChange={(e) => setCvFolds(Number(e.target.value))}
                    min={2} max={10}
                    className="w-16 ml-2 border border-gray-300 rounded px-2 py-1 text-sm"
                  />
                </div>
              )}
              <div>
                <label className="text-xs text-gray-500">Optuna 试验次数</label>
                <input
                  type="number"
                  value={nTrials}
                  onChange={(e) => setNTrials(Number(e.target.value))}
                  min={10} max={200}
                  className="w-20 ml-2 border border-gray-300 rounded px-2 py-1 text-sm"
                />
              </div>
            </div>

            {error && <p className="mt-3 text-red-500 text-xs">{error}</p>}

            <div className="flex gap-3 mt-4">
              <button
                onClick={handleTrain}
                disabled={training || selectedModels.length === 0}
                className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
              >
                {training ? '训练中...' : '开始训练'}
              </button>
              {training && (
                <button
                  onClick={handleCancel}
                  className="border border-red-300 text-red-600 px-4 py-2 rounded-md text-sm hover:bg-red-50 transition"
                >
                  取消
                </button>
              )}
            </div>
          </div>

          {/* Live progress */}
          {progress.length > 0 && (
            <div className="bg-gray-900 rounded-lg p-4 font-mono text-xs text-green-400 max-h-64 overflow-auto">
              {progress.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap">{line}</div>
              ))}
            </div>
          )}

          {/* Results comparison */}
          {Object.keys(modelResults).length > 0 && !training && (
            <div className="mt-6 bg-white border border-gray-200 rounded-lg overflow-hidden">
              <div className="px-5 py-3 border-b border-gray-200">
                <h3 className="text-sm font-medium text-gray-700">结果对比</h3>
              </div>
              <table className="min-w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">模型</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">CV 指标</th>
                    <th className="px-4 py-2 text-right text-xs font-medium text-gray-500">耗时</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {Object.entries(modelResults)
                    .sort(([, a], [, b]) => selectionMetric === 'r2'
                      ? (b.cv_score ?? -Infinity) - (a.cv_score ?? -Infinity)
                      : (a.cv_score ?? Infinity) - (b.cv_score ?? Infinity))
                    .map(([key, r], i) => (
                      <tr key={key} className={i === 0 ? 'bg-green-50' : ''}>
                        <td className="px-4 py-2 font-medium">{key}{i === 0 ? ' 🏆' : ''}</td>
                        <td className="px-4 py-2 text-right font-mono">{r.cv_score?.toFixed(4)}</td>
                        <td className="px-4 py-2 text-right font-mono">{r.duration_sec?.toFixed(1)}s</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
