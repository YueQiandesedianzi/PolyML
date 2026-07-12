import React, { useState, useEffect } from 'react'
import { predict, searchPolymers, listModels } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'
import type { PolymerEntry } from '@/renderer/types/api'

export const PredictPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [smiles, setSmiles] = useState('')
  const [params, setParams] = useState<Record<string, string>>({})
  const [result, setResult] = useState<any>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Model selection
  const [savedModels, setSavedModels] = useState<Array<{ name: string; model_type?: string }>>([])
  const [selectedModel, setSelectedModel] = useState<string>('')

  // Polymer name search
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<PolymerEntry[]>([])
  const [searching, setSearching] = useState(false)

  // Load saved models on mount
  useEffect(() => {
    if (currentProject) {
      listModels(currentProject.id)
        .then((models) => setSavedModels(models))
        .catch(() => setSavedModels([]))
    }
  }, [currentProject?.id])

  const handleSearch = async (q: string) => {
    setSearchQuery(q)
    if (q.length < 2) { setSearchResults([]); return }
    setSearching(true)
    try {
      const results = await searchPolymers(q)
      setSearchResults(results.slice(0, 8))
    } catch { setSearchResults([]) }
    finally { setSearching(false) }
  }

  const handleSelectPolymer = (p: PolymerEntry) => {
    setSmiles(p.smiles)
    setSearchQuery(p.commonName)
    setSearchResults([])
  }

  const handlePredict = async () => {
    if (!currentProject || !smiles.trim()) return
    setRunning(true)
    setError(null)
    try {
      const numericParams: Record<string, number> = {}
      for (const [k, v] of Object.entries(params)) {
        if (v.trim()) numericParams[k] = Number(v)
      }
      const res = await predict(currentProject.id, {
        smiles: smiles.trim(),
        processingParams: numericParams,
        modelName: selectedModel || undefined,
      })
      setResult(res)
    } catch (e: any) {
      setError(e.message || '预测失败')
      setResult(null)
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="max-w-3xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">预测</h2>
      <p className="text-sm text-gray-500 mb-5">使用训练好的模型预测新聚合物的性质。</p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : (
        <div className="space-y-5">
          {/* Polymer lookup */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-sm font-medium text-gray-700 mb-3">聚合物结构</h3>
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => handleSearch(e.target.value)}
                placeholder="搜索聚合物名称（如：聚苯乙烯）"
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
              {searchResults.length > 0 && (
                <div className="absolute z-10 w-full bg-white border border-gray-200 rounded-md mt-1 shadow-lg max-h-48 overflow-auto">
                  {searchResults.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => handleSelectPolymer(p)}
                      className="w-full text-left px-3 py-2 text-sm hover:bg-gray-50 flex justify-between"
                    >
                      <span>{p.commonName} {p.abbreviation ? `(${p.abbreviation})` : ''}</span>
                      <span className="text-gray-400 font-mono text-xs truncate ml-2">{p.smiles}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>

            <div className="mt-3">
              <label className="text-xs text-gray-500 mb-1 block">SMILES 字符串</label>
              <input
                type="text"
                value={smiles}
                onChange={(e) => setSmiles(e.target.value)}
                placeholder="*CC(*)c1ccccc1"
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm font-mono focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>

          {/* Model selection */}
          {savedModels.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-gray-700 mb-3">选择模型</h3>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
              >
                <option value="">使用最佳模型（自动选择）</option>
                {savedModels.map((m) => (
                  <option key={m.name} value={m.name}>
                    {m.name} {m.model_type ? `(${m.model_type})` : ''}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Processing parameters */}
          <div className="bg-white border border-gray-200 rounded-lg p-5">
            <h3 className="text-sm font-medium text-gray-700 mb-3">工艺参数（可选）</h3>
            <div className="text-xs text-gray-400 mb-3">
              添加训练数据中的数值参数（如：Mn、PDI、温度等）
            </div>
            <div className="space-y-2">
              {Object.entries(params).map(([key, val]) => (
                <div key={key} className="flex items-center gap-2">
                  <input
                    type="text"
                    value={key}
                    readOnly
                    className="w-32 border border-gray-200 bg-gray-50 rounded px-2 py-1 text-xs font-mono"
                  />
                  <input
                    type="text"
                    value={val}
                    onChange={(e) => setParams((p) => ({ ...p, [key]: e.target.value }))}
                    className="flex-1 border border-gray-300 rounded px-2 py-1 text-xs"
                  />
                  <button
                    onClick={() => setParams((p) => { const n = { ...p }; delete n[key]; return n })}
                    className="text-xs text-red-400 hover:text-red-600"
                  >
                    ✕
                  </button>
                </div>
              ))}
              <AddParamButton onAdd={(k, v) => setParams((p) => ({ ...p, [k]: v }))} />
            </div>
          </div>

          {error && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-600">{error}</div>}

          <button
            onClick={handlePredict}
            disabled={running || !smiles.trim()}
            className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
          >
            {running ? '预测中...' : '预测'}
          </button>

          {/* Result */}
          {result && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-green-800 mb-2">预测结果</h3>
              <div className="text-3xl font-bold text-gray-800 font-mono">
                {result.prediction?.toFixed(2)}
                <span className="text-sm font-normal text-gray-500 ml-2">{result.units}</span>
              </div>
              <div className="text-xs text-gray-500 mt-2">
                不确定性: ±{result.uncertainty?.toFixed(2)} · 模型: {result.modelUsed}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const AddParamButton: React.FC<{ onAdd: (key: string, val: string) => void }> = ({ onAdd }) => {
  const [open, setOpen] = useState(false)
  const [key, setKey] = useState('')
  const [val, setVal] = useState('')

  if (!open) {
    return (
      <button onClick={() => setOpen(true)} className="text-xs text-primary-600 hover:text-primary-700">
        + 添加参数
      </button>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <input
        autoFocus
        value={key}
        onChange={(e) => setKey(e.target.value)}
        placeholder="名称（如 Mn）"
        className="w-32 border border-gray-300 rounded px-2 py-1 text-xs"
      />
      <input
        value={val}
        onChange={(e) => setVal(e.target.value)}
        placeholder="值"
        className="flex-1 border border-gray-300 rounded px-2 py-1 text-xs"
      />
      <button
        onClick={() => { if (key.trim()) { onAdd(key.trim(), val); setKey(''); setVal(''); setOpen(false) } }}
        className="text-xs bg-primary-600 text-white px-2 py-1 rounded"
      >
        添加
      </button>
      <button onClick={() => setOpen(false)} className="text-xs text-gray-400">✕</button>
    </div>
  )
}
