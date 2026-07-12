import React, { useState, useEffect } from 'react'
import { getDOEFactors, generateDOEDesign, applyDOEDesign } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'
import type { DOEFactor, DOEDesignResponse } from '@/renderer/types/api'

const METHODS = [
  { key: 'full_factorial', name: '全因子实验', desc: '穷举所有组合，实验次数多但信息完整', minFactors: 2 },
  { key: 'fractional_factorial', name: '部分因子', desc: '筛选关键因素，减少实验次数', minFactors: 2 },
  { key: 'lhs', name: '拉丁超方', desc: '空间填充设计，适合连续因素', minFactors: 2 },
  { key: 'box_behnken', name: 'Box-Behnken', desc: '三水平设计，不含极端组合', minFactors: 3 },
  { key: 'ccd', name: '中心复合设计', desc: '五水平设计，适合二次响应面', minFactors: 3 },
]

interface DOEConstraint {
  type: string
  factorNames: string[]
  value: number
  relation: string
}

export const DOEPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [step, setStep] = useState(1)
  const [factors, setFactors] = useState<DOEFactor[]>([])
  const [selectedFactors, setSelectedFactors] = useState<Set<string>>(new Set())
  const [factorRanges, setFactorRanges] = useState<Record<string, { low: number; high: number }>>({})
  const [method, setMethod] = useState('full_factorial')
  const [nSamples, setNSamples] = useState(10)
  const [design, setDesign] = useState<DOEDesignResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [applied, setApplied] = useState(false)
  const [constraints, setConstraints] = useState<DOEConstraint[]>([])

  // Load factors from backend
  useEffect(() => {
    if (currentProject) {
      getDOEFactors(currentProject.id)
        .then((f) => {
          const safe = Array.isArray(f) ? f : []
          setFactors(safe)
          // Initialize ranges from data min/max
          const ranges: Record<string, { low: number; high: number }> = {}
          safe.forEach((factor) => {
            ranges[factor.name] = { low: factor.currentLow, high: factor.currentHigh }
          })
          setFactorRanges(ranges)
        })
        .catch(() => { setFactors([]); setError('加载因子数据失败') })
    }
  }, [currentProject?.id])

  const toggleFactor = (name: string) => {
    setSelectedFactors((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const selectedCount = selectedFactors.size
  const selectedMethod = METHODS.find((m) => m.key === method)

  const estimatedCount = (() => {
    if (selectedCount < 2) return 0
    if (method === 'lhs') return nSamples
    if (method === 'full_factorial') return 2 ** selectedCount
    if (method === 'fractional_factorial') return 2 ** Math.max(1, selectedCount - 1)
    if (method === 'box_behnken') {
      if (selectedCount < 3) return 0
      return selectedCount * (selectedCount - 1) * 2 + 3
    }
    if (method === 'ccd') {
      if (selectedCount < 3) return 0
      return 2 ** selectedCount + 2 * selectedCount + 4
    }
    return 0
  })()

  const handleGenerate = async () => {
    if (!currentProject || selectedCount < 2) return
    setLoading(true)
    setError(null)
    try {
      const factorsPayload = Array.from(selectedFactors).map((name) => ({
        name,
        low: factorRanges[name]?.low ?? 0,
        high: factorRanges[name]?.high ?? 1,
      }))
      const res = await generateDOEDesign(currentProject.id, {
        method,
        factors: factorsPayload,
        nSamples: method === 'lhs' ? nSamples : undefined,
        constraints: constraints.length > 0 ? constraints : undefined,
      })
      setDesign(res)
      setStep(3)
    } catch (e: any) {
      setError(e.message || '生成设计失败')
    } finally {
      setLoading(false)
    }
  }

  const handleApply = async (mode: 'append' | 'predict') => {
    if (!currentProject || !design) return
    setLoading(true)
    try {
      await applyDOEDesign(currentProject.id, {
        mode,
        designMatrix: design.designMatrix,
        smilesTemplate: '*',
      })
      setApplied(true)
    } catch (e: any) {
      setError(e.message || '应用设计失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-4xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">实验设计</h2>
      <p className="text-sm text-gray-500 mb-5">系统性地设计实验方案，优化聚合物配方和工艺参数。</p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : factors.length === 0 ? (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center text-sm text-gray-500">
          请先导入数据并配置列映射（需要至少 2 个数值列作为因素）。
        </div>
      ) : (
        <>
          {/* Step indicator */}
          <div className="flex items-center gap-2 mb-6">
            {[1, 2, 3].map((s) => (
              <React.Fragment key={s}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition ${
                  step >= s ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-500'
                }`}>
                  {s}
                </div>
                <span className="text-xs text-gray-500 mr-2">
                  {s === 1 ? '选择因素' : s === 2 ? '选择方法' : '查看并应用'}
                </span>
                {s < 3 && <div className="flex-1 h-px bg-gray-200" />}
              </React.Fragment>
            ))}
          </div>

          {/* Step 1: Select factors */}
          {step === 1 && (
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-gray-700 mb-3">选择实验因素</h3>
              <p className="text-xs text-gray-400 mb-4">选择数值列作为实验因素，并设置每个因素的低水平和高水平。</p>
              <div className="space-y-3">
                {factors.map((f) => (
                  <div key={f.name} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                    <input
                      type="checkbox"
                      checked={selectedFactors.has(f.name)}
                      onChange={() => toggleFactor(f.name)}
                      className="mt-0.5"
                    />
                    <div className="flex-1">
                      <div className="text-sm font-medium text-gray-800">{f.name}</div>
                      <div className="text-[10px] text-gray-400">
                        数据范围: {f.min.toFixed(2)} ~ {f.max.toFixed(2)} · 均值: {f.mean.toFixed(2)}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <label className="text-[10px] text-gray-400">低</label>
                      <input
                        type="number"
                        value={factorRanges[f.name]?.low ?? f.min}
                        onChange={(e) => setFactorRanges((p) => ({
                          ...p,
                          [f.name]: { ...p[f.name], low: Number(e.target.value) },
                        }))}
                        className="w-20 border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                      />
                      <label className="text-[10px] text-gray-400">高</label>
                      <input
                        type="number"
                        value={factorRanges[f.name]?.high ?? f.max}
                        onChange={(e) => setFactorRanges((p) => ({
                          ...p,
                          [f.name]: { ...p[f.name], high: Number(e.target.value) },
                        }))}
                        className="w-20 border border-gray-300 rounded px-2 py-1 text-xs font-mono"
                      />
                    </div>
                  </div>
                ))}
              </div>
              <button
                onClick={() => setStep(2)}
                disabled={selectedCount < 2}
                className="mt-4 bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
              >
                下一步 ({selectedCount} 个因素已选)
              </button>
            </div>
          )}

          {/* Step 2: Select method */}
          {step === 2 && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 gap-3">
                {METHODS.map((m) => {
                  const valid = selectedCount >= m.minFactors
                  return (
                    <button
                      key={m.key}
                      onClick={() => valid && setMethod(m.key)}
                      disabled={!valid}
                      className={`text-left p-4 rounded-lg border transition ${
                        method === m.key
                          ? 'border-primary-400 bg-primary-50 ring-1 ring-primary-200'
                          : valid
                          ? 'border-gray-200 bg-white hover:bg-gray-50'
                          : 'border-gray-100 bg-gray-50 opacity-50 cursor-not-allowed'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium text-gray-800">{m.name}</span>
                        {!valid && (
                          <span className="text-[10px] text-red-400">需要 {m.minFactors}+ 因素</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500 mt-1">{m.desc}</div>
                    </button>
                  )
                })}
              </div>

              {method === 'lhs' && (
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <label className="text-xs text-gray-500">采样数量</label>
                  <input
                    type="number"
                    value={nSamples}
                    onChange={(e) => setNSamples(Math.max(2, Number(e.target.value)))}
                    min={2}
                    max={100}
                    className="w-20 ml-2 border border-gray-300 rounded px-2 py-1 text-sm"
                  />
                </div>
              )}

              {/* Constraints */}
              <div className="bg-white border border-gray-200 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-medium text-gray-700">约束条件（可选）</h4>
                  <button
                    onClick={() => setConstraints((prev) => [...prev, { type: 'sum', factorNames: Array.from(selectedFactors).slice(0, 2), value: 0, relation: 'lte' }])}
                    className="text-[10px] text-primary-600 hover:text-primary-700"
                  >+ 添加约束</button>
                </div>
                {constraints.length === 0 && (
                  <p className="text-[10px] text-gray-400">无约束 — 所有设计点均保留</p>
                )}
                {constraints.map((c, i) => (
                  <div key={i} className="flex items-center gap-2 mb-2 text-xs">
                    <select
                      value={c.type}
                      onChange={(e) => setConstraints((prev) => prev.map((x, j) => j === i ? { ...x, type: e.target.value } : x))}
                      className="border border-gray-300 rounded px-2 py-1"
                    >
                      <option value="sum">和约束</option>
                      <option value="ratio">比值约束</option>
                      <option value="bound">边界约束</option>
                    </select>
                    <select
                      value={c.relation}
                      onChange={(e) => setConstraints((prev) => prev.map((x, j) => j === i ? { ...x, relation: e.target.value } : x))}
                      className="border border-gray-300 rounded px-2 py-1"
                    >
                      <option value="eq">=</option>
                      <option value="lte">≤</option>
                      <option value="gte">≥</option>
                    </select>
                    <input
                      type="number"
                      value={c.value}
                      onChange={(e) => setConstraints((prev) => prev.map((x, j) => j === i ? { ...x, value: Number(e.target.value) } : x))}
                      className="w-20 border border-gray-300 rounded px-2 py-1"
                    />
                    <span className="text-gray-400">of</span>
                    <select
                      value={c.factorNames.join(',')}
                      onChange={(e) => setConstraints((prev) => prev.map((x, j) => j === i ? { ...x, factorNames: e.target.value.split(',') } : x))}
                      className="border border-gray-300 rounded px-2 py-1 flex-1"
                    >
                      {Array.from(selectedFactors).map((fn) => (
                        <option key={fn} value={fn}>{fn}</option>
                      ))}
                    </select>
                    <button
                      onClick={() => setConstraints((prev) => prev.filter((_, j) => j !== i))}
                      className="text-red-400 hover:text-red-600 text-[10px]"
                    >删除</button>
                  </div>
                ))}
              </div>

              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 text-sm text-blue-700">
                预计生成 <span className="font-mono font-medium">{estimatedCount}</span> 组实验
              </div>

              {error && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-600">{error}</div>}

              <div className="flex gap-3">
                <button onClick={() => setStep(1)} className="border border-gray-300 text-gray-600 px-4 py-2 rounded-md text-sm hover:bg-gray-50">
                  上一步
                </button>
                <button
                  onClick={handleGenerate}
                  disabled={loading || estimatedCount === 0}
                  className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
                >
                  {loading ? '生成中...' : '生成设计'}
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Review and apply */}
          {step === 3 && design && (
            <div className="space-y-4">
              <div className="bg-white border border-gray-200 rounded-lg p-5">
                <h3 className="text-sm font-medium text-gray-700 mb-3">
                  设计矩阵 · {design.method} · {design.nExperiments} 组实验
                  {design.constraintsApplied && design.nBeforeConstraints && design.nBeforeConstraints !== design.nExperiments && (
                    <span className="text-xs text-orange-500 ml-2">
                      （{design.nBeforeConstraints} 组 → 约束过滤后 {design.nExperiments} 组）
                    </span>
                  )}
                </h3>
                <div className="max-h-64 overflow-auto border border-gray-100 rounded">
                  <table className="min-w-full text-xs">
                    <thead className="bg-gray-50 sticky top-0">
                      <tr>
                        <th className="px-3 py-2 text-left font-medium text-gray-500">#</th>
                        {design.factorNames.map((name) => (
                          <th key={name} className="px-3 py-2 text-left font-medium text-gray-500">{name}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {design.designMatrix.map((row, i) => (
                        <tr key={i} className="hover:bg-gray-50">
                          <td className="px-3 py-1.5 text-gray-400 font-mono">{i + 1}</td>
                          {design.factorNames.map((name) => (
                            <td key={name} className="px-3 py-1.5 font-mono">{row[name]?.toFixed(3)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {applied ? (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-sm text-green-700">
                  设计已应用！可前往「特征工程」继续。
                </div>
              ) : (
                <div className="flex gap-3">
                  <button onClick={() => setStep(2)} className="border border-gray-300 text-gray-600 px-4 py-2 rounded-md text-sm hover:bg-gray-50">
                    上一步
                  </button>
                  <button
                    onClick={() => handleApply('append')}
                    disabled={loading}
                    className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
                  >
                    {loading ? '应用中...' : '追加到数据集'}
                  </button>
                  <button
                    onClick={() => handleApply('predict')}
                    disabled={loading}
                    className="border border-primary-300 text-primary-600 px-4 py-2 rounded-md text-sm hover:bg-primary-50 transition"
                  >
                    保存为预测集
                  </button>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}
