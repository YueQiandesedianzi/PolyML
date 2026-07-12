import React, { useState, useCallback, useEffect } from 'react'
import { importData, mapColumns, reloadProjectData } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'
import type { ColumnMapping } from '@/renderer/types/api'

export const DataImportPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [preview, setPreview] = useState<Record<string, any>[] | null>(null)
  const [columns, setColumns] = useState<string[]>([])
  const [detectedTypes, setDetectedTypes] = useState<Record<string, string>>({})
  const [mapping, setMapping] = useState<ColumnMapping>({})
  const [loading, setLoading] = useState(true)
  const [importing, setImporting] = useState(false)
  const [mappingDone, setMappingDone] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Reload saved data when project changes or component mounts
  useEffect(() => {
    if (!currentProject) return
    let cancelled = false
    setLoading(true)
    reloadProjectData(currentProject.id).then((data) => {
      if (cancelled) return
      if (data.hasData) {
        setPreview(data.preview)
        setColumns(data.columns)
        setDetectedTypes(data.detectedTypes)
        setMapping(data.mapping)
        setMappingDone(data.mappingDone)
      } else {
        setPreview(null)
        setColumns([])
        setDetectedTypes({})
        setMapping({})
        setMappingDone(false)
      }
    }).catch(() => {
      if (!cancelled) {
        setPreview(null)
        setColumns([])
        setDetectedTypes({})
        setMapping({})
        setMappingDone(false)
      }
    }).finally(() => {
      if (!cancelled) setLoading(false)
    })
    return () => { cancelled = true }
  }, [currentProject?.id])

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault()
    if (!currentProject) return setError('请先创建项目')
    const file = e.dataTransfer.files[0]
    if (!file) return
    await doImport(file)
  }, [currentProject])

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) await doImport(file)
  }

  const doImport = async (file: File) => {
    if (!currentProject) return
    setImporting(true)
    setError(null)
    try {
      const result = await importData(currentProject.id, file)
      setPreview(result.preview)
      setColumns(result.columns)
      setDetectedTypes(result.detectedTypes)

      // Auto-init mapping from detected types
      const initMapping: ColumnMapping = {}
      for (const [col, dtype] of Object.entries(result.detectedTypes)) {
        initMapping[col] = dtype as any
      }
      setMapping(initMapping)
      setMappingDone(false)
    } catch (e: any) {
      setError(e.message || '导入失败')
    } finally {
      setImporting(false)
    }
  }

  const handleMappingChange = (col: string, type: string) => {
    setMapping((prev) => ({ ...prev, [col]: type as any }))
  }

  const handleSaveMapping = async () => {
    if (!currentProject) return
    try {
      await mapColumns(currentProject.id, mapping)
      setMappingDone(true)
    } catch (e: any) {
      setError(e.message)
    }
  }

  const typeOptions = ['smiles', 'numeric', 'target', 'ignore']

  return (
    <div className="max-w-5xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">数据导入</h2>
      <p className="text-sm text-gray-500 mb-5">上传包含聚合物数据的 CSV 或 Excel 文件。</p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : (
        <>
          {/* Loading state */}
          {loading && !preview && (
            <div className="bg-gray-50 border border-gray-200 rounded-lg p-8 text-center">
              <div className="inline-block w-6 h-6 border-2 border-primary-400 border-t-transparent rounded-full animate-spin mb-2" />
              <p className="text-sm text-gray-500">加载项目数据...</p>
            </div>
          )}

          {/* Drop zone */}
          {!preview && !loading && (
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="border-2 border-dashed border-gray-300 rounded-lg p-12 text-center hover:border-primary-400 transition cursor-pointer"
            >
              <div className="text-4xl mb-3">📂</div>
              <p className="text-sm text-gray-600 mb-2">
                {importing ? '导入中...' : '拖拽 CSV/Excel 文件到此处，或'}
              </p>
              <label className="inline-block bg-primary-600 text-white px-4 py-2 rounded-md text-sm cursor-pointer hover:bg-primary-700 transition">
                {importing ? '处理中...' : '浏览文件'}
                <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileSelect} className="hidden" />
              </label>
            </div>
          )}

          {error && <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-600">{error}</div>}

          {/* Preview + Column Mapping */}
          {preview && (
            <div className="mt-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-700">
                  列映射 · {columns.length} 列 · {preview.length} 行（预览）
                </h3>
                <button
                  onClick={handleSaveMapping}
                  disabled={mappingDone}
                  className="bg-primary-600 text-white px-4 py-1.5 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
                >
                  {mappingDone ? '✓ 已保存' : '保存映射'}
                </button>
              </div>

              {/* Column type selectors */}
              <div className="bg-white border border-gray-200 rounded-lg p-4 mb-4 flex flex-wrap gap-3">
                {columns.map((col) => (
                  <div key={col} className="flex items-center gap-2">
                    <span className="text-xs font-mono text-gray-700">{col}</span>
                    <select
                      value={mapping[col] || 'ignore'}
                      onChange={(e) => handleMappingChange(col, e.target.value)}
                      className="text-xs border border-gray-300 rounded px-2 py-1"
                    >
                      {typeOptions.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>

              {/* Data preview table */}
              <div className="overflow-auto max-h-96 border border-gray-200 rounded-lg">
                <table className="min-w-full text-xs">
                  <thead className="bg-gray-50 sticky top-0">
                    <tr>
                      {columns.map((col) => (
                        <th key={col} className="px-3 py-2 text-left font-medium text-gray-600 border-b">
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {preview.filter((r) => r != null).map((row, i) => (
                      <tr key={i} className="hover:bg-gray-50">
                        {columns.map((col) => (
                          <td key={col} className="px-3 py-1.5 text-gray-700">
                            {String(row[col] ?? '')}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Data Summary Statistics */}
              <DataSummaryStats preview={preview} columns={columns} detectedTypes={detectedTypes} />

              <button
                onClick={() => { setPreview(null); setColumns([]); setMapping({}); setMappingDone(false) }}
                className="mt-4 text-xs text-gray-400 hover:text-gray-600"
              >
                ← 导入其他文件
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

/* ---- Data Summary Statistics ---- */
const DataSummaryStats: React.FC<{
  preview: Record<string, any>[]
  columns: string[]
  detectedTypes: Record<string, string>
}> = ({ preview, columns, detectedTypes }) => {
  const numericCols = columns.filter((c) => detectedTypes[c] === 'numeric')

  if (numericCols.length === 0 || preview.length === 0) return null

  // Compute stats from preview (skip any null/undefined rows)
  const safePreview = preview.filter((r) => r != null)
  const stats = numericCols.map((col) => {
    const vals = safePreview.map((r) => Number(r[col])).filter((v) => !isNaN(v))
    if (vals.length === 0) return null
    const sorted = [...vals].sort((a, b) => a - b)
    const mean = vals.reduce((s, v) => s + v, 0) / vals.length
    const std = Math.sqrt(vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length)
    const q1 = sorted[Math.floor(sorted.length * 0.25)] ?? sorted[0]
    const q3 = sorted[Math.floor(sorted.length * 0.75)] ?? sorted[sorted.length - 1]

    return { col, n: vals.length, min: sorted[0], q1, mean, q3, max: sorted[sorted.length - 1], std }
  }).filter(Boolean)

  if (stats.length === 0) return null

  return (
    <div className="mt-4 bg-white border border-gray-200 rounded-lg p-4">
      <h4 className="text-xs font-medium text-gray-700 mb-3">数值列摘要统计（基于预览数据）</h4>
      <div className="overflow-auto">
        <table className="min-w-full text-[10px]">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-2 py-1.5 text-left font-medium text-gray-500">列名</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">N</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">Min</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">Q1</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">Mean</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">Q3</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">Max</th>
              <th className="px-2 py-1.5 text-right font-medium text-gray-500">Std</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {stats.map((s) => s && (
              <tr key={s.col} className="hover:bg-gray-50">
                <td className="px-2 py-1 font-mono text-gray-700">{s.col}</td>
                <td className="px-2 py-1 text-right text-gray-600">{s.n}</td>
                <td className="px-2 py-1 text-right font-mono text-gray-600">{s.min.toFixed(2)}</td>
                <td className="px-2 py-1 text-right font-mono text-gray-600">{s.q1.toFixed(2)}</td>
                <td className="px-2 py-1 text-right font-mono text-primary-600 font-medium">{s.mean.toFixed(2)}</td>
                <td className="px-2 py-1 text-right font-mono text-gray-600">{s.q3.toFixed(2)}</td>
                <td className="px-2 py-1 text-right font-mono text-gray-600">{s.max.toFixed(2)}</td>
                <td className="px-2 py-1 text-right font-mono text-gray-600">{s.std.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
