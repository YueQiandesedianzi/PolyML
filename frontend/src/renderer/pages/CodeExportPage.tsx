import React, { useState } from 'react'
import { api } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'

export const CodeExportPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const [script, setScript] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [includeComments, setIncludeComments] = useState(true)
  const [copied, setCopied] = useState(false)

  const handleGenerate = async () => {
    if (!currentProject) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.post(`/projects/${currentProject.id}/code-export/generate`, {
        pipeline: 'full',
        includeComments,
      })
      setScript(res.data.script)
    } catch (e: any) {
      setError(e.response?.data?.detail || e.message)
    } finally {
      setLoading(false)
    }
  }

  const handleCopy = async () => {
    await navigator.clipboard.writeText(script)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleDownload = () => {
    if (!currentProject || !script) return
    const blob = new Blob([script], { type: 'text/x-python' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `polyml_reproduce_${currentProject.id.slice(0, 8)}.py`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="max-w-5xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">代码导出</h2>
      <p className="text-sm text-gray-500 mb-6">导出可复现的 Python 脚本，包含完整的特征工程和训练流程。</p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : (
        <>
          <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
            <div className="flex gap-4 items-center">
              <label className="flex items-center gap-2 text-sm cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeComments}
                  onChange={(e) => setIncludeComments(e.target.checked)}
                />
                包含代码注释
              </label>
              <button
                onClick={handleGenerate}
                disabled={loading}
                className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
              >
                {loading ? '生成中...' : '生成脚本'}
              </button>
              {script && (
                <>
                  <button
                    onClick={handleCopy}
                    className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm hover:bg-gray-50 transition"
                  >
                    {copied ? '已复制!' : '复制到剪贴板'}
                  </button>
                  <button
                    onClick={handleDownload}
                    className="border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm hover:bg-gray-50 transition"
                  >
                    下载 .py 文件
                  </button>
                </>
              )}
            </div>
            {error && <p className="mt-3 text-red-500 text-xs">{error}</p>}
          </div>

          {script && (
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <div className="px-4 py-2 bg-gray-800 flex justify-between items-center">
                <span className="text-xs text-gray-400 font-mono">polyml_reproduce.py</span>
                <span className="text-xs text-gray-500">{script.split('\n').length} 行</span>
              </div>
              <pre className="p-4 text-xs text-green-400 font-mono overflow-auto max-h-[600px]">
                {script}
              </pre>
            </div>
          )}
        </>
      )}
    </div>
  )
}
