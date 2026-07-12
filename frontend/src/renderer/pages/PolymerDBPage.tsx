import React, { useState, useEffect } from 'react'
import { listAllPolymers, addPolymer, deletePolymer } from '@/renderer/services/api'
import type { PolymerEntry } from '@/renderer/types/api'

export const PolymerDBPage: React.FC = () => {
  const [polymers, setPolymers] = useState<PolymerEntry[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [search, setSearch] = useState('')
  const [newName, setNewName] = useState('')
  const [newAbbrev, setNewAbbrev] = useState('')
  const [newSmiles, setNewSmiles] = useState('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { loadPolymers() }, [page])

  const loadPolymers = async () => {
    try {
      const { items, total: t } = await listAllPolymers(page, 50)
      setPolymers(items)
      setTotal(t)
    } catch { setPolymers([]) }
  }

  const handleAdd = async () => {
    if (!newName.trim() || !newSmiles.trim()) return
    setError(null)
    try {
      await addPolymer({
        commonName: newName.trim(),
        abbreviation: newAbbrev.trim() || undefined,
        smiles: newSmiles.trim(),
      })
      setNewName(''); setNewAbbrev(''); setNewSmiles('')
      loadPolymers()
    } catch (e: any) { setError(e.message) }
  }

  const handleDelete = async (id: number) => {
    if (!confirm('确定删除此聚合物条目？')) return
    try {
      await deletePolymer(id)
      loadPolymers()
    } catch (e: any) { setError(e.message) }
  }

  const filtered = search
    ? polymers.filter(
        (p) =>
          p.commonName.toLowerCase().includes(search.toLowerCase()) ||
          (p.abbreviation || '').toLowerCase().includes(search.toLowerCase())
      )
    : polymers

  return (
    <div className="max-w-4xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">聚合物数据库</h2>
      <p className="text-sm text-gray-500 mb-5">
        浏览和管理内置的聚合物名称 → SMILES 映射数据库。
      </p>

      {/* Add new polymer */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
        <h3 className="text-sm font-medium text-gray-700 mb-3">添加新聚合物</h3>
        <div className="flex gap-2 mb-2">
          <input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="聚合物名称" className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm" />
          <input value={newAbbrev} onChange={(e) => setNewAbbrev(e.target.value)} placeholder="缩写" className="w-28 border border-gray-300 rounded px-3 py-2 text-sm" />
        </div>
        <div className="flex gap-2">
          <input value={newSmiles} onChange={(e) => setNewSmiles(e.target.value)} placeholder="SMILES（如：*CC(*)c1ccccc1）" className="flex-1 border border-gray-300 rounded px-3 py-2 text-sm font-mono" />
          <button onClick={handleAdd} disabled={!newName.trim() || !newSmiles.trim()} className="bg-primary-600 text-white px-4 py-2 rounded text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition">
            添加
          </button>
        </div>
        {error && <p className="text-red-500 text-xs mt-2">{error}</p>}
      </div>

      {/* Search + list */}
      <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="按名称或缩写搜索..."
            className="w-64 border border-gray-300 rounded px-3 py-1.5 text-sm"
          />
          <span className="text-xs text-gray-400">{total} 种聚合物</span>
        </div>
        <table className="min-w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">名称</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">缩写</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">SMILES</th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500">来源</th>
              <th className="px-4 py-2"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100">
            {filtered.map((p) => (
              <tr key={p.id} className="hover:bg-gray-50">
                <td className="px-4 py-2 font-medium text-gray-800">{p.commonName}</td>
                <td className="px-4 py-2 text-gray-600">{p.abbreviation}</td>
                <td className="px-4 py-2 font-mono text-[10px] text-gray-500 max-w-xs truncate">{p.smiles}</td>
                <td className="px-4 py-2">
                  <span className={`text-[10px] px-1.5 py-0.5 rounded ${p.source === 'builtin' ? 'bg-blue-50 text-blue-600' : 'bg-green-50 text-green-600'}`}>
                    {p.source}
                  </span>
                </td>
                <td className="px-4 py-2">
                  <button onClick={() => handleDelete(p.id)} className="text-xs text-red-400 hover:text-red-600">删除</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {/* Pagination */}
        {total > 50 && (
          <div className="px-4 py-2 border-t border-gray-200 flex justify-between text-xs text-gray-500">
            <button onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1} className="hover:text-primary-600 disabled:opacity-40">← 上一页</button>
            <span>第 {page} 页 / 共 {Math.ceil(total / 50)} 页</span>
            <button onClick={() => setPage((p) => p + 1)} disabled={page * 50 >= total} className="hover:text-primary-600 disabled:opacity-40">下一页 →</button>
          </div>
        )}
      </div>
    </div>
  )
}
