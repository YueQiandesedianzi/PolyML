import React, { useState } from 'react'
import { engineerFeatures, validateCustomExpression } from '@/renderer/services/api'
import { useProjectStore } from '@/renderer/store/projectStore'
import { useFeatureStore } from '@/renderer/store/featureStore'
import type { CustomFeatureRule } from '@/renderer/types/api'

const RULE_TYPES = [
  { key: 'formula', label: '公式', desc: '使用列名和算术运算定义特征' },
  { key: 'substructure', label: '子结构计数', desc: '使用 SMARTS 模式匹配计数' },
  { key: 'bin', label: '分箱', desc: '将数值列等频分箱' },
  { key: 'interaction', label: '交互项', desc: '两列乘积' },
  { key: 'domain', label: '预设公式', desc: '聚合物领域预设公式' },
]

export const FeaturesPage: React.FC = () => {
  const { currentProject } = useProjectStore()
  const { customRules, addRule, updateRule, removeRule } = useFeatureStore()
  const [activeTab, setActiveTab] = useState<'builtin' | 'custom'>('builtin')

  // Built-in feature options
  const [includeDescriptors, setIncludeDescriptors] = useState(true)
  const [includeVK, setIncludeVK] = useState(true)
  const [include3d, setInclude3d] = useState(false)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)

  const handleRun = async () => {
    if (!currentProject) return
    setRunning(true)
    setError(null)
    try {
      const res = await engineerFeatures(currentProject.id, {
        includeDescriptors,
        includeVanKrevelen: includeVK,
        include3d,
        customRules: activeTab === 'custom' && customRules.length > 0 ? customRules : undefined,
      })
      setResult(res)
    } catch (e: any) {
      setError(e.message || '特征工程失败')
    } finally {
      setRunning(false)
    }
  }

  return (
    <div className="max-w-3xl">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">特征工程</h2>
      <p className="text-sm text-gray-500 mb-6">
        将聚合物 SMILES 转换为机器学习可用的数值特征。
      </p>

      {!currentProject ? (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-sm text-yellow-700">
          请先创建或选择一个项目。
        </div>
      ) : (
        <>
          {/* Tab bar */}
          <div className="flex gap-1 mb-5 border-b border-gray-200">
            <button
              onClick={() => setActiveTab('builtin')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition ${
                activeTab === 'builtin'
                  ? 'border-primary-500 text-primary-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              内置特征
            </button>
            <button
              onClick={() => setActiveTab('custom')}
              className={`px-4 py-2 text-sm font-medium border-b-2 transition ${
                activeTab === 'custom'
                  ? 'border-primary-500 text-primary-700'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              自定义特征 {customRules.length > 0 && `(${customRules.length})`}
            </button>
          </div>

          {/* Tab: Built-in features */}
          {activeTab === 'builtin' && (
            <div className="bg-white border border-gray-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-gray-700 mb-4">特征组</h3>
              <div className="space-y-3">
                <label className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeDescriptors}
                    onChange={(e) => setIncludeDescriptors(e.target.checked)}
                    className="mt-0.5"
                  />
                  <div>
                    <div className="text-sm text-gray-800">RDKit 分子描述符</div>
                    <div className="text-xs text-gray-500">
                      ~200+ 描述符：组成、拓扑、电子、Lipinski 等。
                    </div>
                  </div>
                </label>

                <label className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeVK}
                    onChange={(e) => setIncludeVK(e.target.checked)}
                    className="mt-0.5"
                  />
                  <div>
                    <div className="text-sm text-gray-800">Van Krevelen 基团贡献法</div>
                    <div className="text-xs text-gray-500">
                      ~50 种结构基团计数，用于 Tg 和性质预测。
                    </div>
                  </div>
                </label>

                <label className="flex items-start gap-3 cursor-pointer opacity-50 cursor-not-allowed">
                  <input type="checkbox" disabled className="mt-0.5" />
                  <div>
                    <div className="text-sm text-gray-800">3D 描述符</div>
                    <div className="text-xs text-gray-500">
                      基于 3D 构象计算，速度较慢。（开发中）
                    </div>
                  </div>
                </label>
              </div>
            </div>
          )}

          {/* Tab: Custom features */}
          {activeTab === 'custom' && (
            <div className="space-y-3">
              {customRules.map((rule, idx) => (
                <CustomRuleCard
                  key={idx}
                  rule={rule}
                  index={idx}
                  projectId={currentProject?.id || ''}
                  onUpdate={(r) => updateRule(idx, r)}
                  onRemove={() => removeRule(idx)}
                />
              ))}

              <button
                onClick={() => addRule({
                  name: '',
                  ruleType: 'formula',
                  expression: '',
                  params: {},
                })}
                className="w-full p-3 border border-dashed border-gray-300 rounded-lg text-sm text-gray-500 hover:border-primary-400 hover:text-primary-600 transition"
              >
                + 添加自定义规则
              </button>

              {customRules.length === 0 && (
                <div className="bg-gray-50 border border-gray-200 rounded-lg p-6 text-center text-sm text-gray-400">
                  暂无自定义规则。点击上方按钮添加。
                </div>
              )}
            </div>
          )}

          {/* Error */}
          {error && <p className="mt-3 text-red-500 text-xs">{error}</p>}

          {/* Run button */}
          <button
            onClick={handleRun}
            disabled={running}
            className="mt-5 bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition"
          >
            {running ? '计算中...' : '运行特征工程'}
          </button>

          {/* Results */}
          {result && (
            <div className="mt-6 bg-green-50 border border-green-200 rounded-lg p-5">
              <h3 className="text-sm font-medium text-green-800 mb-3">特征生成完成</h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">总特征数：</span>
                  <span className="ml-2 font-mono font-medium">{result.X_shape[1]}</span>
                </div>
                <div>
                  <span className="text-gray-500">样本数：</span>
                  <span className="ml-2 font-mono font-medium">{result.X_shape[0]}</span>
                </div>
                <div>
                  <span className="text-gray-500">RDKit 描述符：</span>
                  <span className="ml-2 font-mono">{result.n_descriptors}</span>
                </div>
                <div>
                  <span className="text-gray-500">Van Krevelen：</span>
                  <span className="ml-2 font-mono">{result.n_van_krevelen}</span>
                </div>
                <div>
                  <span className="text-gray-500">工艺参数：</span>
                  <span className="ml-2 font-mono">{result.n_processing}</span>
                </div>
                {result.n_custom > 0 && (
                  <div>
                    <span className="text-gray-500">自定义特征：</span>
                    <span className="ml-2 font-mono">{result.n_custom}</span>
                  </div>
                )}
                <div>
                  <span className="text-gray-500">SMILES 失败：</span>
                  <span className="ml-2 font-mono">{result.n_smiles_failed}</span>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}


/* ---- Custom Rule Card Component ---- */
const CustomRuleCard: React.FC<{
  rule: CustomFeatureRule
  index: number
  projectId: string
  onUpdate: (rule: CustomFeatureRule) => void
  onRemove: () => void
}> = ({ rule, index, projectId, onUpdate, onRemove }) => {
  const [validation, setValidation] = useState<{ valid: boolean; error?: string } | null>(null)

  const handleValidate = async (expr: string) => {
    if (!expr.trim()) { setValidation(null); return }
    try {
      const res = await validateCustomExpression(projectId, {
        ruleType: rule.ruleType,
        expression: expr,
        availableColumns: [],
      })
      setValidation(res)
    } catch {
      setValidation(null)
    }
  }

  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <select
            value={rule.ruleType}
            onChange={(e) => onUpdate({ ...rule, ruleType: e.target.value as any })}
            className="border border-gray-300 rounded px-2 py-1 text-xs"
          >
            {RULE_TYPES.map((t) => (
              <option key={t.key} value={t.key}>{t.label}</option>
            ))}
          </select>
          <input
            type="text"
            value={rule.name}
            onChange={(e) => onUpdate({ ...rule, name: e.target.value })}
            placeholder="特征名称（可选）"
            className="border border-gray-300 rounded px-2 py-1 text-xs w-36"
          />
        </div>
        <button onClick={onRemove} className="text-xs text-red-400 hover:text-red-600">删除</button>
      </div>

      {/* Expression input based on type */}
      {rule.ruleType === 'formula' && (
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={rule.expression}
            onChange={(e) => onUpdate({ ...rule, expression: e.target.value })}
            onBlur={() => handleValidate(rule.expression)}
            placeholder="例如: Mn / PDI 或 Tg - Tm"
            className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-xs font-mono"
          />
          {validation && (
            <span className={`text-xs ${validation.valid ? 'text-green-500' : 'text-red-500'}`}>
              {validation.valid ? '✓' : '✗'}
            </span>
          )}
        </div>
      )}

      {rule.ruleType === 'substructure' && (
        <input
          type="text"
          value={rule.expression}
          onChange={(e) => onUpdate({ ...rule, expression: e.target.value })}
          placeholder="SMARTS 模式，如 c1ccccc1"
          className="w-full border border-gray-300 rounded px-3 py-1.5 text-xs font-mono"
        />
      )}

      {rule.ruleType === 'bin' && (
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={rule.params.column || ''}
            onChange={(e) => onUpdate({ ...rule, expression: e.target.value, params: { ...rule.params, column: e.target.value } })}
            placeholder="列名"
            className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-xs"
          />
          <input
            type="number"
            value={rule.params.n_bins || 5}
            onChange={(e) => onUpdate({ ...rule, params: { ...rule.params, n_bins: Number(e.target.value) } })}
            min={2}
            max={20}
            className="w-16 border border-gray-300 rounded px-2 py-1.5 text-xs"
          />
          <span className="text-[10px] text-gray-400">箱数</span>
        </div>
      )}

      {rule.ruleType === 'interaction' && (
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={rule.params.column_a || ''}
            onChange={(e) => onUpdate({ ...rule, params: { ...rule.params, column_a: e.target.value }, expression: `${e.target.value} * ${rule.params.column_b || ''}` })}
            placeholder="列 A"
            className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-xs"
          />
          <span className="text-xs text-gray-400">×</span>
          <input
            type="text"
            value={rule.params.column_b || ''}
            onChange={(e) => onUpdate({ ...rule, params: { ...rule.params, column_b: e.target.value }, expression: `${rule.params.column_a || ''} * ${e.target.value}` })}
            placeholder="列 B"
            className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-xs"
          />
        </div>
      )}

      {rule.ruleType === 'domain' && (
        <select
          value={rule.expression}
          onChange={(e) => onUpdate({ ...rule, expression: e.target.value })}
          className="w-full border border-gray-300 rounded px-3 py-1.5 text-xs"
        >
          <option value="">选择预设公式...</option>
          <option value="mark_houwink">Mark-Houwink: [eta] = K * Mw^a</option>
          <option value="fox">Fox 方程: 1/Tg = w1/Tg1 + w2/Tg2</option>
          <option value="gordon_taylor">Gordon-Taylor: Tg 混合规则</option>
        </select>
      )}

      {/* Description */}
      <div className="text-[10px] text-gray-400 mt-2">
        {RULE_TYPES.find((t) => t.key === rule.ruleType)?.desc}
      </div>

      {/* Validation error */}
      {validation && !validation.valid && validation.error && (
        <div className="text-[10px] text-red-500 mt-1">{validation.error}</div>
      )}
    </div>
  )
}
