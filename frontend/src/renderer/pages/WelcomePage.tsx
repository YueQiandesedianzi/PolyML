import React, { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useProjectStore } from '@/renderer/store/projectStore'
import { listProjects } from '@/renderer/services/api'

const STEPS = [
  { icon: '📁', title: '创建项目', desc: '命名并描述你的研究问题', route: '/project' },
  { icon: '📊', title: '导入数据', desc: '上传 CSV 或 Excel 格式的聚合物数据', route: '/data' },
  { icon: '⚙️', title: '特征工程', desc: '从 SMILES 自动生成分子描述符', route: '/features' },
  { icon: '🤖', title: '自动建模', desc: '一键比较 12 种机器学习模型', route: '/automl' },
  { icon: '📈', title: '分析结果', desc: '可视化预测精度和特征重要性', route: '/results' },
  { icon: '🎯', title: '预测新分子', desc: '用训练好的模型预测未知聚合物性质', route: '/predict' },
]

export const WelcomePage: React.FC = () => {
  const navigate = useNavigate()
  const { currentProject, setProjects } = useProjectStore()

  useEffect(() => {
    listProjects().then((list) => setProjects(list)).catch(() => {})
  }, [])

  return (
    <div className="max-w-4xl mx-auto py-10">
      {/* Hero */}
      <div className="text-center mb-12">
        <div className="text-6xl mb-4">🧪</div>
        <h1 className="text-3xl font-bold text-gray-800 mb-3">PolyML</h1>
        <p className="text-lg text-gray-500 mb-1">让聚合物材料机器学习变得简单</p>
        <p className="text-sm text-gray-400 mt-2 max-w-lg mx-auto">
          从数据导入到模型预测，一站式完成聚合物性质的机器学习研究。
        </p>
      </div>

      {/* Quick start prompt */}
      {currentProject ? (
        <div className="bg-primary-50 border border-primary-200 rounded-xl p-5 mb-10 flex items-center justify-between">
          <div>
            <div className="text-sm font-medium text-primary-800">
              当前项目: {currentProject.name}
            </div>
            <div className="text-xs text-primary-600 mt-1">
              {currentProject.dataRowCount > 0
                ? `已有 ${currentProject.dataRowCount} 行数据`
                : '还没有导入数据'}
              {currentProject.description && ` · ${currentProject.description}`}
            </div>
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => navigate(currentProject.dataRowCount > 0 ? '/features' : '/data')}
              className="bg-primary-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-primary-700 transition"
            >
              {currentProject.dataRowCount > 0 ? '继续工作' : '导入数据'}
            </button>
          </div>
        </div>
      ) : (
        <div className="bg-gray-50 border border-gray-200 rounded-xl p-5 mb-10 text-center">
          <p className="text-sm text-gray-600 mb-3">还没有项目？从这里开始：</p>
          <button
            onClick={() => navigate('/project')}
            className="bg-primary-600 text-white px-5 py-2 rounded-md text-sm font-medium hover:bg-primary-700 transition"
          >
            创建第一个项目
          </button>
        </div>
      )}

      {/* Workflow steps */}
      <div className="mb-10">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-4">工作流程</h2>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {STEPS.map((step, i) => {
            const isActive = currentProject && (
              (currentProject.dataRowCount === 0 && step.route === '/data') ||
              (currentProject.dataRowCount > 0 && step.route === '/features')
            )
            return (
              <button
                key={step.route}
                onClick={() => navigate(step.route)}
                className={`text-left p-4 rounded-xl border transition group ${
                  isActive
                    ? 'border-primary-300 bg-primary-50 ring-2 ring-primary-200'
                    : 'border-gray-200 bg-white hover:border-primary-200 hover:shadow-sm'
                }`}
              >
                <div className="flex items-start gap-3">
                  <span className="text-xl shrink-0 mt-0.5">{step.icon}</span>
                  <div>
                    <div className="text-sm font-medium text-gray-800 group-hover:text-primary-600 transition">
                      {step.title}
                    </div>
                    <div className="text-xs text-gray-400 mt-1 leading-relaxed">{step.desc}</div>
                  </div>
                </div>
                {i < STEPS.length - 1 && (
                  <div className="hidden md:block absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 text-gray-300 text-lg pointer-events-none">
                    →
                  </div>
                )}
              </button>
            )
          })}
        </div>
      </div>

      {/* Features */}
      <div className="grid grid-cols-3 gap-6 text-center pt-6 border-t border-gray-100">
        <div>
          <div className="text-2xl mb-2">⚡</div>
          <div className="text-xs font-medium text-gray-700">12 种模型</div>
          <div className="text-[10px] text-gray-400">Ridge → MLP 一键比较</div>
        </div>
        <div>
          <div className="text-2xl mb-2">🧬</div>
          <div className="text-xs font-medium text-gray-700">分子描述符</div>
          <div className="text-[10px] text-gray-400">RDKit 自动生成</div>
        </div>
        <div>
          <div className="text-2xl mb-2">🔬</div>
          <div className="text-xs font-medium text-gray-700">主动学习</div>
          <div className="text-[10px] text-gray-400">GPR 指导实验设计</div>
        </div>
      </div>
    </div>
  )
}
