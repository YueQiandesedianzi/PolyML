# PolyML

![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Electron](https://img.shields.io/badge/Electron-39-47848f.svg)
![React](https://img.shields.io/badge/React-18-61dafb.svg)
![Python](https://img.shields.io/badge/Python-3.11-3776ab.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)

PolyML 是一个聚合物材料机器学习桌面应用，支持从数据导入到模型预测的一站式聚合物性质研究工作流。

## 功能

- 导入 CSV / Excel 数据，自动检测列类型并映射为 SMILES、数值特征或目标变量。
- 基于 RDKit 自动生成 200+ 分子描述符，支持 Van Krevelen 基团贡献和自定义特征公式。
- 一键运行 12 种机器学习模型（Ridge、XGBoost、Gaussian Process 等），Optuna 自动超参优化，K-Fold / LOOCV 交叉验证。
- 可视化预测-实际散点图、残差分析、模型对比和特征重要性。
- 保存训练好的模型，对新聚合物 SMILES 进行性质预测并给出不确定性估计。
- DOE 实验设计（全因子、部分因子、中心复合、Box-Behnken），支持约束条件过滤。
- 主动学习：基于高斯过程回归（GPR）推荐最具信息量的下一组实验。
- AI 智能助手对话，结合项目上下文回答聚合物科学和平台使用问题。
- 内置聚合物 SMILES 数据库，支持搜索、新增和管理。

## 安装

### 下载安装包

Windows 用户可在 GitHub 仓库的 **[Releases](https://github.com/YueQiandesedianzi/PolyML/releases)** 页面下载 `PolyML Setup 0.1.0.exe`。

当前安装包未进行商业代码签名。Windows SmartScreen 如显示未知发布者，请核对 Release 页面提供的信息后再决定是否运行。

### 从源码运行

**环境要求**：Node.js 20+、npm、Anaconda 或 Miniconda（conda 24+）、Python 3.11。

```bash
# 1. 创建 Python 环境
conda env create -f backend/environment.yml
conda activate polyml

# 2. 安装前端依赖
cd frontend
npm install

# 3. 启动开发模式（自动启动后端 + 前端）
npm run dev
```

启动后：
- Python FastAPI 后端运行在 `http://127.0.0.1:18921`
- Electron + React 前端自动打开窗口

常用命令：

```bash
npm run dev          # 开发模式
npm run build        # 构建前端
npm run package      # 打包为 Windows 安装程序
npm run typecheck    # TypeScript 类型检查
```

## 基本流程

1. 创建项目，命名并描述研究问题。
2. 导入 CSV 或 Excel 数据文件，为每列指定类型（SMILES / 数值 / 目标 / 忽略）。
3. 执行特征工程：选择分子描述符、Van Krevelen 基团贡献或自定义特征规则。
4. 进入自动建模，选择模型集合和交叉验证方式，一键训练并对比结果。
5. 在结果分析页查看预测-实际图、残差图、模型性能对比和特征重要性。
6. 保存最佳模型，在预测页输入新聚合物的 SMILES 进行性质预测。

## 项目结构

```text
backend/
  config.py              应用配置（端口、LLM、数据路径）
  main.py                FastAPI 入口
  routers/               API 路由（项目、数据、特征、建模、预测、DOE 等）
  ml/                    ML 核心（训练、评估、描述符、SHAP、主动学习）
  db/                    SQLite 数据库（SQLAlchemy async）
  services/              数据导入、特征工程服务

frontend/
  src/main/              Electron 主进程（窗口管理、Python 子进程）
  src/renderer/
    pages/               页面组件（首页、项目、数据、特征、建模、结果、预测等）
    components/          通用组件（ChatWidget、ErrorBoundary、布局）
    services/            API 调用封装
    store/               Zustand 状态管理
    types/               TypeScript 类型定义
  resources/             应用图标
  electron-builder.yml   打包配置
```

## 技术栈

- **前端**：Electron 39、React 18、TypeScript、Vite 6、Tailwind CSS、Zustand、Plotly.js
- **后端**：Python 3.11、FastAPI、Uvicorn、SQLAlchemy async、aiosqlite
- **ML**：scikit-learn、XGBoost、Optuna、SHAP、RDKit
- **打包**：electron-builder（NSIS 安装程序）

## 已知限制

- Windows 安装包暂未进行代码签名，SmartScreen 可能弹出警告。
- 后端依赖 conda 环境中的 RDKit，用户需先创建 Python 环境才能运行完整功能。
- 分子描述符计算依赖有效的 SMILES 字符串，无效 SMILES 会导致特征工程失败。
- 主动学习的 GPR 模型在数据量极大（>1000 条）时运行较慢。

## License

[MIT](LICENSE)
