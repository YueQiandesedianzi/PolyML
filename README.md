# PolyML

Polymer Materials Machine Learning Desktop App

## Prerequisites

- Node.js 20+
- npm or pnpm
- Anaconda or Miniconda (conda 24+)
- Python 3.11 (via conda)

## Setup

### 1. Create Python environment

```bash
conda env create -f backend/environment.yml
conda activate polyml
```

### 2. Install frontend dependencies

```bash
cd frontend
npm install
```

### 3. Run development

```bash
cd frontend
npm run dev
```

This will:
- Start the Python FastAPI backend on `localhost:18921`
- Start the Electron + React frontend

## Architecture

- **Frontend**: Electron + React + TypeScript + Tailwind
- **Backend**: Python FastAPI (as a subprocess managed by Electron)
- **ML**: scikit-learn, XGBoost, Optuna, SHAP, RDKit
- **DB**: SQLite (via SQLAlchemy async + aiosqlite)

## Data Flow

1. User imports CSV/Excel → column type auto-detection
2. Feature engineering: RDKit 200+ descriptors + Van Krevelen group contributions
3. AutoML: 6 models × Optuna hyperparameter optimization × K-fold CV
4. Results: Parity plot, feature importance, SHAP summary
5. Prediction: SMILES + processing params → predicted value + uncertainty
