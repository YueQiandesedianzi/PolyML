"""Q&A Agent — LLM-powered chat assistant for polymer materials science."""

import json
import httpx
from pathlib import Path
from fastapi import APIRouter, HTTPException
from schemas.base import CamelModel

router = APIRouter(prefix="/api/agent", tags=["agent"])


class ChatMessage(CamelModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(CamelModel):
    messages: list[ChatMessage]
    project_id: str | None = None


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是 PolyML 智能助手，一位聚合物材料科学与数据驱动建模领域的专家。

## 你的能力
- 回答聚合物科学问题（分子设计、结构-性能关系、加工工艺等）
- 指导用户使用 PolyML 平台完成数据分析和建模流程
- 根据用户当前项目状态给出针对性建议

## PolyML 工作流
1. **数据导入** — 上传 CSV/Excel，自动检测列类型（SMILES/数值/目标列）
2. **列映射** — 指定哪些列是 SMILES、数值特征、目标变量
3. **实验设计 (DOE)** — 可选，拉丁超方、Box-Behnken 等方法生成实验方案
4. **特征工程** — RDKit 分子描述符 + Van Krevelen 基团贡献 + 加工参数特征 + 自定义公式
5. **自动建模 (AutoML)** — Ridge, Lasso, RF, XGBoost, SVM, GP 等，Optuna 调参，交叉验证
6. **结果分析** — 预测vs实际图、残差分析、模型对比、相关性、特征重要性
7. **预测** — 对新数据进行预测
8. **主动学习** — 贝叶斯优化推荐下一个实验
9. **代码导出** — 导出可复现的 Python 脚本

## 当前项目上下文
{context}

## 回答规则
- 用用户的语言回答（中文问就中文答，英文问就英文答）
- 算专业但易懂，适当给出解释
- 主动建议下一步操作（基于项目状态）
- 如果用户没问具体问题，先简短问候，然后主动介绍你能做什么"""


def _build_project_context(project_id: str) -> str:
    """Read project state from filesystem and build a context string."""
    from config import settings
    proj_dir = settings.projects_path / project_id
    if not proj_dir.exists():
        return "项目不存在。"

    lines = []

    # project.json
    meta_path = proj_dir / "project.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        lines.append(f"项目名称: {meta.get('name', '未知')}")
        if meta.get("description"):
            lines.append(f"描述: {meta['description']}")
        if meta.get("data_filename"):
            lines.append(f"数据文件: {meta['data_filename']}")
        if meta.get("data_row_count"):
            lines.append(f"数据行数: {meta['data_row_count']}")
        if meta.get("target_column"):
            lines.append(f"目标列: {meta['target_column']}")
        if meta.get("smiles_column"):
            lines.append(f"SMILES列: {meta['smiles_column']}")
        if meta.get("numeric_columns"):
            lines.append(f"数值特征列: {', '.join(meta['numeric_columns'])}")
        fc = meta.get("feature_count", 0)
        if fc:
            lines.append(f"已计算特征数: {fc}")

    # column_mapping.json
    mapping_path = proj_dir / "column_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, encoding="utf-8") as f:
            mapping = json.load(f)
        summary = ", ".join(f"{k}={v}" for k, v in mapping.items())
        lines.append(f"列映射: {summary}")

    # features.npz — just check existence
    features_path = proj_dir / "features.npz"
    if features_path.exists():
        try:
            import numpy as np
            data = np.load(features_path, allow_pickle=True)
            X = data["X"]
            names = data["feature_names"].tolist() if "feature_names" in data else []
            lines.append(f"特征矩阵: {X.shape[0]} 样本 × {X.shape[1]} 特征")
            if names:
                lines.append(f"特征名(前10): {', '.join(names[:10])}")
        except Exception:
            lines.append("特征矩阵: 已计算")

    # training_runs.json
    runs_path = proj_dir / "training_runs.json"
    if runs_path.exists():
        with open(runs_path, encoding="utf-8") as f:
            runs = json.load(f)
        if runs:
            last = runs[-1]
            lines.append(f"训练次数: {len(runs)}")
            best = last.get("best_model", "未知")
            lines.append(f"最近最佳模型: {best}")
            results = last.get("results", {})
            if best in results:
                r = results[best]
                def _fmt(val, key):
                    v = r.get(key, val)
                    return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                lines.append(
                    f"最近训练结果: R²={_fmt('?', 'test_r2')}, "
                    f"RMSE={_fmt('?', 'test_rmse')}, "
                    f"MAE={_fmt('?', 'test_mae')}"
                )

    # Suggest next steps based on what's done
    steps_done = []
    if meta_path.exists():
        with open(meta_path) as f:
            m = json.load(f)
        if m.get("data_filename"):
            steps_done.append("data")
        if m.get("feature_count", 0) > 0:
            steps_done.append("features")
        if runs_path.exists():
            steps_done.append("train")

    suggestions = []
    if "data" not in steps_done:
        suggestions.append("→ 下一步: 导入数据")
    elif "features" not in steps_done:
        suggestions.append("→ 下一步: 进行特征工程")
    elif "train" not in steps_done:
        suggestions.append("→ 下一步: 运行 AutoML 训练")
    else:
        suggestions.append("→ 可以: 查看结果/预测/主动学习/导出代码")

    if suggestions:
        lines.append("")
        lines.extend(suggestions)

    return "\n".join(lines) if lines else "项目数据为空。"


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------
@router.post("/chat")
async def chat(req: ChatRequest):
    from config import settings

    if not settings.llm_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM API key 未配置。请设置环境变量 POLYML_LLM_API_KEY。",
        )

    # Build project context
    project_context = ""
    if req.project_id:
        project_context = _build_project_context(req.project_id)

    system_prompt = SYSTEM_PROMPT.format(
        context=project_context or "用户尚未选择项目，或当前无项目。"
    )

    # Assemble messages (keep last 20 to avoid context overflow)
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in req.messages[-20:]:
        api_messages.append({"role": msg.role, "content": msg.content})

    # Call LLM API
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{settings.llm_base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "messages": api_messages,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
            )
            response.raise_for_status()
            data = response.json()
            reply = data["choices"][0]["message"]["content"]
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"LLM API 返回错误 ({e.response.status_code}): {e.response.text[:300]}",
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API 调用失败: {str(e)}")

    return {"reply": reply}
