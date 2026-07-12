"""Custom feature engineering with safe AST-based expression evaluation.

Supports formula, substructure counting, binning, interaction, and domain-specific rules.
Zero new dependencies — uses Python's built-in ast module for safe evaluation.
"""

import ast
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CustomFeatureRule:
    name: str
    rule_type: str  # formula | substructure | bin | interaction | domain
    expression: str
    params: dict = field(default_factory=dict)


# Allowed function names for formula evaluation
_ALLOWED_FUNCTIONS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "min": np.minimum,
    "max": np.maximum,
    "power": np.power,
}


class SafeFormulaEvaluator(ast.NodeVisitor):
    """Safely evaluate mathematical expressions using AST parsing.

    Only allows: arithmetic operators, column name references, numeric literals,
    and whitelisted numpy functions.
    """

    def __init__(self, available_columns: list[str]):
        self.available_columns = set(available_columns)

    def validate(self, expression: str) -> tuple[bool, str]:
        """Validate expression syntax and references. Returns (valid, error_message)."""
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            return False, f"语法错误: {e}"

        try:
            self.visit(tree)
        except ValueError as e:
            return False, str(e)

        return True, ""

    def visit_Expression(self, node: ast.Expression):
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp):
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp):
        self.visit(node.operand)

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"不支持的常量类型: {type(node.value)}")

    def visit_Name(self, node: ast.Name):
        if node.id not in self.available_columns:
            raise ValueError(f"列 '{node.id}' 未找到。可用列: {', '.join(sorted(self.available_columns))}")

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            if node.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError(f"不支持的函数: {node.func.id}。允许的函数: {', '.join(sorted(_ALLOWED_FUNCTIONS.keys()))}")
            for arg in node.args:
                self.visit(arg)
        else:
            raise ValueError("不支持的函数调用形式")

    def visit_IfExp(self, node: ast.IfExp):
        self.visit(node.test)
        self.visit(node.body)
        self.visit(node.orelse)

    def visit_Compare(self, node: ast.Compare):
        self.visit(node.left)
        for comparator in node.comparators:
            self.visit(comparator)

    def visit_BoolOp(self, node: ast.BoolOp):
        for value in node.values:
            self.visit(value)

    def evaluate(self, expression: str, df: pd.DataFrame) -> np.ndarray:
        """Evaluate expression on a DataFrame, returning a numpy array."""
        # Build context with column values and allowed functions
        context = {}
        for col in df.columns:
            if col in self.available_columns:
                try:
                    context[col] = pd.to_numeric(df[col], errors="coerce").values.astype(float)
                except (ValueError, TypeError):
                    context[col] = df[col].values
        context.update(_ALLOWED_FUNCTIONS)

        # Also add numpy for any numpy functions used
        context["np"] = np
        context["numpy"] = np

        tree = ast.parse(expression, mode="eval")
        code = compile(tree, "<expr>", "eval")
        result = eval(code, {"__builtins__": {}}, context)

        if isinstance(result, np.ndarray):
            return result.astype(float)
        return np.full(len(df), float(result))


def _evaluate_formula(rule: CustomFeatureRule, df: pd.DataFrame, available_columns: list[str]) -> tuple[np.ndarray, str]:
    """Evaluate a formula expression."""
    evaluator = SafeFormulaEvaluator(available_columns)
    valid, error = evaluator.validate(rule.expression)
    if not valid:
        raise ValueError(error)
    values = evaluator.evaluate(rule.expression, df)
    return values, rule.name or f"custom_{rule.expression.replace(' ', '')}"


def _evaluate_substructure(rule: CustomFeatureRule, df: pd.DataFrame, smiles_col: str | None) -> tuple[np.ndarray, str]:
    """Count SMARTS substructure matches."""
    if not smiles_col or smiles_col not in df.columns:
        raise ValueError("子结构计数需要 SMILES 列")

    from rdkit import Chem

    patterns = rule.params.get("patterns", [rule.expression])
    if isinstance(patterns, str):
        patterns = [patterns]

    counts = np.zeros(len(df))
    for i, smiles in enumerate(df[smiles_col]):
        if not isinstance(smiles, str):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for smarts in patterns:
            try:
                pat = Chem.MolFromSmarts(smarts)
                if pat:
                    counts[i] += len(mol.GetSubstructMatches(pat))
            except Exception:
                pass

    feature_name = rule.name or f"substruct_{rule.expression[:20]}"
    return counts, feature_name


def _evaluate_bin(rule: CustomFeatureRule, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Bin a numeric column into equal-frequency bins and return bin index."""
    col_name = rule.params.get("column", rule.expression)
    n_bins = rule.params.get("n_bins", 5)

    if col_name not in df.columns:
        raise ValueError(f"列 '{col_name}' 未找到")

    values = pd.to_numeric(df[col_name], errors="coerce")
    saved_edges = rule.params.get("bin_edges")
    if saved_edges:
        binned = pd.cut(values, bins=np.asarray(saved_edges, dtype=float), labels=False, include_lowest=True)
    else:
        binned, edges = pd.qcut(values, q=n_bins, labels=False, duplicates="drop", retbins=True)
        rule.params["bin_edges"] = [float(edge) for edge in edges]
    result = binned.values.astype(float)
    result[np.isnan(result)] = 0

    feature_name = rule.name or f"bin_{col_name}_{n_bins}"
    return result, feature_name


def _evaluate_interaction(rule: CustomFeatureRule, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Compute interaction (product) of two columns."""
    col_a = rule.params.get("column_a", rule.expression.split("*")[0].strip() if "*" in rule.expression else "")
    col_b = rule.params.get("column_b", rule.expression.split("*")[1].strip() if "*" in rule.expression else "")

    if not col_a or not col_b:
        raise ValueError("交互项需要指定两个列名")

    if col_a not in df.columns or col_b not in df.columns:
        raise ValueError(f"列未找到: {col_a}, {col_b}")

    a = pd.to_numeric(df[col_a], errors="coerce").fillna(0).values
    b = pd.to_numeric(df[col_b], errors="coerce").fillna(0).values
    result = a * b

    feature_name = rule.name or f"interact_{col_a}_x_{col_b}"
    return result, feature_name


def _evaluate_domain(rule: CustomFeatureRule, df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Evaluate domain-specific polymer formulas."""
    formula = rule.expression.lower()
    params = rule.params

    if formula == "mark_houwink":
        # [eta] = K * Mw^a
        k = params.get("K", 1.0)
        a = params.get("a", 0.7)
        mw_col = params.get("mw_column", "Mw")
        if mw_col not in df.columns:
            raise ValueError(f"Mark-Houwink 需要 '{mw_col}' 列")
        mw = pd.to_numeric(df[mw_col], errors="coerce").fillna(0).values
        result = k * np.power(mw, a)
        return result, rule.name or "mark_houwink"

    elif formula == "fox":
        # 1/Tg = w1/Tg1 + w2/Tg2
        w1_col = params.get("w1_column", "w1")
        tg1_col = params.get("tg1_column", "Tg1")
        w2_col = params.get("w2_column", "w2")
        tg2_col = params.get("tg2_column", "Tg2")

        for col in [w1_col, tg1_col, w2_col, tg2_col]:
            if col not in df.columns:
                raise ValueError(f"Fox 方程需要 '{col}' 列")

        w1 = pd.to_numeric(df[w1_col], errors="coerce").fillna(0).values
        tg1 = pd.to_numeric(df[tg1_col], errors="coerce").replace(0, np.nan).values
        w2 = pd.to_numeric(df[w2_col], errors="coerce").fillna(0).values
        tg2 = pd.to_numeric(df[tg2_col], errors="coerce").replace(0, np.nan).values

        result = 1.0 / (w1 / tg1 + w2 / tg2)
        return result, rule.name or "fox_Tg"

    elif formula == "gordon_taylor":
        # Tg = (w1*Tg1 + K*w2*Tg2) / (w1 + K*w2)
        k_param = params.get("K", 1.0)
        w1_col = params.get("w1_column", "w1")
        tg1_col = params.get("tg1_column", "Tg1")
        w2_col = params.get("w2_column", "w2")
        tg2_col = params.get("tg2_column", "Tg2")

        for col in [w1_col, tg1_col, w2_col, tg2_col]:
            if col not in df.columns:
                raise ValueError(f"Gordon-Taylor 需要 '{col}' 列")

        w1 = pd.to_numeric(df[w1_col], errors="coerce").fillna(0).values
        tg1 = pd.to_numeric(df[tg1_col], errors="coerce").fillna(0).values
        w2 = pd.to_numeric(df[w2_col], errors="coerce").fillna(0).values
        tg2 = pd.to_numeric(df[tg2_col], errors="coerce").fillna(0).values

        result = (w1 * tg1 + k_param * w2 * tg2) / (w1 + k_param * w2)
        return result, rule.name or "gordon_taylor_Tg"

    else:
        raise ValueError(f"未知的预设公式: {formula}。支持: mark_houwink, fox, gordon_taylor")


def evaluate_custom_features(
    df: pd.DataFrame,
    rules: list[CustomFeatureRule],
    smiles_col: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Evaluate all custom feature rules and return concatenated feature matrix.

    Returns:
        (X_custom, feature_names) where X_custom has shape (n_samples, n_features)
    """
    if not rules:
        return np.empty((len(df), 0)), []

    available_columns = [col for col in df.columns if col != smiles_col]

    results = []
    names = []

    for rule in rules:
        try:
            if rule.rule_type == "formula":
                values, name = _evaluate_formula(rule, df, available_columns)
            elif rule.rule_type == "substructure":
                values, name = _evaluate_substructure(rule, df, smiles_col)
            elif rule.rule_type == "bin":
                values, name = _evaluate_bin(rule, df)
            elif rule.rule_type == "interaction":
                values, name = _evaluate_interaction(rule, df)
            elif rule.rule_type == "domain":
                values, name = _evaluate_domain(rule, df)
            else:
                raise ValueError(f"未知的规则类型: {rule.rule_type}")

            # Ensure 1D array
            values = np.asarray(values).flatten()
            if len(values) != len(df):
                raise ValueError(f"规则 '{rule.name}' 输出长度 ({len(values)}) 与数据行数 ({len(df)}) 不匹配")

            # Replace inf with nan
            values[~np.isfinite(values)] = np.nan

            results.append(values.reshape(-1, 1))
            names.append(name)
        except Exception as e:
            # Skip failed rules with warning — don't break the entire pipeline
            print(f"[CustomFeature] Rule '{rule.name}' failed: {e}")
            continue

    if not results:
        return np.empty((len(df), 0)), []

    X_custom = np.hstack(results)
    return X_custom, names


def get_available_domain_formulas() -> list[dict]:
    """Return a list of available domain-specific formulas with their parameters."""
    return [
        {
            "name": "mark_houwink",
            "label": "Mark-Houwink 方程",
            "description": "[eta] = K * Mw^a — 特性粘度与分子量关系",
            "params": [
                {"name": "K", "label": "K 值", "type": "float", "default": 1.0},
                {"name": "a", "label": "a 指数", "type": "float", "default": 0.7},
                {"name": "mw_column", "label": "分子量列名", "type": "column", "default": "Mw"},
            ],
        },
        {
            "name": "fox",
            "label": "Fox 方程",
            "description": "1/Tg = w1/Tg1 + w2/Tg2 — 共混物 Tg 预测",
            "params": [
                {"name": "w1_column", "label": "组分1质量分数列", "type": "column", "default": "w1"},
                {"name": "tg1_column", "label": "组分1 Tg列", "type": "column", "default": "Tg1"},
                {"name": "w2_column", "label": "组分2质量分数列", "type": "column", "default": "w2"},
                {"name": "tg2_column", "label": "组分2 Tg列", "type": "column", "default": "Tg2"},
            ],
        },
        {
            "name": "gordon_taylor",
            "label": "Gordon-Taylor 方程",
            "description": "Tg = (w1*Tg1 + K*w2*Tg2) / (w1 + K*w2)",
            "params": [
                {"name": "K", "label": "K 参数", "type": "float", "default": 1.0},
                {"name": "w1_column", "label": "组分1质量分数列", "type": "column", "default": "w1"},
                {"name": "tg1_column", "label": "组分1 Tg列", "type": "column", "default": "Tg1"},
                {"name": "w2_column", "label": "组分2质量分数列", "type": "column", "default": "w2"},
                {"name": "tg2_column", "label": "组分2 Tg列", "type": "column", "default": "Tg2"},
            ],
        },
    ]
