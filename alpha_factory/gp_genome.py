"""
GP expression trees for alpha generation.

Variable-depth expression trees with the full AlphaGen operator set (28 ops).
Evaluation produces (n_stocks, n_days) signal matrix for proper IC computation.
"""

import copy
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import rankdata

from .operators import (
    ALL_OPS, OP_DICT, UNARY_OPS, BINARY_OPS, FEATURES, WINDOWS,
    returns, _safe_div,
)
from .evaluate import (
    evaluate_signals, compute_forward_returns, AlphaMetrics,
)


@dataclass
class Node:
    op: str | None = None
    feature: str | None = None
    param: int = 0
    children: list = field(default_factory=list)

    def __str__(self) -> str:
        if self.op is None:
            return self.feature or "?"
        args = ", ".join(str(c) for c in self.children)
        if self.param > 0:
            return f"{self.op}({args}, {self.param})"
        return f"{self.op}({args})"

    def depth(self) -> int:
        if not self.children:
            return 0
        return 1 + max(c.depth() for c in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)

    def copy(self) -> "Node":
        return copy.deepcopy(self)


def random_tree(max_depth: int = 4, rng: random.Random | None = None) -> Node:
    if rng is None:
        rng = random.Random()
    if max_depth <= 0 or (max_depth <= 1 and rng.random() < 0.5):
        return Node(feature=rng.choice(FEATURES))
    if rng.random() < 0.3:
        return Node(feature=rng.choice(FEATURES))

    if rng.random() < 0.6:
        op_entry = rng.choice(UNARY_OPS)
    else:
        op_entry = rng.choice(BINARY_OPS)

    name, arity, _, has_param, param_range = op_entry
    param = rng.choice(WINDOWS) if has_param else 0
    if has_param and param_range:
        param = rng.randint(*param_range)
    children = [random_tree(max_depth - 1, rng) for _ in range(arity)]
    return Node(op=name, param=param, children=children)


def mutate(tree: Node, rng: random.Random | None = None, max_depth: int = 5) -> Node:
    if rng is None:
        rng = random.Random()
    tree = tree.copy()
    r = rng.random()

    if r < 0.35:
        nodes = _collect_nodes(tree)
        if nodes:
            target = rng.choice(nodes)
            replacement = random_tree(max_depth=2, rng=rng)
            _replace_node(target, replacement)
    elif r < 0.65:
        nodes = _collect_nodes(tree)
        if nodes:
            target = rng.choice(nodes)
            if target.op is None:
                target.feature = rng.choice(FEATURES)
            else:
                arity = len(target.children)
                candidates = [op for op in ALL_OPS if op[1] == arity]
                if candidates:
                    new_op = rng.choice(candidates)
                    target.op = new_op[0]
                    if new_op[3]:
                        target.param = rng.randint(*new_op[4]) if new_op[4] else rng.choice(WINDOWS)
                    else:
                        target.param = 0
    else:
        nodes = [n for n in _collect_nodes(tree) if n.param > 0]
        if nodes:
            target = rng.choice(nodes)
            op_entry = OP_DICT.get(target.op)
            if op_entry and op_entry[3] and op_entry[4]:
                lo, hi = op_entry[4]
                target.param = max(lo, min(hi, target.param + rng.randint(-5, 5)))

    if tree.depth() > max_depth:
        return random_tree(max_depth=3, rng=rng)
    return tree


def crossover(parent1: Node, parent2: Node, rng: random.Random | None = None) -> Node:
    if rng is None:
        rng = random.Random()
    child = parent1.copy()
    donor = parent2.copy()
    child_nodes = _collect_nodes(child)
    donor_nodes = _collect_nodes(donor)
    if child_nodes and donor_nodes:
        target = rng.choice(child_nodes)
        source = rng.choice(donor_nodes)
        _replace_node(target, source.copy())
    if child.depth() > 5:
        return random_tree(max_depth=3, rng=rng)
    return child


def _collect_nodes(tree: Node) -> list[Node]:
    result = [tree]
    for c in tree.children:
        result.extend(_collect_nodes(c))
    return result


def _replace_node(target: Node, replacement: Node) -> None:
    target.op = replacement.op
    target.feature = replacement.feature
    target.param = replacement.param
    target.children = replacement.children


# ── Cross-sectional operator names (need matrix-level eval) ───────────
_CS_OPS = {"cs_abs", "cs_log", "cs_sign", "cs_rank"}
_CS_BINARY_OPS = {"add", "sub", "mul", "div", "pow", "greater", "less"}

# Map CS op names to functions that work on (n_stocks,) slices
# Per-day CS functions (applied to 1D stock vector)
_CS_RANK_FN = lambda x: rankdata(
    np.where(~np.isnan(x), x, 0), nan_policy="omit"
) / max(np.sum(~np.isnan(x)), 1)

# Elementwise CS ops that can be applied to full (n_stocks, n_days) matrix
_CS_ELEMENTWISE = {
    "cs_abs": lambda m: np.abs(m),
    "cs_log": lambda m: np.where(m > 0, np.log(m), 0.0),
    "cs_sign": lambda m: np.sign(m),
}

_CS_BIN_FUNCS = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "mul": lambda x, y: x * y,
    "div": lambda x, y: _safe_div(x, y),
    "pow": lambda x, y: np.sign(x) * np.power(np.abs(x) + 1e-10, np.clip(y, -3, 3)),
    "greater": lambda x, y: np.maximum(x, y),
    "less": lambda x, y: np.minimum(x, y),
}


def _eval_node_1d(node: Node, data: dict[str, np.ndarray]) -> np.ndarray | None:
    """Evaluate a time-series node on one stock's data. Returns 1D (n_days,).

    Only handles TS operators and leaf features. CS operators are handled
    at the matrix level by _eval_node_matrix.
    """
    if node.op is None:
        if node.feature == "returns":
            return returns(data["close"])
        elif node.feature == "vwap":
            return _safe_div(
                (data["high"] + data["low"] + data["close"]) * data["volume"],
                data["volume"] * 3,
            )
        return data.get(node.feature)

    # If this is a CS op, it should be handled by matrix eval. This path
    # is only reached if a CS op is nested inside another CS op's children
    # without going through compute_signals. Evaluate children as 1D and
    # apply the function as a passthrough (per-stock, imperfect but safe).
    child_vals = []
    for c in node.children:
        v = _eval_node_1d(c, data)
        if v is None:
            return None
        child_vals.append(v)

    op_entry = OP_DICT.get(node.op)
    if op_entry is None:
        return None

    _, arity, func, has_param, _ = op_entry
    try:
        if has_param:
            if arity == 1:
                return func(child_vals[0], node.param)
            else:
                return func(child_vals[0], child_vals[1], node.param)
        else:
            if arity == 1:
                return func(child_vals[0])
            else:
                return func(child_vals[0], child_vals[1])
    except Exception:
        return None


def _eval_node_matrix(
    node: Node,
    stock_data: dict[str, dict[str, np.ndarray]],
    tickers: list[str],
    n_days: int,
) -> np.ndarray | None:
    """Evaluate a node on the full (n_stocks, n_days) matrix.

    CS operators are applied cross-sectionally (across stocks per day).
    TS operators and leaves are evaluated per-stock then stacked.
    """
    n_stocks = len(tickers)

    # Leaf node: build matrix from per-stock data
    if node.op is None:
        mat = np.full((n_stocks, n_days), np.nan)
        for i, ticker in enumerate(tickers):
            if node.feature == "returns":
                v = returns(stock_data[ticker]["close"])
            elif node.feature == "vwap":
                d = stock_data[ticker]
                v = _safe_div(
                    (d["high"] + d["low"] + d["close"]) * d["volume"],
                    d["volume"] * 3,
                )
            else:
                v = stock_data[ticker].get(node.feature)
            if v is not None and len(v) >= n_days:
                mat[i] = v[:n_days]
        return mat

    op_name = node.op

    # CS unary: recurse children as matrices, apply cross-sectionally
    if op_name in _CS_OPS:
        if len(node.children) != 1:
            return None
        child_mat = _eval_node_matrix(node.children[0], stock_data, tickers, n_days)
        if child_mat is None:
            return None

        # Elementwise ops: apply to whole matrix (fast)
        if op_name in _CS_ELEMENTWISE:
            return _CS_ELEMENTWISE[op_name](child_mat)

        # cs_rank: must rank per-day (vectorized via argsort)
        if op_name == "cs_rank":
            result = np.full_like(child_mat, np.nan)
            valid_mask = ~np.isnan(child_mat)
            for t in range(n_days):
                col = child_mat[:, t]
                v = valid_mask[:, t]
                n_valid = v.sum()
                if n_valid < 2:
                    result[:, t] = 0.5
                    continue
                order = np.argsort(col)
                ranks = np.empty(n_stocks, dtype=np.float64)
                ranks[order] = np.arange(1, n_stocks + 1)
                result[:, t] = ranks / n_stocks
            return result

        return None

    # CS binary: recurse both children as matrices, apply cross-sectionally
    if op_name in _CS_BINARY_OPS:
        if len(node.children) != 2:
            return None
        left = _eval_node_matrix(node.children[0], stock_data, tickers, n_days)
        right = _eval_node_matrix(node.children[1], stock_data, tickers, n_days)
        if left is None or right is None:
            return None
        func = _CS_BIN_FUNCS[op_name]
        try:
            return func(left, right)
        except Exception:
            return None

    # TS operator: recurse children as matrices, apply per-stock (row)
    child_mats = []
    for c in node.children:
        m = _eval_node_matrix(c, stock_data, tickers, n_days)
        if m is None:
            return None
        child_mats.append(m)

    op_entry = OP_DICT.get(op_name)
    if op_entry is None:
        return None

    _, arity, func, has_param, _ = op_entry
    result = np.full((n_stocks, n_days), np.nan)
    for i in range(n_stocks):
        try:
            if has_param:
                if arity == 1:
                    row = func(child_mats[0][i], node.param)
                else:
                    row = func(child_mats[0][i], child_mats[1][i], node.param)
            else:
                if arity == 1:
                    row = func(child_mats[0][i])
                else:
                    row = func(child_mats[0][i], child_mats[1][i])
            if row is not None and len(row) >= n_days:
                result[i] = row[:n_days]
        except Exception:
            pass

    return result


def compute_signals(
    tree: Node,
    stock_data: dict[str, dict[str, np.ndarray]],
    n_days: int,
) -> np.ndarray:
    """
    Compute alpha signals for all stocks.
    Returns (n_stocks, n_days) signal matrix.

    Uses matrix-level evaluation so cross-sectional operators
    (cs_rank, cs_sign, etc.) work correctly across stocks.
    """
    tickers = list(stock_data.keys())
    result = _eval_node_matrix(tree, stock_data, tickers, n_days)
    if result is None:
        return np.full((len(tickers), n_days), np.nan)
    return result


def evaluate_tree(
    tree: Node,
    stock_data: dict[str, dict[str, np.ndarray]],
    close_prices: np.ndarray,
    fwd_returns_1d: np.ndarray,
    fwd_returns_20d: np.ndarray,
    n_days: int,
) -> AlphaMetrics:
    """Full evaluation of a GP tree with proper methodology."""
    signals = compute_signals(tree, stock_data, n_days)
    return evaluate_signals(
        signals, close_prices, fwd_returns_1d, fwd_returns_20d,
        expression=str(tree),
    )
