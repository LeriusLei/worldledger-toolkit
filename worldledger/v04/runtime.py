"""Bounded interpreter for the draft v0.4 source language.

No Python exec/eval/compile, imports, attributes, loops, I/O, or mutation of
state containers. The only addition over v0.3.1 is bounded local assignment.
"""

from __future__ import annotations

import ast
import time

from worldledger.sandbox.executor import _bounded, CandidateExecutionError, CandidateTimeout
from worldledger.sandbox.policy import (
    DEFAULT_TIMEOUT_SECONDS, MAX_AST_DEPTH, MAX_AST_NODES, MAX_SOURCE_BYTES,
    MAX_STEPS, SIGNATURES, SourceRejected,
)


PURE_BUILTINS = {"abs": abs, "int": int, "len": len, "min": min, "max": max}
EXPRESSION_NODES = (
    ast.Constant, ast.Name, ast.Load, ast.Tuple, ast.List, ast.Dict,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.UnaryOp, ast.UAdd, ast.USub,
    ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.IfExp, ast.Subscript,
    ast.Call,
)
STATEMENT_NODES = (ast.Return, ast.If, ast.Assign)
ALLOWED_NODES = EXPRESSION_NODES + STATEMENT_NODES + (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Store,
)
MAX_LOCALS = 32
MAX_UNPACK = 8


def _target_names(target: ast.AST) -> tuple[str, ...]:
    if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)) and 1 <= len(target.elts) <= MAX_UNPACK:
        return tuple(name for child in target.elts for name in _target_names(child))
    raise SourceRejected("assignment target must be bounded local names")


def validate_source_v04(source: str) -> dict[str, ast.FunctionDef]:
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise SourceRejected("source too large or not text")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SourceRejected(f"syntax: {exc.msg}") from exc
    if len(list(ast.walk(tree))) > MAX_AST_NODES:
        raise SourceRejected("AST node limit exceeded")
    functions: dict[str, ast.FunctionDef] = {}
    for node in tree.body:
        if (not isinstance(node, ast.FunctionDef) or node.name not in SIGNATURES
                or node.name in functions):
            raise SourceRejected("exactly the three required functions are allowed")
        args = node.args
        if (tuple(arg.arg for arg in args.args) != SIGNATURES[node.name]
                or args.posonlyargs or args.kwonlyargs or args.defaults
                or args.kw_defaults or args.vararg or args.kwarg
                or any(arg.annotation is not None for arg in args.args)
                or node.decorator_list
                or node.returns or node.type_comment):
            raise SourceRejected("function signature must match contract")
        functions[node.name] = node
    if set(functions) != set(SIGNATURES):
        raise SourceRejected("missing candidate function")
    for node in functions.values():
        parameters = set(SIGNATURES[node.name])
        locals_ = set(parameters)
        for item in ast.walk(node):
            if isinstance(item, ast.Assign):
                if len(item.targets) != 1:
                    raise SourceRejected("only one assignment target allowed")
                names = _target_names(item.targets[0])
                if len(set(names)) != len(names) or any(
                    name in parameters or name in PURE_BUILTINS or name.startswith("__")
                    for name in names
                ):
                    raise SourceRejected("local assignment name invalid or repeated")
                locals_.update(names)
        if len(locals_) > MAX_LOCALS:
            raise SourceRejected("too many local variables")

        def check(item: ast.AST, depth: int) -> None:
            if depth > MAX_AST_DEPTH or not isinstance(item, ALLOWED_NODES):
                raise SourceRejected("forbidden syntax or AST depth")
            if isinstance(item, ast.FunctionDef):
                if not item.body or not all(isinstance(x, STATEMENT_NODES) for x in item.body):
                    raise SourceRejected("only return/if/local assignment statements allowed")
            if isinstance(item, ast.If):
                if not item.body or not all(isinstance(x, STATEMENT_NODES) for x in item.body + item.orelse):
                    raise SourceRejected("if branches contain forbidden statements")
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load):
                if item.id not in locals_ and item.id not in PURE_BUILTINS:
                    raise SourceRejected("unknown name")
            if isinstance(item, ast.Call):
                if (not isinstance(item.func, ast.Name) or item.func.id not in PURE_BUILTINS
                        or item.keywords or len(item.args) > 4):
                    raise SourceRejected("non-whitelisted call")
            if isinstance(item, ast.Constant):
                value = item.value
                if (type(value) not in (int, bool, str, type(None))
                        or isinstance(value, str) and len(value) > 1024
                        or type(value) is int and value.bit_length() > 128):
                    raise SourceRejected("constant outside limits")
            for child in ast.iter_child_nodes(item):
                check(child, depth + 1)

        check(node, 0)
    return functions


def run_candidate_v04(source: str, state: tuple, action: str,
                      timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> tuple:
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    functions = validate_source_v04(source)
    _bounded(state)
    _bounded(action)
    deadline = time.perf_counter() + timeout_seconds
    steps = 0

    def tick() -> None:
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS or time.perf_counter() >= deadline:
            raise CandidateTimeout("candidate execution limit")

    def expression(node: ast.AST, env: dict):
        tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [expression(child, env) for child in node.elts]
            return _bounded(tuple(values) if isinstance(node, ast.Tuple) else values)
        if isinstance(node, ast.Dict):
            return _bounded({expression(k, env): expression(v, env)
                             for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.Subscript):
            return _bounded(expression(node.value, env)[expression(node.slice, env)])
        if isinstance(node, ast.UnaryOp):
            value = expression(node.operand, env)
            if type(value) is not int:
                raise CandidateExecutionError("unary operator requires integer")
            return _bounded(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BinOp):
            left, right = expression(node.left, env), expression(node.right, env)
            if type(left) is not int or type(right) is not int:
                raise CandidateExecutionError("numeric operators require integers")
            return _bounded(left + right if isinstance(node.op, ast.Add)
                            else left - right if isinstance(node.op, ast.Sub)
                            else left * right)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                for value in node.values:
                    if not expression(value, env):
                        return False
                return True
            for value in node.values:
                if expression(value, env):
                    return True
            return False
        if isinstance(node, ast.Compare):
            left = expression(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = expression(comparator, env)
                if isinstance(op, ast.Eq):
                    okay = left == right
                elif isinstance(op, ast.NotEq):
                    okay = left != right
                elif isinstance(op, ast.Lt):
                    okay = left < right
                elif isinstance(op, ast.LtE):
                    okay = left <= right
                elif isinstance(op, ast.Gt):
                    okay = left > right
                elif isinstance(op, ast.GtE):
                    okay = left >= right
                elif isinstance(op, ast.In):
                    okay = left in right
                else:
                    okay = left not in right
                if not okay:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return expression(node.body if expression(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Call):
            args = [expression(arg, env) for arg in node.args]
            return _bounded(PURE_BUILTINS[node.func.id](*args))
        raise CandidateExecutionError("unhandled expression")

    def assign(target: ast.AST, value, env: dict) -> None:
        tick()
        if isinstance(target, ast.Name):
            env[target.id] = _bounded(value)
            return
        if not isinstance(value, (tuple, list)) or len(value) != len(target.elts):
            raise CandidateExecutionError("unpack shape mismatch")
        for child, item in zip(target.elts, value):
            assign(child, item, env)

    missing = object()

    def statements(items: list[ast.stmt], env: dict):
        for item in items:
            tick()
            if isinstance(item, ast.Return):
                return expression(item.value, env)
            if isinstance(item, ast.Assign):
                assign(item.targets[0], expression(item.value, env), env)
            elif isinstance(item, ast.If):
                result = statements(item.body if expression(item.test, env) else item.orelse, env)
                if result is not missing:
                    return result
        return missing

    def invoke(name: str, *args):
        node = functions[name]
        env = dict(zip(SIGNATURES[name], args))
        value = statements(node.body, env)
        if value is missing:
            raise CandidateExecutionError("missing return")
        return _bounded(value)

    try:
        next_state = invoke("transition", state, action)
        if (type(next_state) is not tuple or len(next_state) != len(state)
                or not all(type(x) is int for x in next_state)):
            raise CandidateExecutionError("transition must return integer state tuple")
        observation = invoke("observation", state, action, next_state)
        reward = invoke("reward", state, action, next_state)
        if type(observation) is not tuple or type(reward) is not int:
            raise CandidateExecutionError("observation/reward contract violated")
        return next_state, reward, observation
    except CandidateTimeout:
        raise
    except CandidateExecutionError:
        raise
    except Exception as exc:
        raise CandidateExecutionError(f"candidate runtime failure: {type(exc).__name__}") from exc
