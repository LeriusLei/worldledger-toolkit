"""Interpret whitelisted syntax without exec/eval/import/file/network access."""

import ast
import sys
import time

from worldledger.sandbox.policy import (
    DEFAULT_TIMEOUT_SECONDS, MAX_STEPS, MAX_VALUE_BYTES, validate_source,
)


class CandidateTimeout(TimeoutError):
    pass


class CandidateExecutionError(ValueError):
    pass


def _size(value, seen=None):
    seen = set() if seen is None else seen
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    size = sys.getsizeof(value)
    if isinstance(value, (tuple, list)):
        size += sum(_size(item, seen) for item in value)
    elif isinstance(value, dict):
        size += sum(_size(key, seen) + _size(item, seen) for key, item in value.items())
    return size


def _bounded(value):
    if type(value) not in (int, bool, str, tuple, list, dict, type(None)):
        raise CandidateExecutionError("unsupported value type")
    if type(value) is int and value.bit_length() > 128:
        raise CandidateExecutionError("integer magnitude limit")
    if _size(value) > MAX_VALUE_BYTES:
        raise CandidateExecutionError("logical memory limit")
    return value


def run_candidate(source: str, state: tuple, action: str,
                  timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> tuple:
    """Return (next_state, reward, observation) under bounded interpretation."""
    if timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    functions = validate_source(source)
    _bounded(state)
    _bounded(action)
    deadline = time.perf_counter() + timeout_seconds
    steps = 0

    def tick():
        nonlocal steps
        steps += 1
        if steps > MAX_STEPS or time.perf_counter() >= deadline:
            raise CandidateTimeout("candidate execution limit")

    def expression(node, env):
        tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, (ast.Tuple, ast.List)):
            values = [expression(item, env) for item in node.elts]
            return _bounded(tuple(values) if isinstance(node, ast.Tuple) else values)
        if isinstance(node, ast.Dict):
            return _bounded({expression(k, env): expression(v, env) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.Subscript):
            value, index = expression(node.value, env), expression(node.slice, env)
            return _bounded(value[index])
        if isinstance(node, ast.UnaryOp):
            value = expression(node.operand, env)
            return _bounded(-value if isinstance(node.op, ast.USub) else +value)
        if isinstance(node, ast.BinOp):
            left, right = expression(node.left, env), expression(node.right, env)
            if type(left) is not int or type(right) is not int:
                raise CandidateExecutionError("numeric operators require integers")
            result = left + right if isinstance(node.op, ast.Add) else left - right if isinstance(node.op, ast.Sub) else left * right
            return _bounded(result)
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(expression(item, env) for item in node.values)
            return any(expression(item, env) for item in node.values)
        if isinstance(node, ast.Compare):
            left = expression(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = expression(comparator, env)
                okay = ((left == right) if isinstance(op, ast.Eq) else
                        (left != right) if isinstance(op, ast.NotEq) else
                        (left < right) if isinstance(op, ast.Lt) else
                        (left <= right) if isinstance(op, ast.LtE) else
                        (left > right) if isinstance(op, ast.Gt) else left >= right)
                if not okay:
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return expression(node.body if expression(node.test, env) else node.orelse, env)
        if isinstance(node, ast.Call):
            builtins = {"abs": abs, "int": int, "len": len, "max": max, "min": min}
            args = [expression(arg, env) for arg in node.args]
            if len(args) > 4:
                raise CandidateExecutionError("too many call arguments")
            return _bounded(builtins[node.func.id](*args))
        raise CandidateExecutionError("unhandled expression")

    def statements(items, env):
        for item in items:
            tick()
            if isinstance(item, ast.Return):
                return expression(item.value, env)
            if isinstance(item, ast.If):
                return statements(item.body if expression(item.test, env) else item.orelse, env)
        raise CandidateExecutionError("missing return")

    def invoke(name, *args):
        node = functions[name]
        env = dict(zip((arg.arg for arg in node.args.args), args))
        return _bounded(statements(node.body, env))

    try:
        next_state = invoke("transition", state, action)
        if type(next_state) is not tuple or len(next_state) != len(state) or not all(type(x) is int for x in next_state):
            raise CandidateExecutionError("transition must return integer state tuple")
        observation = invoke("observation", state, action, next_state)
        reward = invoke("reward", state, action, next_state)
        if type(observation) is not tuple or type(reward) is not int:
            raise CandidateExecutionError("observation/reward contract violated")
        return next_state, reward, observation
    except CandidateTimeout:
        raise
    except Exception as exc:
        raise CandidateExecutionError(f"candidate runtime failure: {type(exc).__name__}") from exc
