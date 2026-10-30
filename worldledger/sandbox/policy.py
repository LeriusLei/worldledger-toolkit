"""Reject every source feature outside the tiny candidate expression language."""

import ast


MAX_SOURCE_BYTES = 65536
MAX_AST_NODES = 2000
MAX_AST_DEPTH = 40
MAX_VALUE_BYTES = 65536
MAX_STEPS = 20000
DEFAULT_TIMEOUT_SECONDS = 0.5
SIGNATURES = {
    "transition": ("state", "action"),
    "observation": ("state", "action", "next_state"),
    "reward": ("state", "action", "next_state"),
}
BUILTINS = {"abs", "int", "len", "max", "min"}
ALLOWED = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If,
    ast.Constant, ast.Name, ast.Load, ast.Tuple, ast.List, ast.Dict,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.UnaryOp, ast.USub, ast.UAdd,
    ast.BoolOp, ast.And, ast.Or, ast.Compare, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.IfExp, ast.Subscript, ast.Call,
)


class SourceRejected(ValueError):
    pass


def validate_source(source: str) -> dict[str, ast.FunctionDef]:
    if not isinstance(source, str) or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise SourceRejected("source too large or not text")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SourceRejected(f"syntax: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise SourceRejected("AST node limit exceeded")
    functions = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in SIGNATURES or node.name in functions:
            raise SourceRejected("exactly the three required functions are allowed")
        if (tuple(arg.arg for arg in node.args.args) != SIGNATURES[node.name]
                or node.args.posonlyargs or node.args.kwonlyargs or node.args.defaults
                or node.args.vararg or node.args.kwarg or node.decorator_list
                or node.returns or node.type_comment):
            raise SourceRejected("function signature must match contract")
        functions[node.name] = node
    if set(functions) != set(SIGNATURES):
        raise SourceRejected("missing candidate function")

    def check(node: ast.AST, depth: int) -> None:
        if depth > MAX_AST_DEPTH or not isinstance(node, ALLOWED):
            raise SourceRejected("forbidden syntax or AST depth")
        if isinstance(node, ast.FunctionDef):
            if not node.body or not all(isinstance(item, (ast.Return, ast.If)) for item in node.body):
                raise SourceRejected("only return/if statements allowed")
        if isinstance(node, ast.If):
            if not node.body or not node.orelse:
                raise SourceRejected("if requires both branches")
            if not all(isinstance(item, (ast.Return, ast.If)) for item in node.body + node.orelse):
                raise SourceRejected("if branches require return/if")
        if isinstance(node, ast.Name) and node.id not in {"state", "action", "next_state"} | BUILTINS:
            raise SourceRejected("unknown name")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in BUILTINS or node.keywords:
                raise SourceRejected("non-whitelisted call")
        if isinstance(node, ast.Constant) and (type(node.value) not in (int, bool, str, type(None))
                                                or isinstance(node.value, str) and len(node.value) > 1024
                                                or type(node.value) is int and node.value.bit_length() > 128):
            raise SourceRejected("constant outside limits")
        for child in ast.iter_child_nodes(node):
            check(child, depth + 1)

    check(tree, 0)
    return functions
