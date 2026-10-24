# Copyright 2026 Rodrigo P. Coelho, sympy-mcp-V2 (https://github.com/rpcoelho17/sympy-mcp-V2).
# Apache-2.0; see LICENSE and NOTICE.
import pytest
from server import (
    intro,
    introduce_expression,
    evaluate_numerically,
    solve_numerically,
    local_vars,
    expressions,
    functions,
)


# Add a fixture to reset global state between tests
@pytest.fixture(autouse=True)
def reset_globals():
    local_vars.clear()
    expressions.clear()
    functions.clear()
    import server

    server.expression_counter = 0
    yield


class TestEvaluateNumericallyTool:
    def test_evaluate_constant(self):
        expr_key = introduce_expression("sqrt(2)*pi")
        result = evaluate_numerically(expr_key)
        assert result.startswith("4.44288293815837")

    def test_evaluate_precision(self):
        expr_key = introduce_expression("pi")
        result = evaluate_numerically(expr_key, precision=30)
        assert result == "3.14159265358979323846264338328"

    def test_evaluate_with_subs(self):
        intro("x", [], [])
        intro("y", [], [])
        expr_key = introduce_expression("x**2 + y")
        result = evaluate_numerically(expr_key, subs={"x": 3, "y": 0.5})
        assert float(result) == pytest.approx(9.5)

    def test_evaluate_unknown_variable(self):
        expr_key = introduce_expression("2")
        result = evaluate_numerically(expr_key, subs={"z": 1})
        assert "error" in result.lower()

    def test_evaluate_missing_expression(self):
        result = evaluate_numerically("nonexistent_key")
        assert "error" in result.lower()


class TestSolveNumericallyTool:
    def test_solve_transcendental(self):
        # cos(x) = x has no closed form
        intro("x", [], [])
        expr_key = introduce_expression("Eq(cos(x), x)")
        result = solve_numerically([expr_key], ["x"], [1])
        assert result.startswith("{x: 0.73908513321516")

    def test_solve_picks_root_near_guess(self):
        intro("x", [], [])
        expr_key = introduce_expression("x**2 - 2")
        assert solve_numerically([expr_key], ["x"], [1]).startswith("{x: 1.41421356")
        assert solve_numerically([expr_key], ["x"], [-1]).startswith("{x: -1.41421356")

    def test_solve_system(self):
        intro("x", [], [])
        intro("y", [], [])
        eq1 = introduce_expression("Eq(x**2 + y**2, 4)")
        eq2 = introduce_expression("Eq(exp(x), y)")
        result = solve_numerically([eq1, eq2], ["x", "y"], [0.5, 1.5])
        values = dict(part.split(": ") for part in result.strip("{}").split(", "))
        x, y = float(values["x"]), float(values["y"])
        assert x**2 + y**2 == pytest.approx(4)
        assert y == pytest.approx(2.718281828459045**x)

    def test_solve_guess_count_mismatch(self):
        intro("x", [], [])
        expr_key = introduce_expression("x - 1")
        result = solve_numerically([expr_key], ["x"], [1, 2])
        assert "error" in result.lower()

    def test_solve_no_convergence(self):
        # x**2 + 1 has no real root, so Newton's method from a real guess fails
        intro("x", [], [])
        expr_key = introduce_expression("x**2 + 1")
        result = solve_numerically([expr_key], ["x"], [1])
        assert "error" in result.lower()
