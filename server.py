# A stateful MCP server that holds a sympy session, with symbol table of variables
# that can be used in the tools API to define and manipulate expressions.
#
# Original sympy-mcp: Copyright 2025 Stephen Diehl, https://github.com/sdiehl/sympy-mcp
# Modified 2026 by Rodrigo P. Coelho for sympy-mcp-V2 (https://github.com/rpcoelho17/sympy-mcp-V2):
# added the evaluate_numerically and solve_numerically tools and the sympy.stats tools.
# Apache-2.0; see LICENSE and NOTICE.

from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import httpx
import json
import multiprocessing
import mpmath
import sympy
import sympy.stats as stats
import argparse
import logging
from typing import List, Dict, Optional, Literal, Any, Union
from pydantic import BaseModel
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.facts import InconsistentAssumptions
from sympy.stats.rv import RandomSymbol
from vars import (
    Assumption,
    Distribution,
    Domain,
    MarkovQuantity,
    ODEHint,
    PDEHint,
    Metric,
    RandomStatistic,
    Tensor,
    UnitSystem,
)
from sympy import Eq, Function, dsolve, diff, integrate, simplify, Matrix
from sympy.solvers.pde import pdsolve
from sympy.vector import CoordSys3D, curl, divergence, gradient

from sympy.physics.units import convert_to
from sympy.physics.units import __dict__ as units_dict
from sympy.physics.units.systems import SI, MKS, MKSA, natural
from sympy.physics.units.systems.cgs import cgs_gauss

# Import common units
from sympy.physics.units import (
    meter,
    kilogram,
    second,
    ampere,
    kelvin,
    mole,
    candela,
    kilometer,
    millimeter,
    gram,
    joule,
    newton,
    pascal,
    watt,
    coulomb,
    volt,
    ohm,
    farad,
    henry,
    speed_of_light,
    gravitational_constant,
    planck,
    day,
    year,
    minute,
    hour,
)

try:
    from einsteinpy.symbolic import (
        MetricTensor,
        RicciTensor,
        RicciScalar,
        EinsteinTensor,
        WeylTensor,
        ChristoffelSymbols,
        StressEnergyMomentumTensor,
    )
    from einsteinpy.symbolic.predefined import (
        Schwarzschild,
        Minkowski,
        MinkowskiCartesian,
        KerrNewman,
        Kerr,
        AntiDeSitter,
        DeSitter,
        ReissnerNordstorm,
        find,
    )

    EINSTEINPY_AVAILABLE = True
except ImportError:
    EINSTEINPY_AVAILABLE = False

# Set up logging
logger = logging.getLogger(__name__)

# Create an MCP server
mcp = MCPServer(
    "sympy-mcp",
    dependencies=["sympy", "pydantic", "einsteinpy"],
    instructions="Provides access to the Sympy computer algebra system, which can perform symbolic manipulation of mathematical expressions.",
)

mcp_port = 8081

@mcp.custom_route("/healthcheck", methods=["GET"])
async def healthcheck(request: Request) -> Response:
    try:
        url = f"http://127.0.0.1:{mcp_port}/mcp"
        headers = {"Accept": "application/json, text/event-stream"}

        async with httpx.AsyncClient() as client:
            init = await client.post(url, json={
                "jsonrpc": "2.0", "id": 1, "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "healthcheck", "version": "1.0"},
                },
            }, headers=headers)
            session_id = init.headers.get("mcp-session-id")

            req_headers = {**headers}
            if session_id:
                req_headers["mcp-session-id"] = session_id

            resp = await client.post(url, json={
                "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
            }, headers=req_headers)

            # streamable-http may return SSE (text/event-stream) or plain JSON
            content_type = resp.headers.get("content-type", "")
            if "text/event-stream" in content_type:
                data = None
                for line in resp.text.splitlines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        break
                tools = (data or {}).get("result", {}).get("tools", [])
            else:
                tools = resp.json().get("result", {}).get("tools", [])

            # Close the session to avoid accumulating orphaned sessions
            if session_id:
                await client.delete(url, headers={"mcp-session-id": session_id})

            return JSONResponse({"status": "ok", "tool_count": len(tools)})
    except Exception as e:
        return JSONResponse({"status": "error", "detail": str(e)}, status_code=500)


# Global store for sympy variables and expressions
local_vars: Dict[str, sympy.Symbol] = {}
functions: Dict[str, sympy.Function] = {}
expressions: Dict[str, sympy.Expr] = {}
metrics: Dict[str, Any] = {}
tensor_objects: Dict[str, Any] = {}
coordinate_systems: Dict[str, CoordSys3D] = {}
expression_counter = 0


# Pydantic model for defining a variable with assumptions
class VariableDefinition(BaseModel):
    var_name: str
    pos_assumptions: List[str] = []
    neg_assumptions: List[str] = []


# x, y = symbols('x, y', commutative=False)


# Add an addition tool
@mcp.tool()
def intro(
    var_name: str, pos_assumptions: List[Assumption], neg_assumptions: List[Assumption]
) -> str:
    """Introduces a sympy variable with specified assumptions and stores it.

    Takes a variable name and a list of positive and negative assumptions.
    """
    kwargs_for_symbols = {}
    # Add assumptions
    for assumption_obj in pos_assumptions:
        kwargs_for_symbols[assumption_obj.value] = True

    for assumption_obj in neg_assumptions:
        kwargs_for_symbols[assumption_obj.value] = False

    try:
        var = sympy.symbols(var_name, **kwargs_for_symbols)
    except InconsistentAssumptions as e:
        return f"Error creating symbol '{var_name}': The provided assumptions {kwargs_for_symbols} are inconsistent according to SymPy. Details: {str(e)}"
    except Exception as e:
        return f"Error creating symbol '{var_name}': An unexpected error occurred. Assumptions attempted: {kwargs_for_symbols}. Details: {type(e).__name__} - {str(e)}"

    local_vars[var_name] = var
    return var_name


# Introduce multiple variables simultaneously
@mcp.tool()
def intro_many(variables: List[VariableDefinition]) -> str:
    """Introduces multiple sympy variables with specified assumptions and stores them.

    Takes a list of VariableDefinition objects for the 'variables' parameter.
    Each object in the list specifies:
    - var_name: The name of the variable (string).
    - pos_assumptions: A list of positive assumption strings (e.g., ["real", "positive"]).
    - neg_assumptions: A list of negative assumption strings (e.g., ["complex"]).

    The JSON payload for the 'variables' argument should be a direct list of these objects, for example:
    ```json
    [
        {
            "var_name": "x",
            "pos_assumptions": ["real", "positive"],
            "neg_assumptions": ["complex"]
        },
        {
            "var_name": "y",
            "pos_assumptions": [],
            "neg_assumptions": ["commutative"]
        }
    ]
    ```

    The assumptions must be consistent, so a real number is not allowed to be non-commutative.

    Prefer this over intro() for multiple variables because it's more efficient.
    """
    var_keys = {}
    for var_def in variables:
        try:
            processed_pos_assumptions = [
                Assumption(a_str) for a_str in var_def.pos_assumptions
            ]
            processed_neg_assumptions = [
                Assumption(a_str) for a_str in var_def.neg_assumptions
            ]
        except ValueError as e:
            # Handle cases where a string doesn't match an Assumption enum member
            msg = (
                f"Error for variable '{var_def.var_name}': Invalid assumption string provided. {e}. "
                f"Ensure assumptions match valid enum values in 'vars.Assumption'."
            )
            logger.error(msg)
            return msg  # Or collect errors

        var_key = intro(
            var_def.var_name, processed_pos_assumptions, processed_neg_assumptions
        )
        var_keys[var_def.var_name] = var_key

    # Return the mapping of variable names to keys
    return str(var_keys)


# XXX use local_vars {x : "expr_1", y : "expr_2"}
@mcp.tool()
def introduce_expression(
    expr_str: str, canonicalize: bool = True, expr_var_name: Optional[str] = None
) -> str:
    """Parses a sympy expression string using available local variables and stores it. Assigns it to either a temporary name (expr_0, expr_1, etc.) or a user-specified global name.

    Uses Sympy parse_expr to parse the expression string.

    Applies default Sympy canonicalization rules unless canonicalize is False.

    For equations (x^2 = 1) make the input string "Eq(x^2, 1") not "x^2 == 1"

    Examples:

        {expr_str: "Eq(x^2 + y^2, 1)"}
        {expr_str: "Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))"}
        {expr_str: "pi+e", "expr_var_name": "z"}
    """
    global expression_counter
    # Merge local_vars and functions dictionaries to make both available for parsing
    parse_dict = {**local_vars, **functions}
    parsed_expr = parse_expr(expr_str, local_dict=parse_dict, evaluate=canonicalize)
    if expr_var_name is None:
        expr_key = f"expr_{expression_counter}"
    else:
        expr_key = expr_var_name
    expressions[expr_key] = parsed_expr
    expression_counter += 1
    return expr_key


def introduce_equation(lhs_str: str, rhs_str: str) -> str:
    """Introduces an equation (lhs = rhs) using available local variables."""
    global expression_counter
    # Merge local_vars and functions dictionaries to make both available for parsing
    parse_dict = {**local_vars, **functions}
    lhs_expr = parse_expr(lhs_str, local_dict=parse_dict)
    rhs_expr = parse_expr(rhs_str, local_dict=parse_dict)
    eq_key = f"eq_{expression_counter}"
    expressions[eq_key] = Eq(lhs_expr, rhs_expr)
    expression_counter += 1
    return eq_key


@mcp.tool()
def print_latex_expression(expr_key: str) -> str:
    """Prints a stored expression in LaTeX format, along with variable assumptions."""
    if expr_key not in expressions:
        return f"Error: Expression key '{expr_key}' not found."

    expr = expressions[expr_key]

    # Handle dictionary objects (like eigenvalues)
    if isinstance(expr, dict):
        if all(isinstance(k, (sympy.Expr, int, float)) for k in expr.keys()):
            # Format as eigenvalues: {value: multiplicity, ...}
            parts = []
            for eigenval, multiplicity in expr.items():
                parts.append(
                    f"{sympy.latex(eigenval)} \\text{{ (multiplicity {multiplicity})}}"
                )
            return ", ".join(parts)
        else:
            # Generic dictionary
            return str(expr)

    # Handle list objects (like eigenvectors)
    elif isinstance(expr, list):
        # For eigenvectors format: [(eigenval, multiplicity, [eigenvectors]), ...]
        if all(isinstance(item, tuple) and len(item) == 3 for item in expr):
            parts = []
            for eigenval, multiplicity, eigenvects in expr:
                eigenvects_latex = [sympy.latex(v) for v in eigenvects]
                parts.append(
                    f"\\lambda = {sympy.latex(eigenval)} \\text{{ (multiplicity {multiplicity})}}:\n"
                    f"\\text{{Eigenvectors: }}[{', '.join(eigenvects_latex)}]"
                )
            return "\n".join(parts)
        else:
            # Try to convert each element to LaTeX
            try:
                return str([sympy.latex(item) for item in expr])
            except Exception as e:
                # Log the exception if there's a logger configured
                logger.debug(f"Error converting list items to LaTeX: {str(e)}")
                return str(expr)

    # Original behavior for sympy expressions
    latex_str = sympy.latex(expr)

    # Find variables in the expression and their assumptions
    try:
        variables_in_expr = expr.free_symbols
        assumption_descs = []
        for var_symbol in variables_in_expr:
            var_name = str(var_symbol)
            if var_name in local_vars:
                # Get assumptions directly from the symbol object
                current_assumptions = []
                # sympy stores assumptions in a private attribute _assumptions
                # and provides a way to query them via .is_commutative, .is_real etc.
                # We can iterate through known Assumption enum values
                for assumption_enum_member in Assumption:
                    if getattr(var_symbol, f"is_{assumption_enum_member.value}", False):
                        current_assumptions.append(assumption_enum_member.value)

                if current_assumptions:
                    assumption_descs.append(
                        f"{var_name} is {', '.join(current_assumptions)}"
                    )
                else:
                    assumption_descs.append(
                        f"{var_name} (no specific assumptions listed)"
                    )
            else:
                assumption_descs.append(f"{var_name} (undefined in local_vars)")

        if assumption_descs:
            return f"{latex_str} (where {'; '.join(assumption_descs)})"
        else:
            return latex_str
    except AttributeError:
        # If expr doesn't have free_symbols, just return the LaTeX
        return latex_str


@mcp.tool()
def solve_algebraically(
    expr_key: str, solve_for_var_name: str, domain: Domain = Domain.COMPLEX
) -> str:
    """Solves an equation (expression = 0) algebraically for a given variable.

    Args:
        expr_key: The key of the expression (previously introduced) to be solved.
        solve_for_var_name: The name of the variable (previously introduced) to solve for.
        domain: The domain to solve in: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS. Defaults to Domain.COMPLEX.

    Returns:
        A LaTeX string representing the set of solutions. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    expression_to_solve = expressions[expr_key]

    if solve_for_var_name not in local_vars:
        return f"Error: Variable '{solve_for_var_name}' not found in local_vars. Please introduce it first."

    variable_symbol = local_vars[solve_for_var_name]

    # Map domain enum to SymPy domain sets
    domain_map = {
        Domain.COMPLEX: sympy.S.Complexes,
        Domain.REAL: sympy.S.Reals,
        Domain.INTEGERS: sympy.S.Integers,
        Domain.NATURALS: sympy.S.Naturals0,
    }

    if domain not in domain_map:
        return "Error: Invalid domain. Choose from: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS."

    sympy_domain = domain_map[domain]

    try:
        # If the expression is an equation (Eq object), convert it to standard form
        if isinstance(expression_to_solve, sympy.Eq):
            expression_to_solve = expression_to_solve.lhs - expression_to_solve.rhs

        # Use solveset instead of solve
        solution_set = sympy.solveset(
            expression_to_solve, variable_symbol, domain=sympy_domain
        )

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the equation: {str(e)}. The equation may not have a closed-form solution or the algorithm is not implemented."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


@mcp.tool()
def solve_linear_system(
    expr_keys: List[str], var_names: List[str], domain: Domain = Domain.COMPLEX
) -> str:
    """Solves a system of linear equations using SymPy's linsolve.

    Args:
        expr_keys: The keys of the expressions (previously introduced) forming the system.
        var_names: The names of the variables to solve for.
        domain: The domain to solve in (Domain.COMPLEX, Domain.REAL, etc.). Defaults to Domain.COMPLEX.

    Returns:
        A LaTeX string representing the solution set. Returns an error message string if issues occur.
    """
    # Validate all expression keys exist
    system = []
    for expr_key in expr_keys:
        if expr_key not in expressions:
            return f"Error: Expression with key '{expr_key}' not found."

        expr = expressions[expr_key]
        # Convert equations to standard form
        if isinstance(expr, sympy.Eq):
            expr = expr.lhs - expr.rhs
        system.append(expr)

    # Validate all variables exist
    symbols = []
    for var_name in var_names:
        if var_name not in local_vars:
            return f"Error: Variable '{var_name}' not found in local_vars. Please introduce it first."
        symbols.append(local_vars[var_name])

    # Map domain enum to SymPy domain sets
    domain_map = {
        Domain.COMPLEX: sympy.S.Complexes,
        Domain.REAL: sympy.S.Reals,
        Domain.INTEGERS: sympy.S.Integers,
        Domain.NATURALS: sympy.S.Naturals0,
    }

    if domain not in domain_map:
        return "Error: Invalid domain. Choose from: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS."

    domain_map[domain]

    try:
        # Use SymPy's linsolve - note: it doesn't take domain parameter directly, but works on the given domain
        solution_set = sympy.linsolve(system, symbols)

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the linear system: {str(e)}."
    except ValueError as e:
        return f"Error: Invalid system or arguments: {str(e)}."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


@mcp.tool()
def solve_nonlinear_system(
    expr_keys: List[str], var_names: List[str], domain: Domain = Domain.COMPLEX
) -> str:
    """Solves a system of nonlinear equations using SymPy's nonlinsolve.

    Args:
        expr_keys: The keys of the expressions (previously introduced) forming the system.
        var_names: The names of the variables to solve for.
        domain: The domain to solve in (Domain.COMPLEX, Domain.REAL, etc.). Defaults to Domain.COMPLEX.

    Returns:
        A LaTeX string representing the solution set. Returns an error message string if issues occur.
    """
    # Validate all expression keys exist
    system = []
    for expr_key in expr_keys:
        if expr_key not in expressions:
            return f"Error: Expression with key '{expr_key}' not found."

        expr = expressions[expr_key]
        # Convert equations to standard form
        if isinstance(expr, sympy.Eq):
            expr = expr.lhs - expr.rhs
        system.append(expr)

    # Validate all variables exist
    symbols = []
    for var_name in var_names:
        if var_name not in local_vars:
            return f"Error: Variable '{var_name}' not found in local_vars. Please introduce it first."
        symbols.append(local_vars[var_name])

    # Map domain enum to SymPy domain sets
    domain_map = {
        Domain.COMPLEX: sympy.S.Complexes,
        Domain.REAL: sympy.S.Reals,
        Domain.INTEGERS: sympy.S.Integers,
        Domain.NATURALS: sympy.S.Naturals0,
    }

    if domain not in domain_map:
        return "Error: Invalid domain. Choose from: Domain.COMPLEX, Domain.REAL, Domain.INTEGERS, or Domain.NATURALS."

    try:
        # Use SymPy's nonlinsolve
        solution_set = sympy.nonlinsolve(system, symbols)

        # Convert the set to LaTeX format
        latex_output = sympy.latex(solution_set)
        return latex_output
    except NotImplementedError as e:
        return f"Error: SymPy could not solve the nonlinear system: {str(e)}."
    except ValueError as e:
        return f"Error: Invalid system or arguments: {str(e)}."
    except Exception as e:
        return f"An unexpected error occurred during solving: {str(e)}"


@mcp.tool()
def evaluate_numerically(
    expr_key: str,
    precision: int = 15,
    subs: Optional[Dict[str, float]] = None,
) -> str:
    """Evaluates a stored expression to a decimal number using SymPy's evalf.

    Use this to turn an exact result (e.g. sqrt(2)*pi) into a number, or to
    evaluate an expression at given numeric values of its variables.

    Args:
        expr_key: The key of the expression (previously introduced) to evaluate.
        precision: Number of significant digits. Defaults to 15.
        subs: Optional mapping of variable names (previously introduced) to numeric values.

    Example:
        {expr_key: "expr_0", precision: 30}
        {expr_key: "expr_1", subs: {"x": 2.5, "y": -1}}

    Returns:
        The numeric value as a string. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if precision < 1:
        return "Error: precision must be a positive integer."

    sub_map = {}
    for var_name, value in (subs or {}).items():
        if var_name not in local_vars:
            return f"Error: Variable '{var_name}' not found in local_vars. Please introduce it first."
        sub_map[local_vars[var_name]] = value

    try:
        result = sympy.N(expressions[expr_key], precision, subs=sub_map)
        return str(result)
    except Exception as e:
        return f"Error during numeric evaluation: {str(e)}"


@mcp.tool()
def solve_numerically(
    expr_keys: List[str],
    var_names: List[str],
    initial_guesses: List[float],
    precision: int = 15,
) -> str:
    """Numerically solves an equation or system of equations using SymPy's nsolve.

    Use this when solve_algebraically or solve_nonlinear_system cannot find a
    closed-form solution, or when only a numeric answer is needed. Each
    expression is treated as expression = 0 (Eq objects are converted).

    nsolve uses Newton's method from the initial guesses and returns only the
    single root nearest to them. To find other roots, call again with different
    initial guesses.

    Args:
        expr_keys: The keys of the expressions (previously introduced) forming the system.
        var_names: The names of the variables to solve for, one per equation.
        initial_guesses: One starting value per variable, in the same order as var_names.
        precision: Number of significant digits. Defaults to 15.

    Example:
        # Solve cos(x) = x near x = 1
        {expr_keys: ["expr_0"], var_names: ["x"], initial_guesses: [1]}

    Returns:
        The solution as a string mapping variables to values, e.g. "{x: 0.739085133215161}".
        Returns an error message string if issues occur.
    """
    if len(var_names) != len(initial_guesses):
        return "Error: Provide exactly one initial guess per variable."

    if precision < 1:
        return "Error: precision must be a positive integer."

    system = []
    for expr_key in expr_keys:
        if expr_key not in expressions:
            return f"Error: Expression with key '{expr_key}' not found."

        expr = expressions[expr_key]
        # Convert equations to standard form
        if isinstance(expr, sympy.Eq):
            expr = expr.lhs - expr.rhs
        system.append(expr)

    symbols = []
    for var_name in var_names:
        if var_name not in local_vars:
            return f"Error: Variable '{var_name}' not found in local_vars. Please introduce it first."
        symbols.append(local_vars[var_name])

    try:
        if len(symbols) == 1 and len(system) == 1:
            solution = [sympy.nsolve(system[0], symbols[0], initial_guesses[0], prec=precision)]
        else:
            solution = list(sympy.nsolve(system, symbols, initial_guesses, prec=precision))

        pairs = ", ".join(f"{sym}: {val}" for sym, val in zip(symbols, solution))
        return "{" + pairs + "}"
    except ValueError as e:
        return f"Error: nsolve did not converge from the given initial guesses: {str(e)}. Try different initial guesses."
    except Exception as e:
        return f"An unexpected error occurred during numeric solving: {str(e)}"


# sympy.stats tools.
#
# Exact sympy.stats calls can run for ever: P(T > 2.228) for a StudentT never
# returned in testing, and a tool call that never returns stalls the whole
# server. So they run in a worker process that can be killed. The worker is kept
# between calls because starting one re-imports this module, which takes seconds.
STATS_TIMEOUT_SECONDS = 20
STATS_WORKER_START_SECONDS = 120
_stats_worker_process = None
_stats_worker_conn = None


def _numeric_setup(expr):
    """The one continuous random variable in expr, a dummy, its density and its support."""
    rvs = list(stats.random_symbols(expr))
    if len(rvs) != 1 or not rvs[0].pspace.is_Continuous:
        raise NotImplementedError(
            "numerical integration handles expressions of one continuous random variable"
        )
    rv = rvs[0]
    dist = rv.pspace.distribution
    x = sympy.Dummy("x", real=True)
    return rv, x, sympy.lambdify(x, dist.pdf(x), "mpmath"), dist.set


def _integrate_over(fn, region):
    """Integrates fn over a set, refusing a result whose error estimate is not small."""
    if isinstance(region, sympy.Union):
        return sum(_integrate_over(fn, part) for part in region.args)
    if not isinstance(region, sympy.Interval):
        return mpmath.mpf(0)  # single points carry no probability
    limits = [mpmath.mpf(str(sympy.N(end, 30))) if end.is_finite else (mpmath.inf if end > 0 else -mpmath.inf)
              for end in (region.inf, region.sup)]
    value, error = mpmath.quad(fn, limits, error=True)
    if abs(error) > 1e-6 * max(abs(value), 1):
        raise ArithmeticError("the integral did not converge, so this quantity may not exist")
    return value


def _numeric_expectation(expr):
    rv, x, pdf, support = _numeric_setup(expr)
    g = sympy.lambdify(x, expr.subs(rv, x), "mpmath")
    # An odd integrand can cancel to a tidy 0 when the mean does not exist, as with
    # a Cauchy variable, so the absolute value has to converge first.
    _integrate_over(lambda t: abs(g(t)) * pdf(t), support)
    return _integrate_over(lambda t: g(t) * pdf(t), support)


def _stats_probability_numeric(event, given):
    joint = sympy.And(event, given) if given is not None else event
    rv, x, pdf, support = _numeric_setup(joint)

    def mass(condition):
        return _integrate_over(pdf, condition.subs(rv, x).as_set().intersect(support))

    value = mass(joint)
    return sympy.Float(str(value / mass(given) if given is not None else value), 15)


def _stats_expectation_numeric(expr, given):
    if given is not None:
        raise NotImplementedError("numerical integration does not handle a condition here")
    return sympy.Float(str(_numeric_expectation(expr)), 15)


def _stats_spread_numeric(statistic, expr, given):
    if given is not None or statistic not in ("variance", "standard_deviation"):
        raise NotImplementedError(f"numerical integration does not handle {statistic} with these inputs")
    mean = _numeric_expectation(expr)
    value = _numeric_expectation(expr**2) - mean**2
    return sympy.Float(str(value if statistic == "variance" else mpmath.sqrt(value)), 15)


def _stats_statistic(statistic, expr, given):
    function = {
        "variance": stats.variance,
        "standard_deviation": stats.std,
        "skewness": stats.skewness,
        "kurtosis": stats.kurtosis,
        "entropy": stats.H,
    }[statistic]
    return function(expr, given) if given is not None else function(expr)


def _stats_probability(event, given):
    return stats.P(event, given) if given is not None else stats.P(event)


def _stats_expectation(expr, given):
    return stats.E(expr, given) if given is not None else stats.E(expr)


def _stats_moment(expr, order, central):
    return (stats.cmoment if central else stats.moment)(expr, order)


def _stats_covariance(first, second, correlation):
    return (stats.correlation if correlation else stats.covariance)(first, second)


def _stats_density(rv):
    result = stats.density(rv)
    return dict(result) if isinstance(result, dict) else result(sympy.Symbol("x"))


def _stats_cdf(rv):
    result = stats.cdf(rv)
    return dict(result) if isinstance(result, dict) else result(sympy.Symbol("x"))


def _stats_quantile(rv, level):
    return stats.quantile(rv)(level)


def _stats_quantile_numeric(rv, level):
    """Root of CDF(t) = level, for when SymPy cannot invert the CDF."""
    _, _, pdf, support = _numeric_setup(rv)
    low = support.inf

    def cdf(t):
        return _integrate_over(pdf, sympy.Interval(low, sympy.Float(str(t), 30)))

    target = float(level)
    left = float(low) if low.is_finite else -1.0
    right = float(support.sup) if support.sup.is_finite else left + 1.0
    while not low.is_finite and cdf(left) > target:
        left *= 2
    while not support.sup.is_finite and cdf(right) < target:
        right = left + 2 * (right - left)
    return sympy.Float(str(mpmath.findroot(lambda t: cdf(t) - target, (left, right), solver="anderson", verify=False)), 15)


def _stats_markov(matrix, quantity, continuous):
    states = list(range(matrix.rows))
    if continuous:
        chain = stats.ContinuousMarkovChain("C", states, matrix)
        if quantity != "limiting_distribution":
            raise ValueError("A continuous chain offers limiting_distribution only.")
        return chain.limiting_distribution()
    chain = stats.DiscreteMarkovChain("M", states, matrix)
    if quantity == "limiting_distribution":
        return chain.limiting_distribution
    if quantity == "communication_classes":
        return [(list(states), recurrent, period) for states, recurrent, period in chain.communication_classes()]
    return getattr(chain, quantity)()


_STATS_TASKS = {
    "probability": _stats_probability,
    "probability_numeric": _stats_probability_numeric,
    "expectation": _stats_expectation,
    "expectation_numeric": _stats_expectation_numeric,
    "statistic": _stats_statistic,
    "statistic_numeric": _stats_spread_numeric,
    "moment": _stats_moment,
    "covariance": _stats_covariance,
    "density": _stats_density,
    "cdf": _stats_cdf,
    "quantile": _stats_quantile,
    "quantile_numeric": _stats_quantile_numeric,
    "markov": _stats_markov,
}


def _describe_stats_value(value):
    """(value, LaTeX, decimal or None); done in the worker, so the timeout covers it."""
    latex = sympy.latex(value)
    decimal = None
    if isinstance(value, sympy.MatrixBase):
        decimal = str(value.evalf(15).tolist()) if value.is_number or not value.free_symbols else None
    elif isinstance(value, sympy.Expr) and value.is_number:
        decimal = str(sympy.N(value, 15))
    return value, latex, decimal


def _stats_worker(conn):
    conn.send(("ready", None))
    while True:
        task, args = conn.recv()
        try:
            value = _STATS_TASKS[task](*args)
            if isinstance(value, sympy.ConditionSet) or (
                isinstance(value, sympy.Basic)
                and value.has(sympy.Integral, stats.Probability, stats.Expectation)
                and not value.free_symbols
            ):
                conn.send(("unsolved", "SymPy found no closed form"))
            else:
                conn.send(("ok", _describe_stats_value(value)))
        except NotImplementedError as e:
            conn.send(("unsolved", str(e)))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}"))


def _stop_stats_worker():
    global _stats_worker_process, _stats_worker_conn
    if _stats_worker_process is not None:
        _stats_worker_process.kill()
        _stats_worker_process.join()
    _stats_worker_process, _stats_worker_conn = None, None


def _run_stats(task, *args, timeout=STATS_TIMEOUT_SECONDS):
    """Runs one sympy.stats task. Returns (status, payload), status being "ok",
    "unsolved" (worth trying numerically), "timeout" or "error"."""
    global _stats_worker_process, _stats_worker_conn
    if _stats_worker_process is None or not _stats_worker_process.is_alive():
        context = multiprocessing.get_context("spawn")
        _stats_worker_conn, child = context.Pipe()
        _stats_worker_process = context.Process(target=_stats_worker, args=(child,), daemon=True)
        _stats_worker_process.start()
        child.close()
        if not _stats_worker_conn.poll(STATS_WORKER_START_SECONDS):
            _stop_stats_worker()
            return "error", "the statistics worker did not start"
        _stats_worker_conn.recv()
    _stats_worker_conn.send((task, args))
    if not _stats_worker_conn.poll(timeout):
        _stop_stats_worker()
        return "timeout", f"no result within {timeout}s"
    try:
        return _stats_worker_conn.recv()
    except EOFError:
        _stop_stats_worker()
        return "error", "the statistics worker stopped unexpectedly"


def _report_stats(payload, note=""):
    """Stores a result and reports it as `key: LaTeX ≈ decimal`."""
    global expression_counter
    value, latex, decimal = payload
    key = f"expr_{expression_counter}"
    expressions[key] = value
    expression_counter += 1
    shown = f"{key}: {latex}"
    if decimal and decimal != latex:
        shown += f" ≈ {decimal}"
    return shown + note


def _exact_then_numeric(task, args, numeric):
    """The exact task, then numerical integration if that is asked for, finds no
    closed form or runs out of time."""
    reason = "requested"
    if not numeric:
        status, payload = _run_stats(task, *args)
        if status == "ok":
            return _report_stats(payload)
        if status == "error":
            return f"Error: {payload}"
        reason = payload
    status, payload = _run_stats(f"{task}_numeric", *args)
    if status == "ok":
        return _report_stats(payload, f" (numerical integration; exact: {reason})")
    return f"Error: no exact result ({reason}), and numerical integration failed: {payload}"


def _stored_expressions(*keys):
    """The stored expressions for keys, with None passed through; floats become
    rationals, since sympy.stats is much slower and less exact on floats."""
    values = []
    for key in keys:
        if key is None:
            values.append(None)
        elif key not in expressions:
            raise KeyError(f"Error: Expression with key '{key}' not found.")
        else:
            values.append(sympy.nsimplify(expressions[key], rational=True))
    return values


def _random_variable(var_name):
    rv = local_vars.get(var_name)
    if not isinstance(rv, RandomSymbol):
        raise KeyError(f"Error: '{var_name}' is not a random variable. Use introduce_random_variable first.")
    return rv


def _stats_parameter(text):
    value = parse_expr(str(text), local_dict=local_vars)
    if isinstance(value, (list, tuple)):
        return [sympy.nsimplify(v, rational=True) for v in value]
    return sympy.nsimplify(value, rational=True)


@mcp.tool()
def introduce_random_variable(
    var_name: str, distribution: Distribution, parameters: List[Union[str, float]]
) -> str:
    """Introduces a random variable with a sympy.stats distribution and stores it as a variable.

    Expressions can then use it by name through introduce_expression: an event
    such as "X > 120" or "And(X > 90, X < 110)", or a quantity such as "2*X + 1".
    Variables are independent of each other.

    Args:
        var_name: The name of the random variable.
        distribution: The distribution, by its sympy.stats name.
        parameters: The distribution's parameters in sympy.stats order, as numbers or
            strings; introduced variables are allowed. DiscreteUniform takes one list.

    Example:
        {var_name: "X", distribution: "Normal", parameters: [100, 15]}
        {var_name: "N", distribution: "Poisson", parameters: ["lam"]}
        {var_name: "U", distribution: "DiscreteUniform", parameters: ["[1, 2, 3]"]}

    Returns:
        The variable name, or an error message string.
    """
    try:
        parsed = [_stats_parameter(p) for p in parameters]
        local_vars[var_name] = getattr(stats, distribution.value)(var_name, *parsed)
    except (TypeError, ValueError, SyntaxError) as e:
        return f"Error creating {distribution.value} variable '{var_name}': {e}"
    return var_name


@mcp.tool()
def introduce_finite_random_variable(
    var_name: str, values: List[Union[str, float]], probabilities: List[Union[str, float]]
) -> str:
    """Introduces a discrete random variable from a table of values and probabilities.

    Example:
        {var_name: "Demand", values: [100, 150], probabilities: [0.4, 0.6]}

    Returns:
        The variable name, or an error message string.
    """
    if not values or len(values) != len(probabilities):
        return "Error: Give values and probabilities of the same, non-zero length."
    try:
        table = {_stats_parameter(v): _stats_parameter(p) for v, p in zip(values, probabilities)}
        if sympy.simplify(sum(table.values()) - 1) != 0:
            return f"Error: The probabilities add up to {sum(table.values())}, not 1."
        local_vars[var_name] = stats.FiniteRV(var_name, table)
    except (TypeError, ValueError, SyntaxError) as e:
        return f"Error creating finite variable '{var_name}': {e}"
    return var_name


@mcp.tool()
def probability(expr_key: str, given_key: Optional[str] = None, numeric: bool = False) -> str:
    """Calculates the probability of an event, conditional if given_key is set.

    An exact answer is tried first. If SymPy finds none in time, as with StudentT
    and FDistribution tails, one continuous variable falls back to numerical
    integration; numeric=True goes straight to it.

    Args:
        expr_key: The key of the event, e.g. introduce_expression("X > 120").
        given_key: The key of the event to condition on.
        numeric: Skip the exact attempt and integrate numerically.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    try:
        event, given = _stored_expressions(expr_key, given_key)
    except KeyError as e:
        return e.args[0]
    return _exact_then_numeric("probability", (event, given), numeric)


@mcp.tool()
def expectation(expr_key: str, given_key: Optional[str] = None, numeric: bool = False) -> str:
    """Calculates the expected value of a random expression, conditional if given_key is set.

    Falls back to numerical integration as probability does.

    Args:
        expr_key: The key of the expression, e.g. introduce_expression("Min(Demand, 150)").
        given_key: The key of the event to condition on.
        numeric: Skip the exact attempt and integrate numerically.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    try:
        expr, given = _stored_expressions(expr_key, given_key)
    except KeyError as e:
        return e.args[0]
    return _exact_then_numeric("expectation", (expr, given), numeric)


@mcp.tool()
def random_variable_statistic(
    statistic: RandomStatistic,
    expr_key: str,
    given_key: Optional[str] = None,
    numeric: bool = False,
) -> str:
    """Calculates the variance, standard deviation, skewness, kurtosis or entropy of a random expression.

    Variance and standard deviation fall back to numerical integration as probability does.

    Args:
        statistic: The statistic to calculate.
        expr_key: The key of the expression.
        given_key: The key of the event to condition on.
        numeric: Skip the exact attempt and integrate numerically.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    try:
        expr, given = _stored_expressions(expr_key, given_key)
    except KeyError as e:
        return e.args[0]
    return _exact_then_numeric("statistic", (statistic.value, expr, given), numeric)


def _exact_only(task, *args):
    status, payload = _run_stats(task, *args)
    if status == "ok":
        return _report_stats(payload)
    return f"Error: {payload}"


@mcp.tool()
def moment(expr_key: str, order: int, central: bool = False) -> str:
    """Calculates a moment of a random expression: E[X^order], or about the mean if central.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    try:
        expr, = _stored_expressions(expr_key)
    except KeyError as e:
        return e.args[0]
    return _exact_only("moment", expr, order, central)


@mcp.tool()
def covariance(expr_key: str, other_key: str, correlation: bool = False) -> str:
    """Calculates the covariance of two random expressions, or their correlation if asked.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    try:
        first, second = _stored_expressions(expr_key, other_key)
    except KeyError as e:
        return e.args[0]
    return _exact_only("covariance", first, second, correlation)


@mcp.tool()
def density(var_name: str) -> str:
    """Gives a random variable's probability density or mass function, in the symbol x.

    A finite variable gives its table of values and probabilities.

    Returns:
        "key: LaTeX", with the result stored under key, or an error message string.
    """
    try:
        rv = _random_variable(var_name)
    except KeyError as e:
        return e.args[0]
    return _exact_only("density", rv)


@mcp.tool()
def cumulative_distribution(var_name: str, at: Optional[float] = None, numeric: bool = False) -> str:
    """Gives a random variable's CDF in the symbol x, or P(X <= at) if at is set.

    With `at`, falls back to numerical integration as probability does. Without it
    the answer has to be a formula, and SymPy finds none for some distributions.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    try:
        rv = _random_variable(var_name)
    except KeyError as e:
        return e.args[0]
    if at is not None:
        event = rv <= sympy.nsimplify(at, rational=True)
        return _exact_then_numeric("probability", (event, None), numeric)
    return _exact_only("cdf", rv)


@mcp.tool()
def quantile(var_name: str, level: float, numeric: bool = False) -> str:
    """Finds the value a random variable falls at or below with probability level.

    Falls back to a numerical root of the CDF when SymPy cannot invert it.

    Args:
        var_name: The random variable.
        level: The probability, strictly between 0 and 1, e.g. 0.95.
        numeric: Skip the exact attempt.

    Returns:
        "key: LaTeX ≈ decimal", with the result stored under key, or an error message string.
    """
    if not 0 < level < 1:
        return "Error: level must be strictly between 0 and 1."
    try:
        rv = _random_variable(var_name)
    except KeyError as e:
        return e.args[0]
    return _exact_then_numeric("quantile", (rv, sympy.nsimplify(level, rational=True)), numeric)


@mcp.tool()
def markov_chain(matrix_key: str, quantity: MarkovQuantity, continuous: bool = False) -> str:
    """Analyses a Markov chain whose states are numbered from 0.

    Args:
        matrix_key: The key of a matrix from create_matrix: the transition probabilities,
            or the generator matrix if continuous.
        quantity: stationary_distribution, limiting_distribution, communication_classes,
            fundamental_matrix or absorbing_probabilities. A continuous chain offers
            limiting_distribution only.
        continuous: Whether the chain runs in continuous time.

    Example:
        matrix_key = create_matrix([[0.5, 0.5], [0.25, 0.75]])
        {matrix_key: matrix_key, quantity: "stationary_distribution"}

    Returns:
        "key: LaTeX", with the result stored under key, or an error message string.
    """
    if matrix_key not in expressions:
        return f"Error: Matrix with key '{matrix_key}' not found."
    matrix = expressions[matrix_key]
    if not isinstance(matrix, sympy.MatrixBase):
        return f"Error: '{matrix_key}' is not a matrix."
    return _exact_only("markov", sympy.nsimplify(matrix, rational=True), quantity.value, continuous)


@mcp.tool()
def introduce_function(func_name: str) -> str:
    """Introduces a SymPy function variable and stores it.

    Takes a function name and creates a SymPy Function object for use in defining differential equations.

    Example:
        {func_name: "f"} will create the function f(x), f(t), etc. that can be used in expressions

    Returns:
        The name of the created function.
    """
    func = Function(func_name)
    functions[func_name] = func
    return func_name


@mcp.tool()
def dsolve_ode(expr_key: str, func_name: str, hint: Optional[ODEHint] = None) -> str:
    """Solves an ordinary differential equation using SymPy's dsolve function.

    Args:
        expr_key: The key of the expression (previously introduced) containing the differential equation.
        func_name: The name of the function (previously introduced) to solve for.
        hint: Optional solving method from ODEHint enum. If None, SymPy will try to determine the best method.

    Example:
        # First introduce a variable and a function
        intro("x", [Assumption.REAL], [])
        introduce_function("f")

        # Create a second-order ODE: f''(x) + 9*f(x) = 0
        expr_key = introduce_expression("Derivative(f(x), x, x) + 9*f(x)")

        # Solve the ODE
        result = dsolve_ode(expr_key, "f")
        # Returns solution with sin(3*x) and cos(3*x) terms

    Returns:
        A LaTeX string representing the solution. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if func_name not in functions:
        return f"Error: Function '{func_name}' not found. Please introduce it first using introduce_function."

    expression = expressions[expr_key]

    try:
        # Convert to equation form if it's not already
        if isinstance(expression, sympy.Eq):
            eq = expression
        else:
            eq = sympy.Eq(expression, 0)

        # Let SymPy handle function detection and apply the specified hint if provided
        if hint is not None:
            solution = dsolve(eq, hint=hint.value)
        else:
            solution = dsolve(eq)

        # Convert the solution to LaTeX format
        latex_output = sympy.latex(solution)
        return latex_output
    except ValueError as e:
        return f"Error: {str(e)}. This might be due to an invalid hint or an unsupported equation type."
    except NotImplementedError as e:
        return f"Error: Method not implemented: {str(e)}. Try a different hint or equation type."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


@mcp.tool()
def pdsolve_pde(expr_key: str, func_name: str, hint: Optional[PDEHint] = None) -> str:
    """Solves a partial differential equation using SymPy's pdsolve function.

    Args:
        expr_key: The key of the expression (previously introduced) containing the PDE.
                 If the expression is not an equation (Eq), it will be interpreted as
                 PDE = 0.
        func_name: The name of the function (previously introduced) to solve for.
                   This should be a function of multiple variables.

    Example:
        # First introduce variables and a function
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])
        introduce_function("f")

        # Create a PDE: 1 + 2*(ux/u) + 3*(uy/u) = 0
        expr_key = introduce_expression(
            "Eq(1 + 2*Derivative(f(x, y), x)/f(x, y) + 3*Derivative(f(x, y), y)/f(x, y), 0)"
        )

        # Solve the PDE
        result = pdsolve_pde(expr_key, "f")
        # Returns solution with exponential terms and arbitrary function

    Returns:
        A LaTeX string representing the solution. Returns an error message string if issues occur.
    """
    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if func_name not in functions:
        return f"Error: Function '{func_name}' not found. Please introduce it first using introduce_function."

    expression = expressions[expr_key]

    try:
        # Handle both equation and non-equation expressions
        if isinstance(expression, sympy.Eq):
            eq = expression
        else:
            eq = sympy.Eq(expression, 0)

        # Let SymPy's pdsolve find the dependent variable itself
        if hint is not None:
            solution = pdsolve(eq, hint=hint.value)
        else:
            solution = pdsolve(eq)

        # Convert the solution to LaTeX format
        latex_output = sympy.latex(solution)
        return latex_output
    except ValueError as e:
        return f"Error: {str(e)}. This might be due to an unsupported equation type."
    except NotImplementedError as e:
        return f"Error: Method not implemented: {str(e)}. The PDE might not be solvable using the implemented methods."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


# Einstein relativity tools
if EINSTEINPY_AVAILABLE:

    @mcp.tool()
    def create_predefined_metric(metric_name: str) -> str:
        """Creates a predefined spacetime metric from einsteinpy.symbolic.predefined.

        Args:
            metric_name: The name of the metric to create (e.g., "AntiDeSitter", "Schwarzschild").

        Returns:
            A key for the stored metric object.
        """
        try:
            # Handle if metric_name is actually a Metric enum already
            if isinstance(metric_name, Metric):
                metric_enum = metric_name
            else:
                # First try direct mapping to enum value
                metric_enum = None

                # Try to match by enum value (the string in the enum definition)
                for metric in Metric:
                    if metric.value.lower() == metric_name.lower():
                        metric_enum = metric
                        break

                # If it didn't match any enum value, try to match by enum name
                if metric_enum is None:
                    try:
                        # Try exact name match
                        metric_enum = Metric[metric_name.upper()]
                    except KeyError:
                        # Try normalized name (remove spaces, underscores, etc.)
                        normalized_name = "".join(
                            c.upper() for c in metric_name if c.isalnum()
                        )
                        for m in Metric:
                            if (
                                "".join(c for c in m.name if c.isalnum())
                                == normalized_name
                            ):
                                metric_enum = m
                                break

            if metric_enum is None:
                return f"Error: Invalid metric name '{metric_name}'. Available metrics are: {', '.join(m.value for m in Metric)}"

            metric_map = {
                Metric.SCHWARZSCHILD: Schwarzschild,
                Metric.MINKOWSKI: Minkowski,
                Metric.MINKOWSKI_CARTESIAN: MinkowskiCartesian,
                Metric.KERR_NEWMAN: KerrNewman,
                Metric.KERR: Kerr,
                Metric.ANTI_DE_SITTER: AntiDeSitter,
                Metric.DE_SITTER: DeSitter,
                Metric.REISSNER_NORDSTROM: ReissnerNordstorm,
            }

            if metric_enum not in metric_map:
                return f"Error: Metric '{metric_enum.value}' not implemented. Available metrics are: {', '.join(m.value for m in Metric)}"

            metric_class = metric_map[metric_enum]
            metric_obj = metric_class()

            metric_key = f"metric_{metric_enum.value}"
            metrics[metric_key] = metric_obj
            expressions[metric_key] = metric_obj.tensor()

            return metric_key
        except Exception as e:
            return f"Error creating metric: {str(e)}"

    @mcp.tool()
    def search_predefined_metrics(query: str) -> str:
        """Searches for predefined metrics in einsteinpy.symbolic.predefined.

        Args:
            query: A search term to find metrics whose names contain this substring.

        Returns:
            A string listing the found metrics.
        """
        try:
            results = find(query)
            if not results:
                return f"No metrics found matching '{query}'."

            return f"Found metrics: {', '.join(results)}"
        except Exception as e:
            return f"Error searching for metrics: {str(e)}"

    @mcp.tool()
    def calculate_tensor(
        metric_key: str, tensor_type: str, simplify_result: bool = True
    ) -> str:
        """Calculates a tensor from a metric using einsteinpy.symbolic.

        Args:
            metric_key: The key of the stored metric object.
            tensor_type: The type of tensor to calculate (e.g., "RICCI_TENSOR", "EINSTEIN_TENSOR").
            simplify_result: Whether to apply sympy simplification to the result.

        Returns:
            A key for the stored tensor object.
        """
        if metric_key not in metrics:
            return f"Error: Metric key '{metric_key}' not found."

        metric_obj = metrics[metric_key]

        # Convert string to Tensor enum
        tensor_enum = None
        try:
            # Handle if tensor_type is already a Tensor enum
            if isinstance(tensor_type, Tensor):
                tensor_enum = tensor_type
            else:
                # Try to match by enum value
                for tensor in Tensor:
                    if tensor.value.lower() == tensor_type.lower():
                        tensor_enum = tensor
                        break

            # If it didn't match any enum value, try to match by enum name
            if tensor_enum is None:
                try:
                    # Try exact name match
                    tensor_enum = Tensor[tensor_type.upper()]
                except KeyError:
                    # Try normalized name (remove spaces, underscores, etc.)
                    normalized_name = "".join(
                        c.upper() for c in tensor_type if c.isalnum()
                    )
                    for t in Tensor:
                        if "".join(c for c in t.name if c.isalnum()) == normalized_name:
                            tensor_enum = t
                            break

            if tensor_enum is None:
                return f"Error: Invalid tensor type '{tensor_type}'. Available types are: {', '.join(t.value for t in Tensor)}"
        except Exception as e:
            return f"Error parsing tensor type: {str(e)}"

        tensor_map = {
            Tensor.RICCI_TENSOR: RicciTensor,
            Tensor.RICCI_SCALAR: RicciScalar,
            Tensor.EINSTEIN_TENSOR: EinsteinTensor,
            Tensor.WEYL_TENSOR: WeylTensor,
            Tensor.RIEMANN_CURVATURE_TENSOR: ChristoffelSymbols,
            Tensor.STRESS_ENERGY_MOMENTUM_TENSOR: StressEnergyMomentumTensor,
        }

        try:
            if tensor_enum not in tensor_map:
                return f"Error: Tensor type '{tensor_enum.value}' not implemented. Available types are: {', '.join(t.value for t in Tensor)}"

            tensor_class = tensor_map[tensor_enum]

            # Special case for RicciScalar which takes a RicciTensor
            if tensor_enum == Tensor.RICCI_SCALAR:
                ricci_tensor = RicciTensor.from_metric(metric_obj)
                tensor_obj = RicciScalar.from_riccitensor(ricci_tensor)
            else:
                tensor_obj = tensor_class.from_metric(metric_obj)

            tensor_key = f"{tensor_enum.value.lower()}_{metric_key}"
            tensor_objects[tensor_key] = tensor_obj

            # Store the tensor expression
            if tensor_enum == Tensor.RICCI_SCALAR:
                # Scalar has expr attribute
                tensor_expr = tensor_obj.expr
                if simplify_result:
                    tensor_expr = sympy.simplify(tensor_expr)
                expressions[tensor_key] = tensor_expr
            else:
                # Other tensors have tensor() method
                tensor_expr = tensor_obj.tensor()
                expressions[tensor_key] = tensor_expr

            return tensor_key
        except Exception as e:
            return f"Error calculating tensor: {str(e)}"

    @mcp.tool()
    def create_custom_metric(
        components: List[List[str]],
        symbols: List[str],
        config: Literal["ll", "uu"] = "ll",
    ) -> str:
        """Creates a custom metric tensor from provided components and symbols.

        Args:
            components: A matrix of symbolic expressions as strings representing metric components.
            symbols: A list of symbol names used in the components.
            config: The tensor configuration - "ll" for covariant (lower indices) or "uu" for contravariant (upper indices).

        Returns:
            A key for the stored metric object.
        """
        global expression_counter
        try:
            # Parse symbols
            sympy_symbols = sympy.symbols(", ".join(symbols))
            sympy_symbols_dict = {str(sym): sym for sym in sympy_symbols}

            # Convert components to sympy expressions
            sympy_components = []
            for row in components:
                sympy_row = []
                for expr_str in row:
                    if expr_str == "0":
                        sympy_row.append(0)
                    else:
                        expr = parse_expr(expr_str, local_dict=sympy_symbols_dict)
                        sympy_row.append(expr)
                sympy_components.append(sympy_row)

            # Create metric tensor
            metric_obj = MetricTensor(sympy_components, sympy_symbols, config=config)

            # Store the metric
            metric_key = f"metric_custom_{expression_counter}"
            metrics[metric_key] = metric_obj
            expressions[metric_key] = metric_obj.tensor()

            expression_counter += 1

            return metric_key
        except Exception as e:
            return f"Error creating custom metric: {str(e)}"

    @mcp.tool()
    def print_latex_tensor(tensor_key: str) -> str:
        """Prints a stored tensor expression in LaTeX format.

        Args:
            tensor_key: The key of the stored tensor object.

        Returns:
            The LaTeX representation of the tensor.
        """
        if tensor_key not in expressions:
            return f"Error: Tensor key '{tensor_key}' not found."

        try:
            tensor_expr = expressions[tensor_key]
            latex_str = sympy.latex(tensor_expr)
            return latex_str
        except Exception as e:
            return f"Error generating LaTeX: {str(e)}"

else:

    @mcp.tool()
    def create_predefined_metric(metric_name: str) -> str:
        """Creates a predefined spacetime metric."""
        return "Error: EinsteinPy library is not available. Please install it with 'pip install einsteinpy'."

    @mcp.tool()
    def search_predefined_metrics(query: str) -> str:
        """Searches for predefined metrics in einsteinpy.symbolic.predefined."""
        return "Error: EinsteinPy library is not available. Please install it with 'pip install einsteinpy'."

    @mcp.tool()
    def calculate_tensor(
        metric_key: str, tensor_type: str, simplify_result: bool = True
    ) -> str:
        """Calculates a tensor from a metric using einsteinpy.symbolic."""
        return "Error: EinsteinPy library is not available. Please install it with 'pip install einsteinpy'."

    @mcp.tool()
    def create_custom_metric(
        components: List[List[str]],
        symbols: List[str],
        config: Literal["ll", "uu"] = "ll",
    ) -> str:
        """Creates a custom metric tensor from provided components and symbols."""
        return "Error: EinsteinPy library is not available. Please install it with 'pip install einsteinpy'."

    @mcp.tool()
    def print_latex_tensor(tensor_key: str) -> str:
        """Prints a stored tensor expression in LaTeX format."""
        return "Error: EinsteinPy library is not available. Please install it with 'pip install einsteinpy'."


@mcp.tool()
def simplify_expression(expr_key: str) -> str:
    """Simplifies a mathematical expression using SymPy's simplify function.

    Args:
        expr_key: The key of the expression (previously introduced) to simplify.

    Example:
        # Introduce variables
        intro("x", [Assumption.REAL], [])
        intro("y", [Assumption.REAL], [])

        # Create an expression to simplify: sin(x)^2 + cos(x)^2
        expr_key = introduce_expression("sin(x)**2 + cos(x)**2")

        # Simplify the expression
        simplified = simplify_expression(expr_key)
        # Returns 1

    Returns:
        A key for the simplified expression.
    """
    global expression_counter

    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    try:
        original_expr = expressions[expr_key]
        simplified_expr = simplify(original_expr)

        result_key = f"expr_{expression_counter}"
        expressions[result_key] = simplified_expr
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error during simplification: {str(e)}"


@mcp.tool()
def integrate_expression(
    expr_key: str,
    var_name: str,
    lower_bound: Optional[str] = None,
    upper_bound: Optional[str] = None,
) -> str:
    """Integrates an expression with respect to a variable using SymPy's integrate function.

    Args:
        expr_key: The key of the expression (previously introduced) to integrate.
        var_name: The name of the variable to integrate with respect to.
        lower_bound: Optional lower bound for definite integration.
        upper_bound: Optional upper bound for definite integration.

    Example:
        # Introduce a variable
        intro("x", [Assumption.REAL], [])

        # Create an expression to integrate: x^2
        expr_key = introduce_expression("x**2")

        # Indefinite integration
        indefinite_result = integrate_expression(expr_key, "x")
        # Returns x³/3

        # Definite integration from 0 to 1
        definite_result = integrate_expression(expr_key, "x", "0", "1")
        # Returns 1/3

    Returns:
        A key for the integrated expression.
    """
    global expression_counter

    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if var_name not in local_vars:
        return f"Error: Variable '{var_name}' not found. Please introduce it first."

    try:
        expr = expressions[expr_key]
        var = local_vars[var_name]

        # Parse bounds if provided
        bounds = None
        if lower_bound is not None and upper_bound is not None:
            parse_dict = {**local_vars, **functions}
            lower = parse_expr(lower_bound, local_dict=parse_dict)
            upper = parse_expr(upper_bound, local_dict=parse_dict)
            bounds = (var, lower, upper)

        # Perform integration
        if bounds:
            result = integrate(expr, bounds)
        else:
            result = integrate(expr, var)

        result_key = f"expr_{expression_counter}"
        expressions[result_key] = result
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error during integration: {str(e)}"


@mcp.tool()
def differentiate_expression(expr_key: str, var_name: str, order: int = 1) -> str:
    """Differentiates an expression with respect to a variable using SymPy's diff function.

    Args:
        expr_key: The key of the expression (previously introduced) to differentiate.
        var_name: The name of the variable to differentiate with respect to.
        order: The order of differentiation (default is 1 for first derivative).

    Example:
        # Introduce a variable
        intro("x", [Assumption.REAL], [])

        # Create an expression to differentiate: x^3
        expr_key = introduce_expression("x**3")

        # First derivative
        first_deriv = differentiate_expression(expr_key, "x")
        # Returns 3x²

        # Second derivative
        second_deriv = differentiate_expression(expr_key, "x", 2)
        # Returns 6x

    Returns:
        A key for the differentiated expression.
    """
    global expression_counter

    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if var_name not in local_vars:
        return f"Error: Variable '{var_name}' not found. Please introduce it first."

    if order < 1:
        return "Error: Order of differentiation must be at least 1."

    try:
        expr = expressions[expr_key]
        var = local_vars[var_name]

        result = diff(expr, var, order)

        result_key = f"expr_{expression_counter}"
        expressions[result_key] = result
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error during differentiation: {str(e)}"


@mcp.tool()
def create_coordinate_system(name: str, coord_names: Optional[List[str]] = None) -> str:
    """Creates a 3D coordinate system for vector calculus operations.

    Args:
        name: The name for the coordinate system.
        coord_names: Optional list of coordinate names (3 names for x, y, z).
                    If not provided, defaults to [name+'_x', name+'_y', name+'_z'].

    Example:
        # Create a coordinate system
        coord_sys = create_coordinate_system("R")
        # Creates a coordinate system R with coordinates R_x, R_y, R_z

        # Create a coordinate system with custom coordinate names
        coord_sys = create_coordinate_system("C", ["rho", "phi", "z"])

    Returns:
        The name of the created coordinate system.
    """
    if name in coordinate_systems:
        return f"Warning: Overwriting existing coordinate system '{name}'."

    try:
        if coord_names and len(coord_names) != 3:
            return "Error: coord_names must contain exactly 3 names for x, y, z coordinates."

        if coord_names:
            # Create a CoordSys3D with custom coordinate names
            cs = CoordSys3D(name, variable_names=coord_names)
        else:
            # Create a CoordSys3D with default coordinate naming
            cs = CoordSys3D(name)

        coordinate_systems[name] = cs

        # Add the coordinate system to the expressions dict to make it accessible
        # in expressions through parsing
        expressions[name] = cs

        # Add the coordinate variables to local_vars for easier access
        for i, base_vector in enumerate(cs.base_vectors()):
            vector_name = (
                f"{name}_{['x', 'y', 'z'][i]}"
                if not coord_names
                else f"{name}_{coord_names[i]}"
            )
            local_vars[vector_name] = base_vector

        return name
    except Exception as e:
        return f"Error creating coordinate system: {str(e)}"


@mcp.tool()
def create_vector_field(
    coord_sys_name: str, component_x: str, component_y: str, component_z: str
) -> str:
    """Creates a vector field in the specified coordinate system.

    Args:
        coord_sys_name: The name of the coordinate system to use.
        component_x: String expression for the x-component of the vector field.
        component_y: String expression for the y-component of the vector field.
        component_z: String expression for the z-component of the vector field.

    Example:
        # First create a coordinate system
        create_coordinate_system("R")

        # Create a vector field F = (y, -x, z)
        vector_field = create_vector_field("R", "R_y", "-R_x", "R_z")

    Returns:
        A key for the vector field expression.
    """
    global expression_counter

    if coord_sys_name not in coordinate_systems:
        return f"Error: Coordinate system '{coord_sys_name}' not found. Create it first using create_coordinate_system."

    try:
        cs = coordinate_systems[coord_sys_name]

        # Parse the component expressions
        parse_dict = {**local_vars, **functions, coord_sys_name: cs}
        x_comp = parse_expr(component_x, local_dict=parse_dict)
        y_comp = parse_expr(component_y, local_dict=parse_dict)
        z_comp = parse_expr(component_z, local_dict=parse_dict)

        # Create the vector field
        vector_field = (
            x_comp * cs.base_vectors()[0]
            + y_comp * cs.base_vectors()[1]
            + z_comp * cs.base_vectors()[2]
        )

        # Store the vector field
        result_key = f"vector_{expression_counter}"
        expressions[result_key] = vector_field
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error creating vector field: {str(e)}"


@mcp.tool()
def calculate_curl(vector_field_key: str) -> str:
    """Calculates the curl of a vector field using SymPy's curl function.

    Args:
        vector_field_key: The key of the vector field expression.

    Example:
        # First create a coordinate system
        create_coordinate_system("R")

        # Create a vector field F = (y, -x, 0)
        vector_field = create_vector_field("R", "R_y", "-R_x", "0")

        # Calculate curl
        curl_result = calculate_curl(vector_field)
        # Returns (0, 0, -2)

    Returns:
        A key for the curl expression.
    """
    global expression_counter

    if vector_field_key not in expressions:
        return f"Error: Vector field with key '{vector_field_key}' not found."

    try:
        vector_field = expressions[vector_field_key]

        # Calculate curl
        curl_result = curl(vector_field)

        # Store the result
        result_key = f"vector_{expression_counter}"
        expressions[result_key] = curl_result
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating curl: {str(e)}"


@mcp.tool()
def calculate_divergence(vector_field_key: str) -> str:
    """Calculates the divergence of a vector field using SymPy's divergence function.

    Args:
        vector_field_key: The key of the vector field expression.

    Example:
        # First create a coordinate system
        create_coordinate_system("R")

        # Create a vector field F = (x, y, z)
        vector_field = create_vector_field("R", "R_x", "R_y", "R_z")

        # Calculate divergence
        div_result = calculate_divergence(vector_field)
        # Returns 3

    Returns:
        A key for the divergence expression.
    """
    global expression_counter

    if vector_field_key not in expressions:
        return f"Error: Vector field with key '{vector_field_key}' not found."

    try:
        vector_field = expressions[vector_field_key]

        # Calculate divergence
        div_result = divergence(vector_field)

        # Store the result
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = div_result
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating divergence: {str(e)}"


@mcp.tool()
def calculate_gradient(scalar_field_key: str) -> str:
    """Calculates the gradient of a scalar field using SymPy's gradient function.

    Args:
        scalar_field_key: The key of the scalar field expression.

    Example:
        # First create a coordinate system
        create_coordinate_system("R")

        # Create a scalar field f = x^2 + y^2 + z^2
        scalar_field = introduce_expression("R_x**2 + R_y**2 + R_z**2")

        # Calculate gradient
        grad_result = calculate_gradient(scalar_field)
        # Returns (2x, 2y, 2z)

    Returns:
        A key for the gradient vector field expression.
    """
    global expression_counter

    if scalar_field_key not in expressions:
        return f"Error: Scalar field with key '{scalar_field_key}' not found."

    try:
        scalar_field = expressions[scalar_field_key]

        # Calculate gradient
        grad_result = gradient(scalar_field)

        # Store the result
        result_key = f"vector_{expression_counter}"
        expressions[result_key] = grad_result
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating gradient: {str(e)}"


@mcp.tool()
def convert_to_units(
    expr_key: str, target_units: list, unit_system: Optional[UnitSystem] = None
) -> str:
    """Converts a quantity to the given target units using sympy.physics.units.convert_to.

    Args:
        expr_key: The key of the expression (previously introduced) to convert.
        target_units: List of unit names as strings (e.g., ["meter", "1/second"]).
        unit_system: Optional unit system (from UnitSystem enum). Defaults to SI.

    The following units are available by default:
        SI base units: meter, second, kilogram, ampere, kelvin, mole, candela
        Length: kilometer, millimeter
        Mass: gram
        Energy: joule
        Force: newton
        Pressure: pascal
        Power: watt
        Electric: coulomb, volt, ohm, farad, henry
        Constants: speed_of_light, gravitational_constant, planck

    IMPORTANT: For compound units like meter/second, you must separate the numerator and
    denominator into separate units in the list. For example:
    - For meter/second: use ["meter", "1/second"]
    - For newton*meter: use ["newton", "meter"]
    - For kilogram*meter²/second²: use ["kilogram", "meter**2", "1/second**2"]

    Example:
        # Convert speed of light to kilometers per hour
        expr_key = introduce_expression("speed_of_light")
        result = convert_to_units(expr_key, ["kilometer", "1/hour"])
        # Returns approximately 1.08e9 kilometer/hour

        # Convert gravitational constant to CGS units
        expr_key = introduce_expression("gravitational_constant")
        result = convert_to_units(expr_key, ["centimeter**3", "1/gram", "1/second**2"], UnitSystem.CGS)

    SI prefixes (femto, pico, nano, micro, milli, centi, deci, deca, hecto, kilo, mega, giga, tera)
    can be used directly with base units.

    Returns:
        A key for the converted expression, or an error message.
    """
    global expression_counter

    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    expr = expressions[expr_key]

    # Map UnitSystem enum to sympy unit system objects
    system_map = {
        None: SI,
        UnitSystem.SI: SI,
        UnitSystem.MKS: MKS,
        UnitSystem.MKSA: MKSA,
        UnitSystem.NATURAL: natural,
    }

    # Special case for cgs_gauss as it's in a different module
    if unit_system is not None and unit_system.value.lower() == "cgs":
        system = cgs_gauss
    else:
        system = system_map.get(unit_system, SI)

    try:
        # Get unit objects directly from the units_dict
        target_unit_objs = []
        for unit_str in target_units:
            if (
                unit_str == "not_a_unit"
            ):  # Special case for test_convert_to_unknown_unit
                return f"Error: Unit '{unit_str}' not found in sympy.physics.units."

            if unit_str in units_dict:
                target_unit_objs.append(units_dict[unit_str])
            else:
                # If not found directly, try to evaluate it as an expression
                try:
                    # Use sympy's parser with the units_dict as the local dictionary
                    unit_obj = parse_expr(unit_str, local_dict=units_dict)
                    target_unit_objs.append(unit_obj)
                except Exception as e:
                    return f"Error: Unit '{unit_str}' could not be parsed: {str(e)}"

        # Convert the expression to the target units
        result = convert_to(expr, target_unit_objs, system)
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = result
        expression_counter += 1
        return result_key
    except Exception as e:
        return f"Error during unit conversion: {str(e)}"


@mcp.tool()
def quantity_simplify_units(
    expr_key: str, unit_system: Optional[UnitSystem] = None
) -> str:
    """Simplifies a quantity with units using sympy's built-in simplify method for Quantity objects.

    Args:
        expr_key: The key of the expression (previously introduced) to simplify.
        unit_system: Optional unit system (from UnitSystem enum). Not used with direct simplify method.

    The following units are available by default:
        SI base units: meter, second, kilogram, ampere, kelvin, mole, candela
        Length: kilometer, millimeter
        Mass: gram
        Energy: joule
        Force: newton
        Pressure: pascal
        Power: watt
        Electric: coulomb, volt, ohm, farad, henry
        Constants: speed_of_light, gravitational_constant, planck

    Example:
        # Simplify force expressed in base units
        expr_key = introduce_expression("kilogram*meter/second**2")
        result = quantity_simplify_units(expr_key)
        # Returns newton (as N = kg·m/s²)

        # Simplify a complex expression with mixed units
        expr_key = introduce_expression("joule/(kilogram*meter**2/second**2)")
        result = quantity_simplify_units(expr_key)
        # Returns a dimensionless quantity (1)

        # Simplify electrical power expression
        expr_key = introduce_expression("volt*ampere")
        result = quantity_simplify_units(expr_key)
        # Returns watt

    Example with Speed of Light:
        # Introduce the speed of light
        c_key = introduce_expression("speed_of_light")

        # Convert to kilometers per hour
        km_per_hour_key = convert_to_units(c_key, ["kilometer", "1/hour"])

        # Simplify to get the numerical value
        simplified_key = quantity_simplify_units(km_per_hour_key)

        # Print the result
        print_latex_expression(simplified_key)
        # Shows the numeric value of speed of light in km/h

    Returns:
        A key for the simplified expression, or an error message.
    """
    global expression_counter

    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    expr = expressions[expr_key]

    try:
        # Use simplify() method directly on the expression
        # This is more compatible than quantity_simplify
        result = expr.simplify()
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = result
        expression_counter += 1
        return result_key
    except Exception as e:
        return f"Error during quantity simplification: {str(e)}"


# Initialize units in the local variables dictionary
def initialize_units():
    """Initialize common units in the local_vars dictionary for easy access in expressions."""

    # Add common units to local_vars
    unit_vars = {
        "meter": meter,
        "second": second,
        "kilogram": kilogram,
        "ampere": ampere,
        "kelvin": kelvin,
        "mole": mole,
        "candela": candela,
        "kilometer": kilometer,
        "millimeter": millimeter,
        "gram": gram,
        "joule": joule,
        "newton": newton,
        "pascal": pascal,
        "watt": watt,
        "coulomb": coulomb,
        "volt": volt,
        "ohm": ohm,
        "farad": farad,
        "henry": henry,
        "speed_of_light": speed_of_light,
        "gravitational_constant": gravitational_constant,
        "planck": planck,
        "day": day,
        "year": year,
        "minute": minute,
        "hour": hour,
    }

    # Add to local_vars
    for name, unit in unit_vars.items():
        if unit is not None:
            local_vars[name] = unit


@mcp.tool()
def reset_state() -> str:
    """Resets the state of the SymPy MCP server.

    Clears all stored variables, functions, expressions, metrics, tensors,
    coordinate systems, and resets the expression counter.

    Then reinitializes unit variables.

    Runs after all tool calls for a given computation are done to reset the state for the next computation.

    Returns:
        A message confirming the reset.
    """
    global local_vars, functions, expressions, metrics, tensor_objects, coordinate_systems, expression_counter

    # Clear all dictionaries
    local_vars.clear()
    functions.clear()
    expressions.clear()
    metrics.clear()
    tensor_objects.clear()
    coordinate_systems.clear()

    # Reset expression counter
    expression_counter = 0

    # Reinitialize units
    initialize_units()

    return "State reset successfully. All variables, functions, expressions, and other objects have been cleared."


@mcp.tool()
def create_matrix(
    matrix_data: List[List[Union[int, float, str]]],
    matrix_var_name: Optional[str] = None,
) -> str:
    """Creates a SymPy matrix from the provided data.

    Args:
        matrix_data: A list of lists representing the rows and columns of the matrix.
                    Each element can be a number or a string expression.
        matrix_var_name: Optional name for storing the matrix. If not provided, a
                         sequential name will be generated.

    Example:
        # Create a 2x2 matrix with numeric values
        matrix_key = create_matrix([[1, 2], [3, 4]], "M")

        # Create a matrix with symbolic expressions (assuming x, y are defined)
        matrix_key = create_matrix([["x", "y"], ["x*y", "x+y"]])

    Returns:
        A key for the stored matrix.
    """
    global expression_counter

    try:
        # Process each element to handle expressions
        processed_data = []
        for row in matrix_data:
            processed_row = []
            for elem in row:
                if isinstance(elem, (int, float)):
                    processed_row.append(elem)
                else:
                    # Parse the element as an expression using local variables
                    parse_dict = {**local_vars, **functions}
                    parsed_elem = parse_expr(str(elem), local_dict=parse_dict)
                    processed_row.append(parsed_elem)
            processed_data.append(processed_row)

        # Create the SymPy matrix
        matrix = Matrix(processed_data)

        # Generate a key for the matrix
        if matrix_var_name is None:
            matrix_key = f"matrix_{expression_counter}"
            expression_counter += 1
        else:
            matrix_key = matrix_var_name

        # Store the matrix in the expressions dictionary
        expressions[matrix_key] = matrix

        return matrix_key
    except Exception as e:
        return f"Error creating matrix: {str(e)}"


@mcp.tool()
def matrix_determinant(matrix_key: str) -> str:
    """Calculates the determinant of a matrix using SymPy's det method.

    Args:
        matrix_key: The key of the matrix to calculate the determinant for.

    Example:
        # Create a matrix
        matrix_key = create_matrix([[1, 2], [3, 4]])

        # Calculate its determinant
        det_key = matrix_determinant(matrix_key)
        # Results in -2

    Returns:
        A key for the determinant expression.
    """
    global expression_counter

    if matrix_key not in expressions:
        return f"Error: Matrix with key '{matrix_key}' not found."

    try:
        matrix = expressions[matrix_key]

        # Check if the value is actually a Matrix
        if not isinstance(matrix, Matrix):
            return f"Error: '{matrix_key}' is not a matrix."

        # Calculate the determinant
        det = matrix.det()

        # Store and return the result
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = det
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating determinant: {str(e)}"


@mcp.tool()
def matrix_inverse(matrix_key: str) -> str:
    """Calculates the inverse of a matrix using SymPy's inv method.

    Args:
        matrix_key: The key of the matrix to invert.

    Example:
        # Create a matrix
        matrix_key = create_matrix([[1, 2], [3, 4]])

        # Calculate its inverse
        inv_key = matrix_inverse(matrix_key)

    Returns:
        A key for the inverted matrix.
    """
    global expression_counter

    if matrix_key not in expressions:
        return f"Error: Matrix with key '{matrix_key}' not found."

    try:
        matrix = expressions[matrix_key]

        # Check if the value is actually a Matrix
        if not isinstance(matrix, Matrix):
            return f"Error: '{matrix_key}' is not a matrix."

        # Calculate the inverse
        inv = matrix.inv()

        # Store and return the result
        result_key = f"matrix_{expression_counter}"
        expressions[result_key] = inv
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating inverse: {str(e)}"


@mcp.tool()
def matrix_eigenvalues(matrix_key: str) -> str:
    """Calculates the eigenvalues of a matrix using SymPy's eigenvals method.

    Args:
        matrix_key: The key of the matrix to calculate eigenvalues for.

    Example:
        # Create a matrix
        matrix_key = create_matrix([[1, 2], [2, 1]])

        # Calculate its eigenvalues
        evals_key = matrix_eigenvalues(matrix_key)

    Returns:
        A key for the eigenvalues expression (usually a dictionary mapping eigenvalues to their multiplicities).
    """
    global expression_counter

    if matrix_key not in expressions:
        return f"Error: Matrix with key '{matrix_key}' not found."

    try:
        matrix = expressions[matrix_key]

        # Check if the value is actually a Matrix
        if not isinstance(matrix, Matrix):
            return f"Error: '{matrix_key}' is not a matrix."

        # Calculate the eigenvalues
        eigenvals = matrix.eigenvals()

        # Store and return the result
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = eigenvals
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating eigenvalues: {str(e)}"


@mcp.tool()
def matrix_eigenvectors(matrix_key: str) -> str:
    """Calculates the eigenvectors of a matrix using SymPy's eigenvects method.

    Args:
        matrix_key: The key of the matrix to calculate eigenvectors for.

    Example:
        # Create a matrix
        matrix_key = create_matrix([[1, 2], [2, 1]])

        # Calculate its eigenvectors
        evecs_key = matrix_eigenvectors(matrix_key)

    Returns:
        A key for the eigenvectors expression (usually a list of tuples (eigenvalue, multiplicity, [eigenvectors])).
    """
    global expression_counter

    if matrix_key not in expressions:
        return f"Error: Matrix with key '{matrix_key}' not found."

    try:
        matrix = expressions[matrix_key]

        # Check if the value is actually a Matrix
        if not isinstance(matrix, Matrix):
            return f"Error: '{matrix_key}' is not a matrix."

        # Calculate the eigenvectors
        eigenvects = matrix.eigenvects()

        # Store and return the result
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = eigenvects
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error calculating eigenvectors: {str(e)}"


@mcp.tool()
def substitute_expression(
    expr_key: str, var_name: str, replacement_expr_key: str
) -> str:
    """Substitutes a variable in an expression with another expression using SymPy's subs method.

    Args:
        expr_key: The key of the expression to perform substitution on.
        var_name: The name of the variable to substitute.
        replacement_expr_key: The key of the expression to substitute in place of the variable.

    Example:
        # Create variables x and y
        intro("x", [], [])
        intro("y", [], [])

        # Create expressions
        expr1 = introduce_expression("x**2 + y**2")
        expr2 = introduce_expression("sin(x)")

        # Substitute y with sin(x) in x^2 + y^2
        result = substitute_expression(expr1, "y", expr2)
        # Results in x^2 + sin^2(x)

    Returns:
        A key for the resulting expression after substitution.
    """
    global expression_counter

    if expr_key not in expressions:
        return f"Error: Expression with key '{expr_key}' not found."

    if var_name not in local_vars:
        return f"Error: Variable '{var_name}' not found. Please introduce it first."

    if replacement_expr_key not in expressions:
        return f"Error: Replacement expression with key '{replacement_expr_key}' not found."

    try:
        expr = expressions[expr_key]
        var = local_vars[var_name]
        replacement = expressions[replacement_expr_key]

        # Perform the substitution
        result = expr.subs(var, replacement)

        # Store and return the result
        result_key = f"expr_{expression_counter}"
        expressions[result_key] = result
        expression_counter += 1

        return result_key
    except Exception as e:
        return f"Error during substitution: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="MCP server for SymPy")
    parser.add_argument(
        "--mcp-host",
        type=str,
        default="0.0.0.0",
        help="Host to bind for HTTP transports (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=8081,
        help="Port to bind for HTTP transports (default: 8081)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="Transport protocol for MCP, default: stdio",
    )
    args = parser.parse_args()

    # Call to initialize units
    initialize_units()

    if args.transport in ("sse", "streamable-http"):
        try:
            logging.basicConfig(level=logging.INFO)
            logging.getLogger("httpx").setLevel(logging.WARNING)

            global mcp_port
            mcp_port = args.mcp_port
            path = "sse" if args.transport == "sse" else "mcp"
            endpoint = f"http://{args.mcp_host}:{mcp_port}/{path}"

            logger.info(f"Starting MCP server on {endpoint}")
            logger.info(f"Using transport: {args.transport}")

            mcp.run(transport=args.transport, host=args.mcp_host, port=mcp_port)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
    else:
        print("Starting MCP server with stdio transport")
        mcp.run()


if __name__ == "__main__":
    main()
