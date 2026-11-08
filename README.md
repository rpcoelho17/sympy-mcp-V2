# Symbolic Algebra MCP Server

> **About this version.** This is **sympy-mcp-V2** by Rodrigo P. Coelho
> ([github.com/rpcoelho17/sympy-mcp-V2](https://github.com/rpcoelho17/sympy-mcp-V2)), which
> is a fork of Stephen Diehl's [sympy-mcp](https://github.com/sdiehl/sympy-mcp).
> Rodrigo P. Coelho contributed both numerical MCP tools in this fork:
>
> | Tool | What it does |
> |------|--------------|
> | `evaluate_numerically` | Evaluates a stored expression to a decimal (`evalf`), at a requested precision, optionally substituting numeric values for its variables |
> | `solve_numerically` | Finds a numeric root of an equation or system with `nsolve` (Newton's method) from initial guesses — works where `solve_algebraically` finds no closed form, e.g. `cos(x) = x` |
>
> `solve_numerically` uses the initial guesses to search for a root. Try different guesses
> to search for other roots.
>
> Rodrigo P. Coelho also contributed the `sympy.stats` tools, which add random variables and
> what can be calculated from them:
>
> | Tool | What it does |
> |------|--------------|
> | `introduce_random_variable` | Creates a variable from any single-variable `sympy.stats` distribution (Normal, Poisson, StudentT, ...); expressions then use it by name |
> | `introduce_finite_random_variable` | Creates a discrete variable from a table of values and probabilities |
> | `probability` | Probability of an event, conditional too |
> | `expectation` | Expected value of a random expression, conditional too |
> | `random_variable_statistic` | Variance, standard deviation, skewness, kurtosis or entropy |
> | `moment` | Raw or central moment of any order |
> | `covariance` | Covariance or correlation of two random expressions |
> | `density` | Probability density or mass function |
> | `cumulative_distribution` | The CDF, or its value at a point |
> | `quantile` | The value a variable falls at or below with a given probability |
> | `markov_chain` | Stationary and limiting distributions, communication classes, fundamental matrix and absorbing probabilities |
>
> These calls run in a worker process with a 20-second limit, because some exact calculations
> never finish, such as a StudentT tail probability. When that happens, or SymPy has no method,
> `probability`, `expectation`, variance, standard deviation, `quantile` and
> `cumulative_distribution` at a point fall back to numerical integration for one continuous
> variable, and the result says so. A quantity that does not exist, such as a Cauchy variable's
> mean, comes back as an error. Joint distributions and random sampling are not exposed.
>
> If you use or redistribute this work, credit it as described in [License](#license).

**The original library is no longer maintained. It is Apache licensed, so feel free to fork it.**

Sympy-MCP is a Model Context Protocol server for allowing LLMs to autonomously perform symbolic mathematics and computer algebra. It exposes numerous tools from SymPy's core functionality to MCP clients for manipulating mathematical expressions and equations.

## Why?

Language models are absolutely abysmal at symbolic manipulation. They hallucinate variables, make up random constants, permute terms and generally make a mess. But we have computer algebra systems specifically built for symbolic manipulation, so we can use tool-calling to orchestrate a sequence of transforms so that the symbolic kernel does all the heavy lifting.

While you can certainly have an LLM generate Mathematica or Python code, if you want to use the LLM as an agent or on-the-fly calculator, it's a better experience to use the MCP server and expose the symbolic tools directly.

The server exposes a subset of symbolic mathematics capabilities including algebraic equation solving, integration and differentiation, vector calculus, tensor calculus for general relativity, and both ordinary and partial differential equations. 

For example, you can ask it in natural language to solve a differential equation:

> Solve the damped harmonic oscillator with forcing term: the mass-spring-damper system described by the differential equation where m is mass, c is the damping coefficient, k is the spring constant, and F(t) is an external force.

$$ m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F(t) $$

Or involving general relativity:

> Compute the trace of the Ricci tensor $R_{\mu\nu}$ using the inverse metric $g^{\mu\nu}$ for Anti-de Sitter spacetime to determine its constant scalar curvature $R$.

## Usage

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). The project requires Python 3.12 or later. uv can install the requested Python version if needed.

Clone the V2 repository and install its locked dependencies:

```shell
git clone https://github.com/rpcoelho17/sympy-mcp-V2.git
cd sympy-mcp-V2
uv sync --locked
```

Then follow the setup for your MCP client below. To run the stdio server from the checkout, use `uv run --locked mcp run server.py:mcp`. An MCP client communicates with the process over standard input and output.

For general relativity tools, install the optional [EinsteinPy](https://github.com/einsteinpy/einsteinpy) dependency with `uv sync --locked --group relativity`. Add `--group relativity` after `run` in your client's uv arguments to keep that group enabled when the client starts the server.

## Available Tools

The sympy-mcp server provides the following tools for symbolic mathematics:

| Tool | Tool ID | Description |
|------|-------|-------------|
| Variable Introduction | `intro` | Introduces a variable with specified assumptions and stores it |
| Multiple Variables | `intro_many` | Introduces multiple variables with specified assumptions simultaneously |
| Expression Parser | `introduce_expression` | Parses an expression string using available local variables and stores it |
| LaTeX Printer | `print_latex_expression` | Prints a stored expression in LaTeX format, along with variable assumptions |
| Algebraic Solver | `solve_algebraically` | Solves an equation algebraically for a given variable over a given domain |
| Linear Solver | `solve_linear_system` | Solves a system of linear equations |
| Nonlinear Solver | `solve_nonlinear_system` | Solves a system of nonlinear equations |
| Numeric Evaluator | `evaluate_numerically` | Evaluates an expression to a decimal number, optionally at given variable values |
| Numeric Solver | `solve_numerically` | Numerically solves an equation or system near initial guesses (nsolve) |
| Random Variable | `introduce_random_variable` | Creates a random variable from a sympy.stats distribution |
| Finite Random Variable | `introduce_finite_random_variable` | Creates a discrete random variable from values and probabilities |
| Probability | `probability` | Calculates the probability of an event, conditional if asked |
| Expectation | `expectation` | Calculates the expected value of a random expression, conditional if asked |
| Random Statistics | `random_variable_statistic` | Calculates variance, standard deviation, skewness, kurtosis or entropy |
| Moment | `moment` | Calculates a raw or central moment |
| Covariance | `covariance` | Calculates the covariance or correlation of two random expressions |
| Density | `density` | Gives a probability density or mass function |
| CDF | `cumulative_distribution` | Gives a CDF, or its value at a point |
| Quantile | `quantile` | Finds the value a random variable falls at or below with a given probability |
| Markov Chain | `markov_chain` | Analyses a discrete or continuous Markov chain from its matrix |
| Function Variable | `introduce_function` | Introduces a function variable for use in differential equations |
| ODE Solver | `dsolve_ode` | Solves an ordinary differential equation |
| PDE Solver | `pdsolve_pde` | Solves a partial differential equation |
| Standard Metric | `create_predefined_metric` | Creates a predefined spacetime metric (e.g. Schwarzschild, Kerr, Minkowski) |
| Metric Search | `search_predefined_metrics` | Searches available predefined metrics |
| Tensor Calculator | `calculate_tensor` | Calculates tensors from a metric (Ricci, Einstein, Weyl tensors) |
| Custom Metric | `create_custom_metric` | Creates a custom metric tensor from provided components and symbols |
| Tensor LaTeX | `print_latex_tensor` | Prints a stored tensor expression in LaTeX format |
| Simplifier | `simplify_expression` | Simplifies a mathematical expression using SymPy's canonicalize function |
| Substitution | `substitute_expression` | Substitutes a variable with an expression in another expression |
| Integration | `integrate_expression` | Integrates an expression with respect to a variable |
| Differentiation | `differentiate_expression` | Differentiates an expression with respect to a variable |
| Coordinates | `create_coordinate_system` | Creates a 3D coordinate system for vector calculus operations |
| Vector Field | `create_vector_field` | Creates a vector field in the specified coordinate system |
| Curl | `calculate_curl` | Calculates the curl of a vector field |
| Divergence | `calculate_divergence` | Calculates the divergence of a vector field |
| Gradient | `calculate_gradient` | Calculates the gradient of a scalar field |
| Unit Converter | `convert_to_units` | Converts a quantity to given target units |
| Unit Simplifier | `quantity_simplify_units` | Simplifies a quantity with units |
| Matrix Creator | `create_matrix` | Creates a SymPy matrix from the provided data |
| Determinant | `matrix_determinant` | Calculates the determinant of a matrix |
| Matrix Inverse | `matrix_inverse` | Calculates the inverse of a matrix |
| Eigenvalues | `matrix_eigenvalues` | Calculates the eigenvalues of a matrix |
| Eigenvectors | `matrix_eigenvectors` | Calculates the eigenvectors of a matrix |

By default variables are predefined with assumptions (similar to how the [symbols()](https://docs.sympy.org/latest/modules/core.html#sympy.core.symbol.symbols) function works in SymPy). Unless otherwise specified the defaut assumptions is that a variable is complex, commutative, term over the complex field $\mathbb{C}$.

| Property | Value |
|----------|-------|
| `commutative` | true |
| `complex` | true |
| `finite` | true |
| `infinite` | false |

## Claude Desktop Setup

Open the Claude Desktop configuration file for your platform:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

Add the following server to `mcpServers`. Replace the placeholders with the absolute paths to the uv executable and the V2 checkout directory. In Windows JSON paths, use forward slashes or escape each backslash. Restart Claude Desktop after saving.

```json
{
  "mcpServers": {
    "sympy-mcp-v2": {
      "command": "/ABSOLUTE_PATH_TO_UV",
      "args": [
        "run",
        "--locked",
        "--directory",
        "/ABSOLUTE_PATH_TO_SYMPY_MCP_V2",
        "mcp",
        "run",
        "server.py:mcp"
      ]
    }
  }
}
```

## Cursor Setup

Use the same configuration in `~/.cursor/mcp.json`, with the paths set to your uv executable and V2 checkout.

## VS Code Setup

Add the following to `.vscode/mcp.json` in your workspace. Replace both path placeholders, then start the server from the MCP configuration.

```json
{
  "servers": {
    "sympy-mcp-v2": {
      "command": "/ABSOLUTE_PATH_TO_UV",
      "args": [
        "run",
        "--locked",
        "--directory",
        "/ABSOLUTE_PATH_TO_SYMPY_MCP_V2",
        "mcp",
        "run",
        "server.py:mcp"
      ]
    }
  }
}
```

## HTTP Transport (Streamable HTTP / SSE)

The server supports MCP over HTTP using the [streamable-http transport](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports#streamable-http) introduced in MCP spec 2025-03-26. This replaces the legacy SSE transport and exposes a single `/mcp` endpoint that clients connect to over HTTP.

This is the recommended transport when running the server as a standalone process or in a container, because it allows any HTTP-capable MCP client to connect without needing to launch the server as a subprocess.

```bash
# Run locally with HTTP transport
uv run python server.py --transport streamable-http

# Override host/port
uv run python server.py --transport streamable-http --mcp-host 127.0.0.1 --mcp-port 9000
```

The legacy `--transport sse` flag is still supported for backward compatibility.

A `/healthcheck` endpoint is also exposed that runs a full MCP protocol round-trip (initialize → tools/list → session teardown) and returns `{"status": "ok", "tool_count": N}`.

## Running in Container

From the V2 checkout, build and run the image with Docker:

```bash
docker build -t sympy-mcp-v2 .
docker run --rm -p 127.0.0.1:8081:8081 sympy-mcp-v2
```

Connect your HTTP-capable MCP client to `http://127.0.0.1:8081/mcp`.

If you use the included Docker Compose configuration, run `docker compose -f docker/docker-compose.yml up --build` from the checkout. Compose maps the container's port 8081 to host port 8082, so its MCP endpoint is `http://localhost:8082/mcp`.

## Example Interaction 1 : Differential Equations

Here's an example of how the tool-calling works in a conversation with an AI assistant using this MCP server:

**User**: Solve the damped harmonic oscillator with forcing term: the mass-spring-damper system described by the differential equation where m is mass, c is the damping coefficient, k is the spring constant, and F(t) is an external force.

$$ m\frac{d^2x}{dt^2} + c\frac{dx}{dt} + kx = F(t) $$

**Assistant**: (Internal tool call) `intro_many` to introduce variables:
```json
{
  "variables": [
    {"var_name": "t", "pos_assumptions": ["real"], "neg_assumptions": []},
    {"var_name": "m", "pos_assumptions": ["real", "positive"], "neg_assumptions": []},
    {"var_name": "c", "pos_assumptions": ["real", "positive"], "neg_assumptions": []},
    {"var_name": "k", "pos_assumptions": ["real", "positive"], "neg_assumptions": []}
  ]
}
```

**Assistant**: (Internal tool call) `introduce_function` to define x(t):
```json
{
  "func_name": "x"
}
```

**Assistant**: (Internal tool call) `introduce_expression` to create the differential equation:
```json
{
  "expr_str": "Eq(m*Derivative(x(t), t, t) + c*Derivative(x(t), t) + k*x(t), F(t))"
}
```

**Assistant**: (Internal tool call) `print_latex_expression` to display the equation nicely:
```json
{
  "expr_key": "expr_12"
}
```

Result:

$$
c \frac{d}{d t} x{\left(t \right)} + k x{\left(t \right)} + m \frac{d^{2}}{d t^{2}} x{\left(t \right)} = F{\left(t \right)}
$$

**Assistant**: (Internal tool call) `introduce_function` to define F(t):
```json
{
  "func_name": "F"
}
```

**Assistant**: (Internal tool call) `dsolve_ode` to solve the differential equation:
```json
{
  "expr_key": "expr_12",
  "func_name": "x"
}
```

Result:

$$
x{\left(t \right)} = C_{1} e^{\frac{t \left(- c + \sqrt{c^{2} - 4 k m}\right)}{2 m}} + C_{2} e^{- \frac{t \left(c + \sqrt{c^{2} - 4 k m}\right)}{2 m}} + \cdots
$$

## Example Interaction 2: General Relativity

**User**: Compute the trace of the Ricci tensor $R_{\mu\nu}$ using the inverse metric $g^{\mu\nu}$ for Anti-de Sitter spacetime to determine its constant scalar curvature $R$.

**Assistant**: (Internal tool call) `create_predefined_metric` to create the Anti-de Sitter metric:
```json
{
  "metric_name": "AntiDeSitter"
}
```

**Assistant**: (Internal tool call) `calculate_tensor` to calculate the Ricci tensor:
```json
{
  "metric_key": "metric_AntiDeSitter",
  "tensor_type": "RICCI_TENSOR",
  "simplify_result": true
}
```

**Assistant**: (Internal tool call) `print_latex_tensor` to display the Ricci tensor:
```json
{
  "tensor_key": "riccitensor_metric_AntiDeSitter"
}
```

Result:

$$
\left[\begin{matrix}3 & 0 & 0 & 0\\\\0 & - 3 \cos^{2}{\left(t \right)} & 0 & 0\\\\0 & 0 & - 3 \cos^{2}{\left(t \right)} \sinh^{2}{\left(\chi \right)} & 0\\\\0 & 0 & 0 & - 3 \sin^{2}{\left(\theta \right)} \cos^{2}{\left(t \right)} \sinh^{2}{\left(\chi \right)}\end{matrix}\right]
$$

**Assistant**: (Internal tool call) `calculate_tensor` to calculate the Ricci scalar:
```json
{
  "metric_key": "metric_AntiDeSitter",
  "tensor_type": "RICCI_SCALAR",
  "simplify_result": true
}
```

**Assistant**: (Internal tool call) `print_latex_tensor` to display the Ricci scalar:
```json
{
  "tensor_key": "ricciscalar_metric_AntiDeSitter"
}
```

Result:

$$
-12
$$

## Example Interaction 3: Coupled ODE System (Fluid Dynamics)

This example demonstrates solving a coupled system of ODEs and verifying the solution against an algebraic steady-state analysis, a task where LLMs typically hallucinate without a CAS to ground each step.

**User**:

> Use the sympy-mcp tools to solve this symbolically. Do not compute by hand.
>
> Two cylindrical tanks are connected by a pipe. Tank 1 has cross-sectional area A₁ = 2 m² and receives a constant inflow of Q = 0.5 m³/s. Water drains from Tank 1 into Tank 2 through a pipe with flow rate proportional to the height difference: q₁₂ = k·(h₁ - h₂) where k = 0.3 m²/s. Tank 2 has cross-sectional area A₂ = 1 m² and drains to the outside at rate q₂ = k·h₂ with the same k.
>
> Set up and solve the coupled system of ODEs for the water heights h₁(t) and h₂(t), starting from empty tanks (h₁(0) = 0, h₂(0) = 0). Then find the steady-state heights as t → ∞ by solving the equilibrium equations algebraically, and verify they match the long-term solution of the ODEs.

**Assistant**: (Internal tool chain)

1. `intro_many`: introduce `t`, `k`, `A1`, `A2`, `Q` with real/positive assumptions
2. `introduce_function` × 2: introduce `h1(t)` and `h2(t)` as unknown functions
3. `introduce_expression` × 2: encode the mass-balance ODEs:

$$A_1 \frac{dh_1}{dt} = Q - k(h_1 - h_2), \quad A_2 \frac{dh_2}{dt} = k(h_1 - h_2) - k h_2$$

4. `substitute_expression`: substitute numeric values for `k`, `A1`, `A2`, `Q`
5. `dsolve_ode` × 2: solve the coupled system; apply initial conditions via `substitute_expression`
6. `introduce_expression` × 2: encode equilibrium equations (derivatives set to zero)
7. `solve_linear_system`: solve the 2×2 algebraic system for `h1*`, `h2*`
8. `print_latex_expression`: display both the time-domain solution and the steady-state values

## Security Disclaimer

This server runs on your computer and gives the language model access to run Python logic. Notably it uses Sympy's `parse_expr` to parse mathematical expressions, which is uses `eval` under the hood, effectively allowing arbitrary code execution. By running the server, you are trusting the code that Claude generates. Running in the Docker image is slightly safer, but it's still a good idea to review the code before running it.


## License

Copyright 2025 Stephen Diehl (original sympy-mcp).
Modifications copyright 2026 Rodrigo P. Coelho (sympy-mcp-V2).

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

**Attribution.** Under section 4(d) of the license, anyone who redistributes this work or a
work derived from it must carry the attributions in the [NOTICE](NOTICE) file, by including
them in a NOTICE file, in their documentation, or wherever such third-party notices normally
appear. Credit both authors and sources:

- Rodrigo P. Coelho — https://github.com/rpcoelho17/sympy-mcp-V2
- Stephen Diehl — https://github.com/sdiehl/sympy-mcp
