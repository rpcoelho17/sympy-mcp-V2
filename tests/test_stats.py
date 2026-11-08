# Copyright 2026 Rodrigo P. Coelho, sympy-mcp-V2 (https://github.com/rpcoelho17/sympy-mcp-V2).
# Apache-2.0; see LICENSE and NOTICE.
import re

import pytest
import sympy
import sympy.stats
from server import (
    covariance,
    create_matrix,
    cumulative_distribution,
    density,
    expectation,
    expressions,
    functions,
    intro,
    introduce_expression,
    introduce_finite_random_variable,
    introduce_random_variable,
    local_vars,
    markov_chain,
    moment,
    probability,
    quantile,
    random_variable_statistic,
)
from vars import Distribution, MarkovQuantity, RandomStatistic


@pytest.fixture(autouse=True)
def reset_globals():
    local_vars.clear()
    expressions.clear()
    functions.clear()
    import server

    server.expression_counter = 0
    yield


def rv(name, distribution, *parameters):
    assert introduce_random_variable(name, Distribution(distribution), list(parameters)) == name


def decimal(result):
    """The number in a "key: LaTeX ≈ decimal" result."""
    assert not result.startswith("Error"), result
    match = re.search(r"≈ (-?[\d.]+(?:e-?\d+)?)", result)
    # A result that is already a decimal is shown once, without the ≈.
    return float(match.group(1) if match else latex(result).split(" (")[0])


def latex(result):
    assert not result.startswith("Error"), result
    return result.split(": ", 1)[1].split(" ≈ ")[0]


class TestRandomVariables:
    def test_every_distribution_exists_in_sympy_stats(self):
        for distribution in Distribution:
            assert callable(getattr(sympy.stats, distribution.value))

    def test_wrong_parameter_count(self):
        result = introduce_random_variable("X", Distribution.NORMAL, [1])
        assert result.startswith("Error")

    def test_invalid_parameter(self):
        result = introduce_random_variable("X", Distribution.NORMAL, [0, -1])
        assert result.startswith("Error")

    def test_finite_probabilities_must_add_to_one(self):
        assert introduce_finite_random_variable("Q", [1, 2], [0.5, 0.6]).startswith("Error")

    def test_finite_length_mismatch(self):
        assert introduce_finite_random_variable("Q", [1, 2], [1]).startswith("Error")

    def test_symbolic_parameter(self):
        intro("lam", [], [])
        local_vars["lam"] = sympy.Symbol("lam", positive=True)
        rv("T", "Exponential", "lam")
        assert latex(expectation(introduce_expression("T"))) == r"\frac{1}{lam}"


class TestProbability:
    def test_normal_tail(self):
        rv("X", "Normal", 0, 1)
        assert decimal(probability(introduce_expression("X > 1.96"))) == pytest.approx(0.0249979, abs=1e-6)

    def test_two_dice(self):
        rv("D1", "Die", 6)
        rv("D2", "Die", 6)
        assert latex(probability(introduce_expression("D1 + D2 > 7"))) == r"\frac{5}{12}"

    def test_conditional(self):
        rv("D", "Die", 6)
        result = probability(introduce_expression("D > 4"), introduce_expression("D > 2"))
        assert latex(result) == r"\frac{1}{2}"

    def test_poisson(self):
        rv("N", "Poisson", 3)
        assert decimal(probability(introduce_expression("N <= 2"))) == pytest.approx(0.4231901, abs=1e-6)

    def test_student_t_tail_falls_back_to_numeric(self):
        # The exact attempt does not finish; the tail is scipy's t.sf(2.228, 10).
        rv("T", "StudentT", 10)
        result = probability(introduce_expression("T > 2.228"))
        assert "numerical integration" in result
        assert decimal(result) == pytest.approx(0.0250059, abs=1e-5)

    def test_f_tail_numeric(self):
        rv("F", "FDistribution", 3, 20)
        result = probability(introduce_expression("F > 3.098"), numeric=True)
        assert decimal(result) == pytest.approx(0.0500184, abs=1e-5)

    def test_numeric_conditional(self):
        rv("X", "Normal", 0, 1)
        result = probability(introduce_expression("X > 1"), introduce_expression("X > 0"), numeric=True)
        assert decimal(result) == pytest.approx(0.3173105, abs=1e-5)

    def test_numeric_two_variables_is_refused(self):
        rv("A", "Normal", 0, 1)
        rv("B", "Normal", 0, 1)
        result = probability(introduce_expression("A + B > 1"), numeric=True)
        assert result.startswith("Error")

    def test_missing_expression(self):
        assert probability("nonexistent").startswith("Error")


class TestExpectationAndSpread:
    def test_expectation_of_min(self):
        introduce_finite_random_variable("Demand", [100, 200], [0.4, 0.6])
        assert latex(expectation(introduce_expression("Min(Demand, 150)"))) == "130"

    def test_conditional_expectation(self):
        rv("D", "Die", 6)
        assert latex(expectation(introduce_expression("D"), introduce_expression("D > 4"))) == r"\frac{11}{2}"

    def test_f_mean_falls_back_to_numeric(self):
        # SymPy has no moment generating function for F; the mean is d2 / (d2 - 2).
        rv("F", "FDistribution", 3, 20)
        result = expectation(introduce_expression("F"))
        assert "numerical integration" in result
        assert decimal(result) == pytest.approx(20 / 18, abs=1e-6)

    def test_cauchy_mean_is_not_reported_as_a_number(self):
        rv("C", "Cauchy", 0, 1)
        assert expectation(introduce_expression("C"), numeric=True).startswith("Error")

    def test_student_t_infinite_variance_is_not_reported_as_a_number(self):
        rv("T", "StudentT", 2)
        key = introduce_expression("T")
        assert random_variable_statistic(RandomStatistic.VARIANCE, key, numeric=True).startswith("Error")

    @pytest.mark.parametrize("statistic,expected", [
        (RandomStatistic.VARIANCE, "4"),
        (RandomStatistic.STANDARD_DEVIATION, "2"),
        (RandomStatistic.SKEWNESS, "2"),
        (RandomStatistic.KURTOSIS, "9"),
    ])
    def test_exponential_statistics(self, statistic, expected):
        rv("E", "Exponential", sympy.Rational(1, 2).__str__())
        assert latex(random_variable_statistic(statistic, introduce_expression("E"))) == expected

    def test_entropy(self):
        rv("E", "Exponential", 2)
        result = random_variable_statistic(RandomStatistic.ENTROPY, introduce_expression("E"))
        assert decimal(result) == pytest.approx(1 - 0.693147, abs=1e-5)

    def test_student_t_variance_numeric(self):
        rv("T", "StudentT", 10)
        result = random_variable_statistic(RandomStatistic.VARIANCE, introduce_expression("T"))
        assert decimal(result) == pytest.approx(1.25, abs=1e-5)

    def test_moments(self):
        rv("E", "Exponential", 2)
        key = introduce_expression("E")
        assert latex(moment(key, 3)) == r"\frac{3}{4}"
        assert latex(moment(key, 2, central=True)) == r"\frac{1}{4}"

    def test_covariance_and_correlation(self):
        rv("A", "Normal", 0, 1)
        rv("B", "Normal", 0, 1)
        total, a = introduce_expression("A + B"), introduce_expression("A")
        assert latex(covariance(total, a)) == "1"
        assert decimal(covariance(total, a, correlation=True)) == pytest.approx(0.7071068, abs=1e-6)


class TestDistributionFunctions:
    def test_density(self):
        rv("E", "Exponential", 2)
        assert latex(density("E")) == r"2 e^{- 2 x}"

    def test_finite_density(self):
        introduce_finite_random_variable("Q", [1, 2], [0.25, 0.75])
        assert "frac{3}{4}" in latex(density("Q"))

    def test_cdf_formula(self):
        rv("U", "Uniform", 0, 2)
        assert "x" in latex(cumulative_distribution("U"))

    def test_cdf_at_value(self):
        rv("U", "Uniform", 0, 2)
        assert latex(cumulative_distribution("U", at=0.5)) == r"\frac{1}{4}"

    def test_f_cdf_at_value_numeric(self):
        rv("F", "FDistribution", 3, 20)
        assert decimal(cumulative_distribution("F", at=3.098, numeric=True)) == pytest.approx(0.9499816, abs=1e-5)

    def test_quantile_exact(self):
        rv("E", "Exponential", 2)
        assert decimal(quantile("E", 0.95)) == pytest.approx(1.497866, abs=1e-5)

    def test_quantile_normal(self):
        rv("X", "Normal", 0, 1)
        assert decimal(quantile("X", 0.975)) == pytest.approx(1.959964, abs=1e-5)

    def test_quantile_chi_square_numeric(self):
        # SymPy returns an unsolved set here, so the numeric root takes over.
        rv("C", "ChiSquared", 5)
        assert decimal(quantile("C", 0.95)) == pytest.approx(11.070498, abs=1e-4)

    def test_quantile_student_t_numeric(self):
        rv("T", "StudentT", 10)
        assert decimal(quantile("T", 0.975, numeric=True)) == pytest.approx(2.228139, abs=1e-4)

    def test_quantile_bad_level(self):
        rv("X", "Normal", 0, 1)
        assert quantile("X", 1.5).startswith("Error")

    def test_not_a_random_variable(self):
        intro("x", [], [])
        assert density("x").startswith("Error")


class TestMarkovChain:
    def test_stationary_distribution(self):
        key = create_matrix([[0.5, 0.5], [0.25, 0.75]])
        result = markov_chain(key, MarkovQuantity.STATIONARY_DISTRIBUTION)
        assert r"\frac{1}{3}" in result and r"\frac{2}{3}" in result

    def test_limiting_distribution(self):
        key = create_matrix([[0.5, 0.5], [0.25, 0.75]])
        result = markov_chain(key, MarkovQuantity.LIMITING_DISTRIBUTION)
        assert r"\frac{1}{3}" in result

    def test_absorbing_probabilities(self):
        key = create_matrix([[1, 0, 0], [0.5, 0, 0.5], [0, 0, 1]])
        result = markov_chain(key, MarkovQuantity.ABSORBING_PROBABILITIES)
        assert r"\frac{1}{2}" in result

    def test_communication_classes(self):
        key = create_matrix([[0.5, 0.5], [0.25, 0.75]])
        assert not markov_chain(key, MarkovQuantity.COMMUNICATION_CLASSES).startswith("Error")

    def test_continuous_limiting_distribution(self):
        # M/M/1/2 with arrival rate 1 and service rate 2.
        key = create_matrix([[-1, 1, 0], [2, -3, 1], [0, 2, -2]])
        result = markov_chain(key, MarkovQuantity.LIMITING_DISTRIBUTION, continuous=True)
        assert r"\frac{4}{7}" in result

    def test_not_a_matrix(self):
        assert markov_chain(introduce_expression("1 + 1"), MarkovQuantity.STATIONARY_DISTRIBUTION).startswith("Error")
