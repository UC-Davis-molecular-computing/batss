import math
from typing import Iterable, Sequence
from mpmath import hyp3f2, mpf, binomial, hyper, mp
from mpmath import psi
import numpy as np
import numpy.typing as npt
from scipy.special import binom, gammaln


def main():
    from math import sqrt

    n = 10**3
    k = round(sqrt(n))
    o = 2
    g = 1
    trials = 10**6
    seed = 0

    rng = np.random.default_rng(seed)

    gammas_params = gammas_params_matching_hypo(n, k, o, g, 10)
    gammas_samples = sample_gammas_sum(rng, gammas_params, trials)
    hypo_samples = sample_hypos(rng, n, k, o, g, trials)
    print(f"gammas_samples: {gammas_samples}")
    print(f"hypo_samples: {hypo_samples}")

def relative_error(a: float, b: float) -> float:
    if a == 0:
        return abs(b)
    if b == 0:
        return abs(a)
    return abs(a - b) / min(abs(a), abs(b))

def sample_gammas_sum(
    rng: np.random.Generator, gammas: np.ndarray, size: int
) -> npt.NDArray[np.float64]:
    """
    Sample from a list of gamma distributions with parameters (shape, scale), given as a 2D numpy array,
    and then return their sum. Do this `size` times, and return the result as a 1D numpy array.
    """
    shapes = np.repeat(gammas[:, 0], size).reshape(gammas.shape[0], size)
    scales = np.repeat(gammas[:, 1], size).reshape(gammas.shape[0], size)
    samples = rng.gamma(shapes, scales)
    s = np.sum(samples, axis=0)
    return s


# TODO: measure time to sample hypo directly and compared to sampling gamma
def sample_hypo(rng: np.random.Generator, n: int, k: int, o: int, g: int) -> float:
    indices = np.arange(k)
    scales = 1.0 / binom(n + indices * g, o)
    exponential_samples = rng.exponential(scales, size=k)
    hypo_sample = np.sum(exponential_samples)
    return float(hypo_sample)


def sample_hypos(
    rng: np.random.Generator, n: int, k: int, o: int, g: int, size: int
) -> npt.NDArray[np.float64]:
    """
    Sample from a hypoexponential distribution summing exponentials having rates
    n choose c, n+g choose c, n+2*g choose c, ..., n+(k-1)*g choose c.
    (directly, by sampling k exponentials with those rates and summing them)
    """
    indices = np.arange(k)
    scales = 1.0 / binom(n + indices * g, o)
    scales = np.repeat(scales, size).reshape(scales.shape[0], size)
    exp_samples = rng.exponential(scales, size=(k, size))
    samples = np.sum(exp_samples, axis=0)
    return samples


def sample_gamma_matching_hypo(
    rng: np.random.Generator, n: int, k: int, o: int, g: int
) -> float:
    shape, scale = gamma_params_matching_hypo(n, k, o, g)
    sample = rng.gamma(shape, scale)
    return float(sample)


def gammas_params_matching_hypo(
    n: int, k: int, o: int, g: int, num_gammas: int
) -> npt.NDArray[np.float64]:
    """
    Compute the parameters of `num_gammas` Gamma distributions, whose sum matches the mean and variance of a
    hypoexponential distribution summing exponentials having rates
    (reciprocals of expected values of individual exponentials)
        n choose c, n+g choose c, n+2*g choose c, ..., n+(k-1)*g choose c.
    The parameters for the gammas are returned as a 2D ndarray representing pairs (shape, scale).

    If `num_gammas` evenly divides `k`, so that `k` / `num_gammas` is a integer `s`, each gamma distribution
    is chosen to match a hypoexponential distribution with `s` exponentials. The i'th such
    gamma distribution has the same mean and variance as the hypoexponential distribution
    corresponding to the i'th block of `s` exponentials in the original hypoexponential distribution.
    If `num_gammas` does not evenly divide `k`, the last gamma distribution is chosen to match a
    hypoexponential distribution corresponding to the final `k` % `num_gammas` exponentials in the
    original hypoexponential.
    """
    if num_gammas > k:
        raise ValueError("num_gammas must be less than or equal to k")
    if num_gammas <= 0:
        raise ValueError("num_gammas must be greater than 0")

    # Calculate the number of exponentials in each block
    block_size = k // num_gammas
    remainder = k % num_gammas

    gammas_f: list[tuple[float, float]] = []
    for i in range(num_gammas):
        # print(f'Block {i}: n={n+i*block_size}, k={block_size}')
        shape, scale = gamma_params_matching_hypo(n + i * block_size, block_size, o, g)
        gammas_f.append((shape, scale))

    if remainder > 0:
        # Handle the last block with the remainder
        # print(f'Block {num_gammas}: n={n+num_gammas*block_size}, k={remainder}')
        shape, scale = gamma_params_matching_hypo(
            n + num_gammas * block_size, remainder, o, g
        )
        gammas_f.append((shape, scale))

    gammas = np.array(gammas_f, dtype=np.float64)

    if np.min(gammas) < 0:
        raise ValueError(
            "Shape and scale parameters must be positive, "
            "but gammas contains negative entries:\n"
            f"{gammas}"
        )

    return gammas


from functools import lru_cache


# XXX: many of the return types here should be `mpf` instead of `float`, but
# mpmath does not appear to support using mpf as a type,, despite calling it a "type"
# in their documentation: https://mpmath.org/doc/current/basics.html#number-types
# @lru_cache
def gamma_params_matching_hypo(
    n: int,
    k: int,
    o: int,
    g: int,
) -> tuple[float, float]:
    """
    Compute the parameters (shape, scale) of a Gamma distribution that matches the
    mean and variance of a hypoexponential distribution summing exponentials with rates
    n choose c, n+g choose c, n+2*g choose c, ..., n+(k-1)*g choose c.
    """
    mean = mean_hypo_adaptive(n, k, o, g)
    var = var_hypo_adaptive(n, k, o, g)
    shape = mean**2 / var
    scale = var / mean
    return shape, scale


def mean_hypo_adaptive(n: int, k: int, o: int, g: int) -> float:
    """
    Compute the mean of a hypoexponential distribution summing exponentials with rates
    n choose o, n+g choose o, n+2*g choose o, ..., n+(k-1)*g choose o.

    The mean is given by the formula sum_{i=0}^{k-1} 1 / binomial(n + i * g, o),
    although it is calculated in one of a few different ways depending on which is 
    expected to be most efficient (hence the term "adaptive").

    Args:
        n: Starting value for the binomial coefficient
        k: Number of exponentials to sum
        o: Order parameter (the 'o' in 'n choose o')
        g: Step size between consecutive binomial coefficients

    Returns:
        The mean of the hypoexponential distribution
    """
    assert n >= 1, "n must be at least 1"
    assert k >= 1, "k must be at least 1"
    assert g >= 1, "g must be at least 1"
    assert o >= 1, "o must be at least 1"
    last_term = 1.0 / math.comb(n + (k - 1) * g, o)
    assert last_term > 0.0

    first_term = 1.0 / math.comb(n, o)
    assert first_term > 0.0
    relative_error_first_last_term = relative_error(first_term, last_term)
    if relative_error_first_last_term < 0.1:
        geom_mean = math.sqrt(first_term * last_term)
        print(f"first term: {first_term/ 1.0496696290309689e-13}, last term: {last_term/ 1.0496696290309689e-13}, geom_mean: {geom_mean/ 1.0496696290309689e-13}")
        return float(k * geom_mean)

    time_mean_direct_np = predicted_mean_direct_np(n, k, o, g)
    time_mean_hypo = predicted_mean_hypo(n, k, o, g)

    if time_mean_direct_np < time_mean_hypo:
        return mean_direct_np(n, k, o, g)
    else:
        return mean_hypo_polygamma_identity(n, k, o, g)


def var_hypo_adaptive(n: int, k: int, o: int, g: int) -> float:
    """
    Compute the variance of a hypoexponential distribution summing exponentials with rates
    n choose o, n+g choose o, n+2*g choose o, ..., n+(k-1)*g choose o.

    The variance is given by the formula sum_{i=0}^{k-1} 1 / binomial(n + i * g, o)^2,
    although it is calculated in one of a few different ways depending on which is 
    expected to be most efficient (hence the term "adaptive").

    Args:
        n: Starting value for the binomial coefficient
        k: Number of exponentials to sum
        o: Order parameter (the 'o' in 'n choose o')
        g: Step size between consecutive binomial coefficients

    Returns:
        The variance of the hypoexponential distribution
    """
    assert n >= 1, "n must be at least 1"
    assert k >= 1, "k must be at least 1"
    assert g >= 1, "g must be at least 1"
    assert o >= 1, "o must be at least 1"
    try:
        last_term = 1.0 / math.comb(n + (k - 1) * g, o) ** 2
    except OverflowError:
        return 0.0
    if last_term <= 0.0:
        return 0.0

    first_term = 1.0 / math.comb(n, o) ** 2
    assert first_term > 0.0
    relative_error_first_last_term = relative_error(first_term, last_term)
    if relative_error_first_last_term < 0.5:
        geom_mean = math.sqrt(first_term * last_term)
        return float(k * geom_mean)

    time_var_direct_np = predicted_var_direct_np(n, k, o, g)
    time_var_hypo = predicted_var_hypo(n, k, o, g)

    if time_var_direct_np < time_var_hypo:
        return var_direct_np(n, k, o, g)
    else:
        return var_hypo_polygamma_identity(n, k, o, g)


def precision_for(term) -> int:
    return (math.ceil(-math.log(term, 2)) + 53 + 29) // 30 * 30


def mean_hypo_polygamma_identity(n: int, k: int, o: int, g: int) -> float:
    try:
        last_term = 1.0 / math.comb(n + (k - 1) * g, o)
    except OverflowError:
        return 0.0
    if last_term <= 0.0:
        return 0.0
    
    precision = precision_for(last_term)
    with mp.workprec(precision):
        terms = []

        # We can compute successive binomial coefficients using the identity
        #   comb(n,   k) = comb(n,   k-1) * (n-k+1) // k
        # or using our variable names:
        #   comb(o-1, m) = comb(o-1, m-1) * (o-m) // m
        # In practice with small values of o, math.comb is much faster than psi, so 
        # we keep it this way for readability.
        assert o < 100 # If o >> 100, then the shortcut actually is helpful, e.g. math.comb(500,250)
                       # takes roughly the same time to compute as psi, about 5 μs.
        for m in range(o):
            coeff = math.comb(o - 1, m)
            coeff_with_sign = coeff if m % 2 == 0 else -coeff

            arg2 = mpf(n - (o - 1 - m)) / g
            arg1 = arg2 + k

            terms.extend([coeff_with_sign * psi(0, arg1), -coeff_with_sign * psi(0, arg2)])

        result = mp.fsum(terms) * o / g
    return float(result)


def var_hypo_polygamma_identity(n: int, k: int, o: int, g: int) -> float:
    try:
        last_term = 1.0 / math.comb(n + (k - 1) * g, o) ** 2
    except OverflowError:
        return 0.0
    if last_term <= 0.0:
        return 0.0

    precision = precision_for(last_term)
    with mp.workprec(precision):
        # first sum
        #   o^2 / g^2
        #   * sum_{m=0}^{o-1}
        #       binom{o-1}{m}^2
        #       * [ psi_1(     (n-(o-1-m)+ig) / g )
        #         - psi_1( k + (n-(o-1-m)+ig) / g )]
        first_sum_terms = []
        assert o < 100
        for m in range(o):
            # see comments in mean_hypo_polygamma_identity for why we use math.comb here
            # instead of being more clever about computing successive binomial coefficients.
            coeff = math.comb(o - 1, m) ** 2
            if (n - (o - 1 - m)) % g == 0:
                arg1 = (n - (o - 1 - m)) // g
            else:
                arg1 = mpf(n - (o - 1 - m)) / g
            arg2 = k + arg1
            first_sum_terms.extend([coeff * psi(1, arg1), -coeff * psi(1, arg2)])
        first_sum = mp.fsum(first_sum_terms)
        first_sum_multiplier = mpf(o**2) / (g**2)
        first_sum *= first_sum_multiplier

        # second sum
        #   + 2 o^2 / g
        #     sum_{m=0}^{o-1}
        #       sum_{j=m+1}^{o-1}
        #         (-1)^{m+j} / (m-j) * \binom{o-1}{m} * \binom{o-1}{j}
        #         * [
        #           + psi_0(     (n-(o-1-m)) / g )
        #           - psi_0( k + (n-(o-1-m)) / g )
        #           - psi_0(     (n-(o-1-j)) / g )
        #           + psi_0( k + (n-(o-1-j)) / g )
        #           ].
        second_sum_terms = []
        for m in range(o):
            o_minus_1_choose_m = math.comb(o - 1, m)
            for j in range(m + 1, o):
                o_minus_1_choose_j = math.comb(o - 1, j)
                negate = (m + j) % 2 == 1
                bin_coeff_prod = o_minus_1_choose_m * o_minus_1_choose_j
                if bin_coeff_prod % (m - j) == 0:
                    coeff = bin_coeff_prod // (m - j)
                else:
                    coeff = mpf(bin_coeff_prod) / (m - j)
                
                coeff_with_sign = coeff if not negate else -coeff

                arg1 = (
                    (n - (o - 1 - m)) // g
                    if (n - (o - 1 - m)) % g == 0
                    else mpf(n - (o - 1 - m)) / g
                )
                arg2 = k + arg1
                arg3 = (
                    (n - (o - 1 - j)) // g
                    if (n - (o - 1 - j)) % g == 0
                    else mpf(n - (o - 1 - j)) / g
                )
                arg4 = k + arg3
                if arg1 != arg3 or arg2 != arg4:
                    # otherwise the sums of terms will be identically 0
                    term1 = coeff_with_sign * psi(0, arg1)
                    term2 = -coeff_with_sign * psi(0, arg2)
                    term3 = -coeff_with_sign * psi(0, arg3)
                    term4 = coeff_with_sign * psi(0, arg4)
                    second_sum_terms.extend([term1, term2, term3, term4])
        second_sum = mp.fsum(second_sum_terms)
        second_sum_multiplier = mpf(2) * o**2 / g
        second_sum *= second_sum_multiplier

        result = first_sum + second_sum
        assert result > 0
        return result


def format_time(seconds):
    """Format time with appropriate units like %timeit does"""
    if seconds >= 1.0:
        return f"{seconds:.3g} s"
    elif seconds >= 1e-3:
        return f"{seconds*1e3:.3g} ms"
    elif seconds >= 1e-6:
        return f"{seconds*1e6:.3g} μs"
    else:
        return f"{seconds*1e9:.3g} ns"


def smart_timeit(stmt, print_: bool = True, setup: str = "pass", repeat=7):
    """timeit with automatic scaling and %timeit-like output"""
    import timeit

    timer = timeit.Timer(stmt, setup)

    # Auto-determine number of loops
    number, _ = timer.autorange()

    if number > 10:
        number = int(number / 10)

    # Take multiple measurements
    times = timer.repeat(repeat=repeat, number=number)
    # best_time = min(times) if number > 1 else float(np.median(times))
    best_time = float(np.median(times))
    time_per_loop = best_time / number

    if print_:
        # best_str = 'best' if number > 1 else 'median'
        best_str = "median"
        print(
            f"{format_time(time_per_loop)} per loop; {number} loops, {best_str} of {repeat}"
        )
    return time_per_loop


##########################################################
# Functions below here are early ideas not used anymore.


def general_mean_hypo_hypergeometric(n: int, k: int, o: int, g: int) -> float:
    # mpmath does not have a special implementation of hypergeometric 4F3 or above,
    # so we use the general implementation:
    # https://mpmath.org/doc/current/functions/hypergeometric.html#hyper
    # Wolfram alpha does not have a closed form for this, but we can infer the closed
    # form from the first few formulas for g=1, g=2, g=3:
    # g=1: https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+i%2C+o%29
    # g=2: https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+2*i%2C+o%29
    # g=3: https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1%2Fbinomial%28n+%2B+3*i%2C+o%29
    # XXX: This is VERY slow, it can take over a second to compute for example, for n=10**8, k=10**4, o=4, g=4

    as1 = [1.0]
    bs1 = []
    as2 = [1.0]
    bs2 = []

    for i in range(1, int(g) + 1):
        as1.append((n - o + i) / g)
        bs1.append((n + i) / g)

        as2.append(k + (n - o + i) / g)
        bs2.append(k + (n + i) / g)

    num1 = hyper(as1, bs1, 1)
    den1 = binomial(n, o)
    num2 = hyper(as2, bs2, 1)
    den2 = binomial(n + g * k, o)

    return num1 / den1 - num2 / den2


def mean_hypo_o2(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 2)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+2%29
    arg11 = k + (n - 1) / g
    arg12 = k + n / g
    arg21 = (n - 1) / g
    arg22 = n / g
    sum1 = psi(0, arg11) - psi(0, arg12)
    sum2 = psi(0, arg21) - psi(0, arg22)
    # print(f'mean_hypo_o2 {sum1=}')
    # print(f'mean_hypo_o2 {sum2=}')
    return 2 * (sum1 - sum2) / g


def mean_hypo_o3(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 3)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+3%29
    return (
        3
        * (
            psi(0, k + (n - 2) / g)
            - 2 * psi(0, k + (n - 1) / g)
            + psi(0, k + n / g)
            - (psi(0, (n - 2) / g) - 2 * psi(0, (n - 1) / g) + psi(0, n / g))
        )
    ) / g


def mean_hypo_o4(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 4)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+4%29
    return (
        4
        * (
            psi(0, k + (n - 3) / g)
            - 3 * psi(0, k + (n - 2) / g)
            + 3 * psi(0, k + (n - 1) / g)
            - psi(0, k + n / g)
            - (
                psi(0, (n - 3) / g)
                - 3 * psi(0, (n - 2) / g)
                + 3 * psi(0, (n - 1) / g)
                - psi(0, n / g)
            )
        )
    ) / g


def mean_hypo_o5(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 5)
    # https://www.wolframalpha.com/input?i=sum_%7Bi%3D0%7D%5E%7Bk-1%7D+1+%2F+binomial%28n+%2B+g*i%2C+5%29
    assert g >= 1, "g must be at least 1"
    # print(f'mean_hypo_o5: n={n}, k={k}, g={g}')
    return (
        5
        * (
            psi(0, k + (n - 4) / g)
            - 4 * psi(0, k + (n - 3) / g)
            + 6 * psi(0, k + (n - 2) / g)
            - 4 * psi(0, k + (n - 1) / g)
            + psi(0, k + n / g)
            - (
                psi(0, (n - 4) / g)
                - 4 * psi(0, (n - 3) / g)
                + 6 * psi(0, (n - 2) / g)
                - 4 * psi(0, (n - 1) / g)
                + psi(0, n / g)
            )
        )
    ) / g


def mean_hypo_o6(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 6)
    # This is the next obvious thing: coefficients are from g'th row of Pascal's triangle.
    assert g >= 1, "g must be at least 1"
    # print(f'mean_hypo_o5: n={n}, k={k}, g={g}')
    return (
        6
        * (
            psi(0, k + (n - 5) / g)
            - 5 * psi(0, k + (n - 4) / g)
            + 10 * psi(0, k + (n - 3) / g)
            - 10 * psi(0, k + (n - 2) / g)
            + 5 * psi(0, k + (n - 1) / g)
            - psi(0, k + n / g)
            - (
                psi(0, (n - 5) / g)
                - 5 * psi(0, (n - 4) / g)
                + 10 * psi(0, (n - 3) / g)
                - 10 * psi(0, (n - 2) / g)
                + 5 * psi(0, (n - 1) / g)
                - psi(0, n / g)
            )
        )
    ) / g


def mean_hypo_o7(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 7)
    assert g >= 1, "g must be at least 1"
    return (
        7
        * (
            psi(0, k + (n - 6) / g)
            - 6 * psi(0, k + (n - 5) / g)
            + 15 * psi(0, k + (n - 4) / g)
            - 20 * psi(0, k + (n - 3) / g)
            + 15 * psi(0, k + (n - 2) / g)
            - 6 * psi(0, k + (n - 1) / g)
            + psi(0, k + n / g)
            - (
                psi(0, (n - 6) / g)
                - 6 * psi(0, (n - 5) / g)
                + 15 * psi(0, (n - 4) / g)
                - 20 * psi(0, (n - 3) / g)
                + 15 * psi(0, (n - 2) / g)
                - 6 * psi(0, (n - 1) / g)
                + psi(0, n / g)
            )
        )
    ) / g


def mean_hypo_o8(n: int, k: int, g: int) -> float:
    # sum_{i=0}^{k-1} 1 / binomial(n + g*i, 8)
    assert g >= 1, "g must be at least 1"
    return (
        8
        * (
            psi(0, k + (n - 7) / g)
            - 7 * psi(0, k + (n - 6) / g)
            + 21 * psi(0, k + (n - 5) / g)
            - 35 * psi(0, k + (n - 4) / g)
            + 35 * psi(0, k + (n - 3) / g)
            - 21 * psi(0, k + (n - 2) / g)
            + 7 * psi(0, k + (n - 1) / g)
            - psi(0, k + n / g)
            - (
                psi(0, (n - 7) / g)
                - 7 * psi(0, (n - 6) / g)
                + 21 * psi(0, (n - 5) / g)
                - 35 * psi(0, (n - 4) / g)
                + 35 * psi(0, (n - 3) / g)
                - 21 * psi(0, (n - 2) / g)
                + 7 * psi(0, (n - 1) / g)
                - psi(0, n / g)
            )
        )
    ) / g


from functools import wraps
from typing import Callable, Any, TypeVar

T = TypeVar("T")


def adaptive_precision(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator that adaptively increases precision until convergence. Although this can
    be used to decorate any function, it should only be used if some of the arguments/
    variables in that function are mpmath types, which might benefit from increased precision.
    The function is run with increasingly larger precision contexts, and the results
    are compared for convergence, to adaptively test how much precision is required.
    It stops when maximum precision of 180 bits is reached, or two consecutive results
    are within a relative tolerance of 1e-15.

    The decorated function is responsible for handling its own argument conversions
    and working with mpmath types as needed. This decorator only manages the
    precision context and convergence testing.

    This decorator will:
    1. Calculate the function at increasing precisions: 53 (default), 60, 90, 120, ..., 300 bits
    2. Stop early if consecutive results converge within relative tolerance of 1e-15
    3. Return the higher precision value when convergence is achieved

    The reason we choose multiples of 30 after the default 53 is that Python ints are represented
    by 30 bit chunks, and mpmath uses Python ints under the hood to represent the bits of its
    mantissa.

    Args:
        func: Function that returns a numeric result suitable for convergence testing

    Returns:
        Decorated function that adaptively computes the result with sufficient precision.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Precision sequence in bits
        precisions: list[int] = [270]  # [53] + list(range(60, 1200, 30))

        prev_result: T | None = None

        for prec in precisions:
            # print(f"trying precision {prec=}")
            with mp.workprec(prec):
                # Calculate function value
                result: T = func(*args, **kwargs)

                # Check convergence if we have a previous result and both are numeric
                if prev_result is not None:
                    # Attempt convergence check for numeric types
                    if result != 0.0 and result == prev_result:
                        return result
                    abs_prev = abs(prev_result)  # type: ignore
                    if abs_prev > 0:
                        min_result_abs = min(abs(result), abs(prev_result))  # type: ignore
                        if min_result_abs == 0:
                            min_result_abs = max(abs(result), abs(prev_result))  # type: ignore
                        rel_diff = abs(prev_result - result) / min_result_abs  # type: ignore
                        if rel_diff < 1e-15 and result != 0:
                            return result  # Return higher precision value

                # print(f"  result at {prec=:3}: {result}")
                prev_result = result

        # If we reach here, return the final result at highest precision
        return prev_result  # type: ignore

    return wrapper



###################################################################
# more direct ways to compute mean and variance of hypoexponentials 
# used to verify faster ways give same answer

import scipy


def reciprocals(n: int, k: int, o: int, g: int) -> npt.NDArray[np.float64]:
    top_values = np.arange(k) * g + n
    binomial_values = scipy.special.comb(top_values, o)
    return 1.0 / binomial_values


def reciprocals_gamma(
    n: int, k: int, o: int, g: int, square: bool
) -> npt.NDArray[np.float64]:
    indices = np.arange(k) * g
    log_binom = gammaln(n + indices + 1) - gammaln(o + 1) - gammaln(n + indices - o + 1)
    coef = -2 if square else -1
    return np.exp(coef * log_binom)


def mean_direct_np_gamma(n: int, k: int, o: int, g: int) -> float:
    return float(np.sum(reciprocals_gamma(n, k, o, g, False)))


def var_direct_np_gamma(n: int, k: int, o: int, g: int) -> float:
    return float(np.sum(reciprocals_gamma(n, k, o, g, True)) )


def mean_direct_np(n: int, k: int, o: int, g: int) -> float:
    return float(np.sum(reciprocals(n, k, o, g)))


def var_direct_np(n: int, k: int, o: int, g: int) -> float:
    return float(np.sum(reciprocals(n, k, o, g) ** 2))


def mean_direct(n: int, k: int, o: int, g: int) -> float:
    terms = []
    for i in range(int(k)):
        terms.append(1 / math.comb(n + i * g, o))
    s = mp.fsum(terms)
    return s


def var_direct(n: int, k: int, o: int, g: int) -> float:
    terms = []
    for i in range(int(k)):
        terms.append(1 / math.comb(n + i * g, o))
    s = mp.fsum(terms, squared=True)
    return s


def generate_running_time_data():
    import numpy as np
    import math
    import importlib
    import gamma

    importlib.reload(gamma)
    from tqdm.auto import tqdm
    from tqdm.contrib import itertools
    import json

    rng = np.random.default_rng(42)
    running_time_data_sample_hypo = []
    running_time_data_sample_gamma = []
    trials_each_input = 5
    for g, o, exponent, k_multiplier in itertools.product(
        # range(1, 2),
        # range(1, 2),
        # range(2, 6, 2),
        range(1, 6),
        range(1, 6),
        range(14, 1, -2),
        [5, 1, 1 / 5],
    ):
        n = 10**exponent
        sqrt_n = round(math.sqrt(n))
        k = int(sqrt_n * k_multiplier)
        for trial in range(1, trials_each_input + 1):
            if trial == 1:
                print(
                    f'{"*"*80}\n* {g=}, {o=}, n = 10^{exponent}, k = {k:,} (sqrt(n) * {k_multiplier})'
                )
            print(f"| trial {trial}:", end=" ")

            print(f"hypo:  ", end="")
            mean_time_sample_hypo = gamma.smart_timeit(
                lambda: gamma.sample_hypo(rng, n, k, o, g)
            )
            print(f"|          gamma: ", end="")
            mean_time_sample_gamma = gamma.smart_timeit(
                lambda: gamma.sample_gamma_matching_hypo(rng, n, k, o, g)
            )
            print(
                f"|          hypo/gamma ratio: {mean_time_sample_hypo / mean_time_sample_gamma}"
            )

            running_time_data_sample_hypo.append((n, k, o, g, mean_time_sample_hypo))
            running_time_data_sample_gamma.append((n, k, o, g, mean_time_sample_gamma))

            # output every iteration so we can check out the data even as it's running
            running_time_data = {
                "sample_hypo": running_time_data_sample_hypo,
                "sample_gamma": running_time_data_sample_gamma,
            }
            with open("running_time_data_vary_k.json", "w") as f:
                json.dump(running_time_data, f, indent=4)


def num_with_prefix(
    prefix: Sequence[float], sequences: Sequence[Sequence[float]]
) -> int:
    """
    Returns number of sequences in `sequences` that start with `prefix`.
    """
    num_with_prefix = 0
    for sequence in sequences:
        is_prefix = True
        for i, value in enumerate(prefix):
            if i >= len(sequence) or sequence[i] != value:
                is_prefix = False
                break
        if is_prefix:
            num_with_prefix += 1
    return num_with_prefix


def generate_running_time_data_mean_var(
    fn: str,
    f1: Callable[[int, int, int, int], float],
    f1name: str,
    f2: Callable[[int, int, int, int], float],
    f2name: str,
) -> None:
    import numpy as np
    import math
    from tqdm.contrib import itertools
    import json
    import os

    name_length_max = max(len(f1name), len(f2name))

    if os.path.exists(fn):
        with open(fn, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {
            f1name: [],
            f2name: [],
        }
    running_time_data_1 = json_data[f1name]
    running_time_data_2 = json_data[f2name]

    rng = np.random.default_rng(42)
    trials_each_input = 5
    for g, o, exponent, k_multiplier in itertools.product(
        # range(1, 2),
        # range(1, 2),
        # range(2, 6, 2),
        range(1, 6),
        range(1, 6),
        range(2, 15, 2),
        # range(14, 1, -2),
        # [5, 1, 1/5],
        [1 / 2, 1, 2],
    ):
        n = 10**exponent
        sqrt_n = round(math.sqrt(n))
        k = int(sqrt_n * k_multiplier)
        for trial in range(1, trials_each_input + 1):
            num_with_prefix_1 = num_with_prefix([n, k, o, g], running_time_data_1)
            num_with_prefix_2 = num_with_prefix([n, k, o, g], running_time_data_2)

            if trial == 1:
                print(
                    f'{"*"*80}\n* {g=}, {o=}, n = 10^{exponent}, k = {k:,} (sqrt(n) * {k_multiplier})'
                )

            need1 = num_with_prefix_1 < trials_each_input
            need2 = num_with_prefix_2 < trials_each_input

            if not need1 and not need2:
                print(
                    f"skipping since we already have {trials_each_input} trials for {n=}, {k=}, {o=}, {g=}"
                )
                break

            print(f"| trial {trial}:", end=" ")

            if need1:
                print(f"{f1name:{name_length_max}}: ", end="")
                mean_time_1 = smart_timeit(lambda: f1(n, k, o, g))
                running_time_data_1.append((n, k, o, g, mean_time_1))

            if need2:
                print(f"|          {f2name:{name_length_max}}: ", end="")
                mean_time_2 = smart_timeit(lambda: f2(n, k, o, g))
                running_time_data_2.append((n, k, o, g, mean_time_2))

            if need1 and need2:
                print(f"|          log_2({f1name}/{f2name}): {math.log(mean_time_1) - math.log(mean_time_2):.1f}")  # type: ignore

            # output every iteration where we actually generated new trials,
            # so we can check out the data even as it's running
            running_time_data = {
                f1name: running_time_data_1,
                f2name: running_time_data_2,
            }
            with open(fn, "w") as f:
                json.dump(running_time_data, f, indent=4)


def generate_running_time_data_sampling(
    rng: np.random.Generator,
    fn: str,
    f1: Callable[[np.random.Generator, int, int, int, int], float],
    f1name: str,
    f2: Callable[[np.random.Generator, int, int, int, int], float],
    f2name: str,
) -> None:
    import numpy as np
    import math
    from tqdm.contrib import itertools
    import json
    import os

    name_length_max = max(len(f1name), len(f2name))

    if os.path.exists(fn):
        with open(fn, "r") as f:
            json_data = json.load(f)
    else:
        json_data = {
            f1name: [],
            f2name: [],
        }
    running_time_data_1 = json_data[f1name]
    running_time_data_2 = json_data[f2name]

    trials_each_input = 5
    for g, o, exponent, k_multiplier in itertools.product(
        # range(1, 2),
        # range(1, 2),
        # range(2, 6, 2),
        range(1, 6),
        range(1, 6),
        range(2, 15, 2),
        # range(14, 1, -2),
        # [5, 1, 1/5],
        [1 / 2, 1, 2],
    ):
        n = 10**exponent
        sqrt_n = round(math.sqrt(n))
        k = int(sqrt_n * k_multiplier)
        for trial in range(1, trials_each_input + 1):
            num_with_prefix_1 = num_with_prefix([n, k, o, g], running_time_data_1)
            num_with_prefix_2 = num_with_prefix([n, k, o, g], running_time_data_2)

            if trial == 1:
                print(
                    f'{"*"*80}\n* {g=}, {o=}, n = 10^{exponent}, k = {k:,} (sqrt(n) * {k_multiplier})'
                )

            need1 = num_with_prefix_1 < trials_each_input
            need2 = num_with_prefix_2 < trials_each_input

            if not need1 and not need2:
                print(
                    f"skipping since we already have {trials_each_input} trials for {n=}, {k=}, {o=}, {g=}"
                )
                break

            print(f"| trial {trial}:", end=" ")

            if need1:
                print(f"{f1name:{name_length_max}}: ", end="")
                mean_time_1 = smart_timeit(lambda: f1(rng, n, k, o, g))
                running_time_data_1.append((n, k, o, g, mean_time_1))

            if need2:
                print(f"|          {f2name:{name_length_max}}: ", end="")
                mean_time_2 = smart_timeit(lambda: f2(rng, n, k, o, g))
                running_time_data_2.append((n, k, o, g, mean_time_2))

            if need1 and need2:
                print(f"|          log_2({f1name}/{f2name}): {math.log(mean_time_1) - math.log(mean_time_2):.1f}")  # type: ignore

            # output every iteration where we actually generated new trials,
            # so we can check out the data even as it's running
            running_time_data = {
                f1name: running_time_data_1,
                f2name: running_time_data_2,
            }
            with open(fn, "w") as f:
                json.dump(running_time_data, f, indent=4)

# hard-won empirical measurements of running times
# These are very difficult to fit with any precision,
# but empirically these almost always seem with 2x or 3x
# of the actual running time (the latter of which has very high variance).


def predicted_mean_direct_np(n: int, k: int, o: int, g: int) -> float:
    offset = 10**-5
    if k < 10**3:
        return 10**-5 + offset
    return k * 10**-8 + offset


def predicted_mean_hypo(n: int, k: int, o: int, g: int) -> float:
    n_offset = o * max(0.0, 1.2 * 10**-3 - n * 0.5 * 10**-5 / g) / 10
    coef = 0.9 * 10**-5
    if o == 1:
        coef_offset = 0.5
    elif o == 2:
        coef_offset = 3.0
    else:
        coef_offset = 12.0
    return coef * o**2 + n_offset + coef_offset * 10**-4 / o**2


def predicted_var_direct_np(n: int, k: int, o: int, g: int) -> float:
    if k <= 10**3:
        return 5 * 10**-5
    return k * 10**-7 / 2.7


def predicted_var_hypo(n: int, k: int, o: int, g: int) -> float:
    n_offset = o * max(0.0, 1.2 * 10**-3 - n * 0.5 * 10**-5 / g)
    coef = 0.000035
    return coef * o**3 + n_offset + 3 * 10**-4 / o**2


if __name__ == "__main__":
    # main()

    generate_running_time_data_sampling(
        np.random.default_rng(42),
        "running_time_data_sampling.json",
        sample_hypo,
        "sample_hypo",
        sample_gamma_matching_hypo,
        "sample_gamma_matching_hypo",
    )

    # generate_running_time_data_mean_var(
    #     "running_time_data_compute_mean.json",
    #     mean_direct_np,
    #     "mean_direct_np",
    #     mean_hypo,
    #     "mean_hypo",
    # )

    # generate_running_time_data_mean_var(
    #     'running_time_data_compute_var.json',
    #     var_direct_np, 'var_direct_np',
    #     var_hypo, 'var_hypo',
    # )
