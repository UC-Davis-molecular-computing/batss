# Utility file for using mpmath to run sample_coll, to avoid floating point error,
# because mpmath is the only arbitrary-precision library we can find that implements lgamma.

from mpmath import mp, mpf
import time

def sample_coll(n: int, r: int, o: int, g: int, u: float) -> int:
    assert o > 0
    assert g >= 0
    assert 0 <= r <= n
    assert 0 <= u <= 1
    if r == n:
        return 0
    # To be safe, we'll just put most things as mpm floats to start out.
    # Leaving o as an int since it's used in ranges.
    # Might be better to update this and do something clever, since some of the arithmetic
    # can certainly be done without conversion, so there could be some time savings.
    n = mpf(n)
    r = mpf(r)
    g = mpf(g)
    u = mpf(u)


    # If more decimal places is much less efficient, we could do something smart here and
    # figure out how much precision we need before computing. 
    mp.dps = 30
    ln_u = mp.ln(u)
    lhs = mp.loggamma(n - r + 1) - ln_u

    if g > 0:
        for j in range(o):
            num_static_terms = mp.ceil((n - g - j) / g)
            lhs += num_static_terms * mp.ln(g)
            lhs += mp.loggamma((n - j) / g)
            lhs -= mp.loggamma((n - (num_static_terms * g) - j) / g)
    else:
        pass

    t_lo: int = 0
    t_hi: int = 1 + int((n - r) / o)

    while t_lo < t_hi - 1:
        t_mid: int = (t_lo + t_hi) // 2
        rhs = mp.loggamma(n - r - (t_mid * o) + 1)
        if g > 0:
            for j in range(o):
                num_dynamic_terms = mp.ceil((n + g * (t_mid - 1) - j) / g)
                rhs += num_dynamic_terms * mp.ln(g)
                rhs += mp.loggamma((n + (g * t_mid) - j) / g)
                rhs -= mp.loggamma((n + g * (t_mid - num_dynamic_terms) - j) / g)
        else:
            for j in range(o):
                rhs += t_mid * mp.ln(n - j)
        if lhs < rhs:
            t_hi = t_mid
        else:
            t_lo = t_mid
            
    return t_lo