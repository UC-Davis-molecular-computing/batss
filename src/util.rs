use num_bigint::BigInt;
use rand::rngs::SmallRng;
use rand::Rng;

use rand_distr::{Distribution, Hypergeometric, StandardUniform};
// use rug::Float;

#[allow(unused_imports)]
use crate::flame;

pub fn binomial_as_f64(n: u64, k: u64) -> f64 {
    // We're hand-implementing this as a loop because statrs's algorithm winds up with floating
    // point errors and naive algorithms that multiply ints will overflow. And k is going to
    // be the order of the CRN, so it's very few iterations.
    let mut out = n as f64;
    for i in 1..k {
        out *= (n - i) as f64;
        out /= (i + 1) as f64;
    }
    return out;
}

pub fn ln_gamma(x: f64) -> f64 {
    // println!("ue version called");
    ln_gamma_special(x)
}

pub fn ln_factorial(x: u64) -> f64 {
    // println!("non ue version called");
    ln_factorial_manual(x)
}

pub fn hypergeometric_sample(popsize: u64, good: u64, draws: u64, rng: &mut SmallRng) -> u64 {
    // println!("hypergeometric_sample_manual");
    hypergeometric_sample_manual(popsize, good, draws, rng)
}

pub fn multinomial_sample(n: u64, pix: &Vec<f64>, result: &mut [u64], rng: &mut SmallRng) {
    multinomial_sample_manual(n, pix, result, rng);
}

/////////////////////////////////////////////////////////////////////////////////
// ln_gamma

// As always, if the statrs crate implements something, it's slower than alternatives.
// Here we use the special crate instead, which is about 40% faster.
// const ln_gamma: fn(f64) -> f64 = statrs::function::gamma::ln_gamma;
// UPDATE: The custom implementation below, adapted from R, is slight faster, maybe 10-20%,
// than the implementation in the special crate.

pub fn ln_gamma_special(x: f64) -> f64 {
    // special implements these as methods that can be called on f64's if we use special::Gamma,
    // but that gives a Rust warning about possibly the method name being used in Rust in the future.
    // We can call the method directly like this, but for some reason it returns a pair,
    // and the output of ln_gamma is the first element.
    special::Gamma::ln_gamma(x).0
}

// adapted from C source for R standard library
// https://github.com/SurajGupta/r-source/blob/a28e609e72ed7c47f6ddfbb86c85279a0750f0b7/src/nmath/lgamma.c#L44
// Since we only call it on non-negative integers, we can simplify some of the code that was handling
// negative integers and NaN values.
const M_LN_SQRT_2PI: f64 = 0.918938533204672741780329736406; // log(sqrt(2*pi)) == log(2*pi)/2
const M_LN_SQRT_2PI_128: f128 = 0.918938533204672741780329736406; // log(sqrt(2*pi)) == log(2*pi)/2

const MAX_NEEDED_PRECISION: f128 = 1.0e-30;
// See OEIS A046969, A046968.
const MAX_BERNOULLI_TERMS: usize = 4;
const BERNOULLI_COEFFS: [f128; MAX_BERNOULLI_TERMS] = [
    0.0833333333333333333333333333333292,
    -0.00277777777777777777777777777777777,
    0.000793650793650793650793650793650791,
    -0.000595238095238095238095238095238093,
];

const XMAX_LN_GAMMA: f64 = 2.5327372760800758e+305;

pub fn ln_gamma_manual(x: f64) -> f64 {
    let ret: f64;
    if x <= 10.0 {
        return special::Gamma::gamma(x as f64).ln();
    }
    if x > XMAX_LN_GAMMA {
        return f64::INFINITY;
    }
    let lnx = x.ln();
    // In C code this only happens if x > 0, but since our x is usize,
    // we do this unconditionally.
    if x > 1e17 {
        ret = x * (lnx - 1.0);
    } else if x > 4934720.0 {
        ret = M_LN_SQRT_2PI + (x - 0.5) * lnx - x;
    } else {
        ret = M_LN_SQRT_2PI + (x - 0.5) * lnx - x + lgammacor(x);
    }
    ret
}

pub fn ln_gamma_manual_high_precision(x: f128) -> f128 {
    if x > MIN_LARGE_LN_GAMMA_INPUT {
        return ln_gamma_manual_high_precision_large(x);
    } else {
        // If molecular count is below this value, we can get enough precision with f64.
        // println!(
        //     "x is {:?}, as f64 is {:?}, output is {:?} turning into {:?}.",
        //     x,
        //     x as f64,
        //     ln_gamma(x as f64),
        //     ln_gamma(x as f64) as f128
        // );
        return ln_gamma(x as f64) as f128;
    }
}

const MIN_LARGE_LN_GAMMA_INPUT: f128 = 100_000_000_000.0;
pub fn ln_gamma_manual_high_precision_large(x: f128) -> f128 {
    assert!(
        x > MIN_LARGE_LN_GAMMA_INPUT,
        "high precision ln_gamma requires large input."
    );
    // flame::start("manual_high_precision");

    // flame::start("ln call");
    let lnx = ln_f128(x);
    // flame::end("ln call");

    // flame::start("f64 ln call");
    // let _lnf64 = (x as f64).ln();
    // flame::end("f64 ln call");

    // let a64: f64 = 123.456;
    // let b64: f64 = 789.124;
    // flame::start("f64 random arithmetic");
    // let _ = a64 + b64;
    // let _ = a64 - b64;
    // let _ = a64 * b64;
    // let _ = a64 / b64;
    // flame::end("f64 random arithmetic");

    // let a128: f128 = 123.456;
    // let b128: f128 = 789.124;
    // flame::start("f128 random arithmetic");
    // let _ = a128 + b128;
    // let _ = a128 - b128;
    // let _ = a128 * b128;
    // let _ = a128 / b128;
    // flame::end("f128 random arithmetic");

    // flame::start("initial computation");
    let mut ret = M_LN_SQRT_2PI_128 + (x - 0.5) * lnx - x;
    // flame::end("initial computation");
    // Stirling's series converges quickly when x >> 1.
    for i in 0..MAX_BERNOULLI_TERMS {
        // if i == 0 {
        //     flame::start("iteration 1");
        // } else {
        //     flame::start("iteration 2+");
        // }
        let delta = BERNOULLI_COEFFS[i] / x.powi((2 * i + 1) as i32);
        ret += delta;
        // if i == 0 {
        //     flame::end("iteration 1");
        // } else {
        //     flame::end("iteration 2+");
        // }
        if delta < MAX_NEEDED_PRECISION {
            break;
        }
    }
    // flame::end("manual_high_precision");
    ret
}

const MAX_PRECOMPUTED_DENOMINATOR: usize = 10;
const PRECOMPUTED_SMALL_RATIONAL_LN_GAMMAS: [[f128; MAX_PRECOMPUTED_DENOMINATOR];
    MAX_PRECOMPUTED_DENOMINATOR] = [
    [
        0.0,
        0.572364942924700087071713675677,
        0.985420646927767069187174036978,
        1.28802252469807745737061044022,
        1.52406382243078452488105649393,
        1.71673343507824046052784630959,
        1.87916927159583583645595640935,
        2.01941835755379634532029052117,
        2.1427318003766931048804078849,
        2.25271265173420595986970164637,
    ],
    [
        0.0,
        0.0,
        0.303150275147523568675862817372,
        0.572364942924700087071713675677,
        0.796677817701783766544735962392,
        0.985420646927767069187174036978,
        1.14712149837668758048594413508,
        1.28802252469807745737061044022,
        1.41259045980878492068007539775,
        1.52406382243078452488105649393,
    ],
    [
        0.693147180559945309417232121458,
        -0.120782237635245222345518445782,
        0.0,
        0.203280951431295371481432971862,
        0.398233858069234899616854220401,
        0.572364942924700087071713675677,
        0.726345819754632570220435629001,
        0.863073982270647462405089094134,
        0.985420646927767069187174036978,
        1.09579799481807552167716814237,
    ],
    [
        1.79175946922805500081247735838,
        0.0,
        -0.113191641740342622208071199945,
        0.0,
        0.152059678399837588778292602291,
        0.303150275147523568675862817372,
        0.443775813036245464728024884701,
        0.572364942924700087071713675677,
        0.689587614192281125729376166555,
        0.796677817701783766544735962392,
    ],
    [
        3.1780538303479456196469416013,
        0.284682870472919159632494669683,
        -0.102314832960640813302150298092,
        -0.0982718364218131614638538026966,
        0.0,
        0.121143631331105023032813163223,
        0.243724444697223435187555499451,
        0.360829495488940181184957685823,
        0.470451103123104811872368298075,
        0.572364942924700087071713675677,
    ],
    [
        4.78749174278204599424770093452,
        0.693147180559945309417232121458,
        0.0,
        -0.120782237635245222345518445782,
        -0.0853740900033158497197028392999,
        0.0,
        0.100539277239754911051385489132,
        0.203280951431295371481432971862,
        0.303150275147523568675862817372,
        0.398233858069234899616854220401,
    ],
    [
        6.5792512120101009950601782929,
        1.20097360234707422481602188145,
        0.174490430711438305231147806049,
        -0.0844011210204855559577860341314,
        -0.119612914172371298638791249376,
        -0.0750260341498145402846310487928,
        0.0,
        0.0858587072253343235023655837695,
        0.174080346879489937370590331159,
        0.260867246531666514385732417017,
    ],
    [
        8.52516136106541430016553103635,
        1.79175946922805500081247735838,
        0.408510790805349869903363798211,
        0.0,
        -0.112591765696755783588659875903,
        -0.113191641740342622208071199945,
        -0.0667408774594774686493963340981,
        0.0,
        0.0748837305136277044515377215345,
        0.152059678399837588778292602291,
    ],
    [
        10.6046029027452502284172274007,
        2.45373657084244222050414250344,
        0.693147180559945309417232121458,
        0.124871714892396594302441287613,
        -0.0710838729143721669880024880193,
        -0.120782237635245222345518445782,
        -0.105641470118680415202176486904,
        -0.0600231841260395829314058432074,
        0.0,
        0.0663762397347429711887167398671,
    ],
    [
        12.8018274800814696112077178746,
        3.1780538303479456196469416013,
        1.02178829109864191894125531257,
        0.284682870472919159632494669683,
        0.0,
        -0.102314832960640813302150298092,
        -0.120952040632571043489671877519,
        -0.0982718364218131614638538026966,
        -0.0544927769595262779100825889486,
        0.0,
    ],
];

pub fn ln_gamma_small_rational(num: usize, den: usize) -> f128 {
    // flame::start("small_rational");
    assert!(
        den <= MAX_PRECOMPUTED_DENOMINATOR,
        "For now, we're assuming generativity is less than 10."
    );
    assert!(
        num <= den,
        "ln_gamma_small_rational should only be called on values between 0 and 1."
    );
    // flame::end("small_rational");
    PRECOMPUTED_SMALL_RATIONAL_LN_GAMMAS[num - 1][den - 1]
}

// pub fn ln_gamma_small_rational_rug(prec: u32, num: usize, den: usize) -> Float {
//     // flame::start("small_rational");
//     assert!(
//         den <= MAX_PRECOMPUTED_DENOMINATOR,
//         "For now, we're assuming generativity is less than 10."
//     );
//     assert!(
//         num <= den,
//         "ln_gamma_small_rational should only be called on values between 0 and 1."
//     );
//     // flame::end("small_rational");
//     Float::with_val(
//         prec,
//         PRECOMPUTED_SMALL_RATIONAL_LN_GAMMAS[num - 1][den - 1] as f64,
//     )
// }

const LN2: f128 = 0.693147180559945309417232121458179; /* 3fe62e42 fee00000 */
// const LN2_LO: f128 = 1.90821492927058770002e-10; /* 3dea39ef 35793c76 */
// Thanks to Danny Hermes for publishing a script that could be easily tweaked to compute
// enough of these values at high precision: https://gist.github.com/dhermes/105da2a3c9861c90ea39
const LG1: f128 =
    0.66666666666666666666666666875815545229093281985458066108081113465975841990625437013082687;
const LG2: f128 =
    0.39999999999999999999999440533548502210613316290926048049503731640033630216780851704019313;
const LG3: f128 =
    0.28571428571428571429094613297509361650334345377104797971745365245556418301332109522197954;
const LG4: f128 =
    0.22222222222222221975826738908555894301120591697217296006666261633589966652891750933338693;
const LG5: f128 =
    0.18181818181818250245686109265789351993548969174418157011902382456177825291397840701706558;
const LG6: f128 =
    0.1538461538460317890154588944351663231610679351082742582556706529319135758482086718167677;
const LG7: f128 =
    0.13333333334802997389443791002138961200581615283256823691072831395861433250848751297250711;
const LG8: f128 =
    0.11764705759536556953993447310433762572717017723371393550392034866773259858110658340392759;
const LG9: f128 =
    0.10526322994030205329396160427129955291121798214743001999261352557788817275781538560620414;
const LG10: f128 =
    0.095235140504843371935608507213095115484437687391731985549670821305657519885864575021233968;
const LG11: f128 =
    0.087039345837935769603583774487860193007973401003664029391266891327754138987831114010227144;
const LG12: f128 =
    0.078493794248594670268165101799026327271815909183231905843461379742199434030144534975373602;
const LG13: f128 =
    0.089908734431217458910014676930010764603174233285125307525098603452056911758204218409684487;
const SQRT2: f128 = 1.41421356237309504880168872420977;

// f128 natural log. Mostly copied from https://github.com/rust-lang/libm/blob/master/libm/src/math/log.rs
pub fn ln_f128(x: f128) -> f128 {
    let print: bool = x == 156951994071_u64 as f128;
    // flame::start("Part 1");
    // assert!(x >= 1.0, "ln_f128 assumes its input is at least 1.");
    // Get the exponent from the f128. It has one sign bit followed by 15 exponent bits.
    // https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format
    // flame::start("bitshift");
    let exponent_raw = (x.to_bits() << 1) >> 113;
    // TODO: try making exponent 16 or 32 bits.
    // flame::end("bitshift");
    // flame::start("cast_signed");
    let exponent = exponent_raw.cast_signed() - 0x3FFF;
    // flame::end("cast_signed");
    // We don't expect to call ln on anything less than 1.
    // assert!(
    //     exponent >= 0,
    //     "exponent should be nonnegative, since we assume x >= 1.0"
    // );

    // flame::end("Part 1");
    // flame::start("Part 2");
    // Get a value betwen 1 and 2.
    // TODO: rename/add extra variables to make it easier to go back and understand this in comparison
    // with the document that explains it well.
    // Need to use u64 for the base so that the result fits for large population sizes.
    let mut normalized_x = x / 2u64.pow(exponent as u32) as f128;
    let mut k = exponent;
    if normalized_x > SQRT2 {
        normalized_x *= 0.5;
        k += 1;
    }
    let k_times_ln_2 = k as f128 * LN2;
    // flame::end("Part 2");
    // flame::start("Part 3");

    let f: f128 = normalized_x - 1.0;
    let hfsq: f128 = 0.5 * f * f;
    let s: f128 = f / (2.0 + f);
    let z: f128 = s * s;
    let w: f128 = z * z;
    // flame::end("Part 3");
    // flame::start("Part 4");
    let t1: f128 = w * (LG2 + w * (LG4 + w * (LG6 + w * (LG8 + w * (LG10 + w * LG12)))));
    let t2: f128 =
        z * (LG1 + w * (LG3 + w * (LG5 + w * (LG7 + w * (LG9 + w * (LG11 + w * LG13))))));
    let r: f128 = t2 + t1;
    // flame::end("Part 4");
    let out = s * (hfsq + r) - hfsq + f + k_times_ln_2;
    if print {
        println!(
            "Out, normx, f, z, w, exponent: {:?}, {:?}, {:?}, {:?}, {:?}, {:?}",
            f128_to_decimal(out),
            f128_to_decimal(normalized_x),
            f128_to_decimal(f),
            f128_to_decimal(z),
            f128_to_decimal(w),
            exponent
        );
    }
    out
}

const ALGMCS: [f64; 15] = [
    0.1666389480451863247205729650822,
    -0.1384948176067563840732986059135e-4,
    0.9810825646924729426157171547487e-8,
    -0.1809129475572494194263306266719e-10,
    0.6221098041892605227126015543416e-13,
    -0.3399615005417721944303330599666e-15,
    0.2683181998482698748957538846666e-17,
    -0.2868042435334643284144622399999e-19,
    0.3962837061046434803679306666666e-21,
    -0.6831888753985766870111999999999e-23,
    0.1429227355942498147573333333333e-24,
    -0.3547598158101070547199999999999e-26,
    0.1025680058010470912000000000000e-27,
    -0.3401102254316748799999999999999e-29,
    0.1276642195630062933333333333333e-30,
];
const NALGM: usize = 5;
const XBIG_LGAMMACOR: f64 = 94906265.62425156;
const XMAX_LGAMMACOR: f64 = 3.745194030963158e306;

fn lgammacor(x: f64) -> f64 {
    assert!(x >= 10.0);
    assert!(x < XMAX_LGAMMACOR);
    if x < 10.0 {
        return 0.0;
    } else if x >= XBIG_LGAMMACOR {
        panic!("x = {x} must be less than {XBIG_LGAMMACOR}");
    } else if x < XBIG_LGAMMACOR {
        let tmp = 10.0 / x;
        return chebyshev_eval(tmp * tmp * 2.0 - 1.0) / x;
    }
    1.0 / (x * 12.0)
}

fn chebyshev_eval(x: f64) -> f64 {
    let n = NALGM;
    let a = ALGMCS;

    if n < 1 || n > 1000 {
        panic!("n must be between 1 and 1000, got {n}");
    }
    if x < -1.1 || x > 1.1 {
        panic!("x must be between -1.1 and 1.1, got {x}");
    }

    let twox = x * 2.0;
    let mut b0 = 0.0;
    let mut b1 = 0.0;
    let mut b2 = 0.0;

    for i in (1..=n).rev() {
        b2 = b1;
        b1 = b0;
        b0 = twox * b1 - b2 + a[n - i];
    }
    (b0 - b2) * 0.5
}

/////////////////////////////////////////////////////////////////////////////////
// log_factorial

// We precompuate log(k!) for k = 0, 1, ..., MAX_FACTORIAL-1
// technically MAX_FACTORIAL is the SIZE of array, not the max k for which we compute ln(k!),
// starting at ln(0!), so this goes up to ln((MAX_FACTORIAL-1)!).
// We use this cache because numpy does, out of paranoia, but in practice it's actually really not
// any faster than using the Stirling approximation, and the Stirling approximation is surprisingly
// accurate even for small values of k. I'm not sure why numpy uses this cache, maybe
// because Numerical Recipes in C recommends that: https://numerical.recipes/book.html
// since it defers to log_gamma for large, which might be slower than the Stirling approximation
// used for integer k in log_factorial.
const MAX_FACTORIAL: usize = 126;

pub fn create_log_fact_cache() -> [f64; MAX_FACTORIAL] {
    let mut cache = [0.0; MAX_FACTORIAL];

    let mut i: usize = 1;
    let mut ln_fact: f64 = 0.0;
    while i < MAX_FACTORIAL {
        // using the identity ln(k!) = ln((k-1)!) + ln(k)
        ln_fact += (i as f64).ln();
        cache[i] = ln_fact; // ln(0!) = 0 is a special case but we already populated with 0.0
        i += 1;
    }
    cache
}

use lazy_static::lazy_static;
lazy_static! {
    static ref LOGFACT: [f64; MAX_FACTORIAL] = create_log_fact_cache();
}

const HALFLN2PI: f64 = 0.9189385332046728;

pub fn ln_factorial_manual(k: u64) -> f64 {
    if k < MAX_FACTORIAL as u64 {
        let ret = LOGFACT[k as usize];
        return ret;
    }
    // Use the Stirling approximation for large x
    let k = k as f64;
    let ret =
        (k + 0.5) * k.ln() - k + (HALFLN2PI + (1.0 / k) * (1.0 / 12.0 - 1.0 / (360.0 * k * k)));
    ret
}

/////////////////////////////////////////////////////////////////////////////////
// hypergeometric_sample

pub fn hypergeometric_sample_statrs(
    popsize: usize,
    good: usize,
    draws: usize,
    rng: &mut SmallRng,
) -> Result<usize, String> {
    let hypergeometric_result = Hypergeometric::new(popsize as u64, good as u64, draws as u64);
    if hypergeometric_result.is_err() {
        return Err(String::from(format!(
            "Hypergeometric distribution creation error: {:?}",
            hypergeometric_result.unwrap_err(),
        )));
    }
    let hypergeometric = hypergeometric_result.unwrap();
    let h64: u64 = rng.sample(hypergeometric);
    let h = h64 as usize;
    Ok(h)
}

// adapted from numpy's implementation of the hypergeometric distribution (as of April 2025)
// https://github.com/numpy/numpy/blob/b76bb2329032809229e8a531ba3179c34b0a3f0a/numpy/random/src/distributions/random_hypergeometric.c#L246
pub fn hypergeometric_sample_manual(
    popsize: u64,
    good: u64,
    draws: u64,
    rng: &mut SmallRng,
) -> u64 {
    if good > popsize {
        panic!("good must be less than or equal to popsize");
    }
    if draws > popsize {
        panic!("sample must be less than or equal to popsize");
    }
    if draws >= 10 && draws <= popsize - 10 {
        hypergeometric_sample_hrua(popsize, good, draws, rng)
    } else {
        hypergeometric_sample_naive(popsize, good, draws, rng)
    }
}

fn hypergeometric_sample_naive(popsize: u64, good: u64, draws: u64, rng: &mut SmallRng) -> u64 {
    // This is the simpler naive implementation for small samples.
    // https://github.com/numpy/numpy/blob/b76bb2329032809229e8a531ba3179c34b0a3f0a/numpy/random/src/distributions/random_hypergeometric.c#L46
    // variable name translations from numpy source code:
    //   total --> popsize
    //   good  --> good
    //   bad   --> popsize - good
    //   draws --> sample
    let mut remaining_total = popsize;
    let mut remaining_good = good;
    let mut computed_sample = draws;
    if computed_sample > popsize / 2 {
        computed_sample = remaining_total - computed_sample;
    }
    while computed_sample > 0 && remaining_good > 0 && remaining_total > remaining_good {
        // random_range(0..=max) returns an integer in
        // [0, max] *inclusive*, so we decrement remaining_total before
        // passing it to random_range().
        remaining_total -= 1;
        if rng.random_range(0..=remaining_total) < remaining_good {
            // Selected a "good" one, so decrement remaining_good.
            remaining_good -= 1;
        }
        computed_sample -= 1;
    }
    if remaining_total == remaining_good {
        // Only "good" choices are left.
        remaining_good -= computed_sample
    }
    if draws > popsize / 2 {
        remaining_good
    } else {
        good - remaining_good
    }
}

// adapted from numpy's implementation of the hypergeometric_hrua algorithm
// https://github.com/numpy/numpy/blob/b76bb2329032809229e8a531ba3179c34b0a3f0a/numpy/random/src/distributions/random_hypergeometric.c#L119
const D1: f64 = 1.7155277699214135; // 2*sqrt(2/e)
const D2: f64 = 0.8989161620588988; // 3 - 2*sqrt(3/e)
pub fn hypergeometric_sample_hrua(popsize: u64, good: u64, sample: u64, rng: &mut SmallRng) -> u64 {
    let bad = popsize - good;
    let computed_sample = sample.min(popsize - sample);
    let mingoodbad = good.min(bad);
    let maxgoodbad = good.max(bad);

    /*
     *  Variables that do not match Stadlober (1989)
     *    Here               Stadlober
     *    ----------------   ---------
     *    mingoodbad            M
     *    popsize               N
     *    computed_sample       n
     */
    let p = mingoodbad as f64 / popsize as f64;
    let q = maxgoodbad as f64 / popsize as f64;

    let mu = computed_sample as f64 * p; // mean of the distribution

    let a = mu + 0.5;

    let var = ((popsize - computed_sample) as f64 * computed_sample as f64 * p * q
        / (popsize as f64 - 1.0)) as f64; // variance of the distribution

    let c = var.sqrt() + 0.5;

    /*
     *  h is 2*s_hat (See Stadlober's thesis (1989), Eq. (5.17); or
     *  Stadlober (1990), Eq. 8).  s_hat is the scale of the "table mountain"
     *  function that dominates the scaled hypergeometric PMF ("scaled" means
     *  normalized to have a maximum value of 1).
     */
    let h = D1 * c + D2;

    let m = ((computed_sample + 1) as f64 * (mingoodbad + 1) as f64 / (popsize + 2) as f64) as u64;

    let g = ln_factorial(m)
        + ln_factorial(mingoodbad - m)
        + ln_factorial(computed_sample - m)
        + ln_factorial(maxgoodbad + m - computed_sample);

    /*
     *  b is the upper bound for random samples:
     *  ... min(computed_sample, mingoodbad) + 1 is the length of the support.
     *  ... floor(a + 16*c) is 16 standard deviations beyond the mean.
     *
     *  The idea behind the second upper bound is that values that far out in
     *  the tail have negligible probabilities.
     *
     *  There is a comment in a previous version of this algorithm that says
     *      "16 for 16-decimal-digit precision in D1 and D2",
     *  but there is no documented justification for this value.  A lower value
     *  might work just as well, but I've kept the value 16 here.
     */
    let b = (computed_sample.min(mingoodbad) + 1).min((a + 16.0 * c).floor() as u64);

    let mut k: u64;
    loop {
        let u = rng.random::<f64>();
        let v = rng.random::<f64>(); // "U star" in Stadlober (1989)
        let x = a + h * (v - 0.5) / u;

        // fast rejection:
        if x < 0.0 || x >= b as f64 {
            continue;
        }

        k = x.floor() as u64;

        let gp = ln_factorial(k)
            + ln_factorial(mingoodbad - k)
            + ln_factorial(computed_sample - k)
            + ln_factorial(maxgoodbad + k - computed_sample);

        let t = g - gp;

        // fast acceptance:
        if (u * (4.0 - u) - 3.0) <= t {
            break;
        }

        // fast rejection:
        if u * (u - t) >= 1.0 {
            continue;
        }

        if 2.0 * u.ln() <= t {
            // acceptance
            break;
        }
    }

    if good > bad {
        k = computed_sample - k;
    }

    if computed_sample < sample {
        k = good - k;
    }

    k
}

/////////////////////////////////////////////////////////////////////////////////
// multinomial_sample

const SMALL_EXPECTED_FAILURE_THRESHOLD: f64 = 1.0 / 1_000.0;
pub fn binomial_sample(n: u64, p: f64, mut rng: &mut SmallRng) -> u64 {
    // let n: usize = 2517438726;
    // let p = 0.9999999999999994;
    // TODO: this is a terrible, terrible hack, to get around a bug in rand_distr::Binomial
    // that happens when called with the above n and p.
    let expected_failures = n as f64 * (1.0 - p);
    if expected_failures < SMALL_EXPECTED_FAILURE_THRESHOLD && n > core::i32::MAX as u64 {
        let mut out = n;
        while out > 0 {
            let val: f64 = rng.sample(StandardUniform);
            if val < expected_failures {
                out -= 1;
            } else {
                println!("{:?}, {:?}, {:?}", out, n, expected_failures);
                return out;
            }
        }
    }
    let binomial_distribution = rand_distr::Binomial::new(n as u64, p).unwrap();
    let sample = binomial_distribution.sample(&mut rng);
    sample
}

// port of numpy's multinomial sample to Rust, using rand_distr::Binomial as the underlying binomial sampler
// https://github.com/numpy/numpy/blob/4961a1414bba2222016f29a03dcf75e6034a13f7/numpy/random/src/distributions/distributions.c#L1726
pub fn multinomial_sample_manual(n: u64, pix: &Vec<f64>, result: &mut [u64], rng: &mut SmallRng) {
    assert_eq!(
        pix.len(),
        result.len(),
        "pix and result must have the same length in multinomial_sample_manual"
    );
    let mut remaining_p = 1.0;
    let d = pix.len(); // in numpy C code, pix is just a pointer so they need pix's array length too
    let mut dn = n;
    // Original Cython implementation zeroed out the result array initially, but
    // since we are overwriting the array, we only zero out the entries if we break out of the loop early.
    for j in 0..(d - 1) {
        result[j] = binomial_sample(dn, pix[j] / remaining_p, rng);
        dn -= result[j];
        if dn <= 0 {
            // erase old values in remainder of result array
            for i in (j + 1)..d {
                result[i] = 0;
            }
            break;
        }
        remaining_p -= pix[j];
    }
    result[d - 1] = dn;
}

pub fn f128_to_decimal(x: f128) -> String {
    // Handle special cases first
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x.is_sign_positive() {
            "inf".to_string()
        } else {
            "-inf".to_string()
        };
    }
    if x == 0.0 {
        return "0.0".to_string();
    }

    // Extract IEEE 754 binary128 components
    let bits = x.to_bits();
    let sign = (bits >> 127) != 0;
    let exponent = ((bits >> 112) & 0x7FFF) as i32;
    let mantissa = bits & 0x0000_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF_FFFF;

    // Handle sign
    let sign_str = if sign { "-" } else { "" };

    // IEEE 754 binary128 has:
    // - 1 sign bit
    // - 15 exponent bits (bias = 16383)
    // - 112 mantissa bits

    let bias = 16383;
    let actual_exponent = exponent - bias;

    // Build the significand (1.mantissa for normal numbers)
    let mut significand = BigInt::from(1_u128 << 112); // Implicit leading 1
    significand += BigInt::from(mantissa);

    // Calculate the actual value: significand * 2^(actual_exponent - 112)
    let power_of_2 = actual_exponent - 112;

    let mut result = significand;

    if power_of_2 >= 0 {
        // Multiply by 2^power_of_2
        result <<= power_of_2;
        format!("{}{}.0", sign_str, result)
    } else {
        // Divide by 2^(-power_of_2)
        // This is where we need to do decimal division
        let divisor = BigInt::from(1_u128) << (-power_of_2);

        // Perform long division to get decimal representation
        let quotient = &result / &divisor;
        let remainder = &result % &divisor;

        if remainder == BigInt::from(0) {
            format!("{}{}.0", sign_str, quotient)
        } else {
            // Calculate decimal places
            let mut decimal_digits = String::new();
            let mut current_remainder = remainder * 10;

            for _ in 0..50 {
                // Limit to 50 decimal places
                let digit: BigInt = &current_remainder / &divisor;
                decimal_digits.push_str(&digit.to_string());
                current_remainder = (&current_remainder % &divisor) * 10;

                if current_remainder == BigInt::from(0) {
                    break;
                }
            }

            format!("{}{}.{}", sign_str, quotient, decimal_digits)
        }
    }
}
