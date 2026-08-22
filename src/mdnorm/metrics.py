"""Performance statistics, and how much of one is selection.

Everything earlier in this library is about getting the data right. This
module is about the last step, where a correct dataset still produces a
misleading number — because the number was chosen.

A Sharpe ratio computed from one strategy is an estimate. The same Sharpe
ratio, reported after trying two hundred parameter sets and keeping the best,
is a maximum, and the maximum of two hundred draws from noise is not small.
That is the fifth way a backtest flatters you, and unlike the other four it
survives a perfect pipeline: no value was read early, no label overlapped a
test block, no sample was chosen after the fact, no figure was revised. The
arithmetic is right. The selection is the problem::

    from mdnorm import sharpe_report, deflated_sharpe_ratio

    rep = sharpe_report(daily_returns, periods_per_year=Decimal(252))
    rep.sharpe_annualised     # the number that goes in the deck
    rep.probabilistic         # the probability it is above zero at all
    rep.warnings              # what the number does not tell you

**Sharpe ratios here are per period unless you annualise them yourself.**
:func:`sharpe_ratio` divides mean by standard deviation and stops. There is no
default multiplication by the square root of 252, for the same reason there is
none in :mod:`mdnorm.features`: it is right for daily bars on a market with 252
sessions and wrong for every other input, and being wrong by a constant is the
hardest kind of wrong to notice. :func:`annualise_sharpe` does it in one
explicit line once you have stated the calendar.

**The distributional functions take a per-period Sharpe.** :func:`probabilistic_sharpe_ratio`,
:func:`min_track_record_length` and :func:`deflated_sharpe_ratio` are all
defined against the observation frequency. Handing them an annualised figure
produces a confident, plausible, wrong answer, and nothing in the arithmetic
can detect it. Pass the raw per-period ratio and the number of observations it
came from.

**Zero dispersion is not a ratio.** A return series that never moved has no
Sharpe, a strategy with no losing period in the sample has no measurable
downside, and a curve that never fell has no drawdown. These return ``None``
rather than zero or infinity, because each of them is a statement about the
sample being too short, not about the strategy being perfect.

**Missing observations are dropped and counted.** A series arriving from
:func:`mdnorm.features.returns` begins with a hole by construction, so refusing
gaps outright would be useless. They are skipped, and every report says how
many — a bare :func:`sharpe_ratio` gives you the ratio with no way to see that
half the sample was absent, which is why the reports exist.

The deflation machinery follows Bailey and López de Prado, *The Sharpe Ratio
Efficient Frontier* (2012) and *The Deflated Sharpe Ratio* (2014). The
distributional parts are evaluated in double precision; the moments are not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "Drawdown",
    "Moments",
    "SharpeReport",
    "moments",
    "equity_curve",
    "drawdowns",
    "max_drawdown",
    "sharpe_ratio",
    "annualise_sharpe",
    "sortino_ratio",
    "calmar_ratio",
    "hit_rate",
    "profit_factor",
    "turnover",
    "probabilistic_sharpe_ratio",
    "min_track_record_length",
    "expected_max_sharpe",
    "trial_variance",
    "deflated_sharpe_ratio",
    "sharpe_report",
]

_Series = Sequence[Optional[Decimal]]

#: Working precision, matching :mod:`mdnorm.features`. Wide enough that
#: rounding never reaches a figure anyone reports.
_PRECISION = 34

#: Euler-Mascheroni constant, used by :func:`expected_max_sharpe`.
_EULER = 0.5772156649015329

_ZERO = Decimal(0)
_ONE = Decimal(1)


# -- the normal distribution, without a dependency ---------------------------


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# Coefficients from Peter Acklam's inverse normal CDF approximation, refined
# below by one Halley step so the result is accurate to roughly 1e-15.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)
_P_LOW = 0.02425


def _phi_inv(p: float) -> float:
    """Inverse of the standard normal CDF, for ``0 < p < 1``."""
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    if p < _P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        x = ((((( _C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q
             + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    elif p <= 1.0 - _P_LOW:
        q = p - 0.5
        r = q * q
        x = ((((( _A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r
             + _A[5]) * q / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r
                              + _B[4]) * r + 1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -((((( _C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q
              + _C[5]) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    # One Halley refinement against the true CDF.
    e = _phi(x) - p
    u = e * math.sqrt(2.0 * math.pi) * math.exp(x * x / 2.0)
    return x - u / (1.0 + x * u / 2.0)


def _dec(value: float) -> Decimal:
    """A float result as a Decimal, without inventing precision it lacks."""
    if math.isnan(value) or math.isinf(value):
        raise ValueError("result is not a finite number")
    return Decimal(repr(value))


# -- moments -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Moments:
    """The first four sample moments of a return series.

    ``kurtosis`` is not excess kurtosis: a normal distribution scores 3, not 0.
    The deflation formulas below are written against that convention and the
    difference between the two is a factor nobody spots in a report.
    """

    observations: int
    skipped: int
    mean: Optional[Decimal]
    stdev: Optional[Decimal]
    skewness: Optional[Decimal]
    kurtosis: Optional[Decimal]

    @property
    def excess_kurtosis(self) -> Optional[Decimal]:
        """``kurtosis - 3``, for reports that use the other convention."""
        return None if self.kurtosis is None else self.kurtosis - 3


def _clean(values: _Series) -> Tuple[List[Decimal], int]:
    """Present observations, and how many were missing."""
    kept: List[Decimal] = []
    skipped = 0
    for v in values:
        if v is None:
            skipped += 1
        else:
            kept.append(Decimal(v))
    return kept, skipped


def moments(values: _Series, *, ddof: int = 1) -> Moments:
    """Count, mean, standard deviation, skewness and kurtosis in one pass.

    ``ddof`` applies to the standard deviation only. Skewness and kurtosis are
    the population estimators, which is what the Sharpe deflation formulas
    expect; the sample-corrected versions differ by a factor that matters only
    for very short series, where the whole exercise is unsound anyway.
    """
    if ddof < 0:
        raise ValueError("ddof must not be negative")
    kept, skipped = _clean(values)
    n = len(kept)
    if n == 0:
        return Moments(0, skipped, None, None, None, None)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        mean = sum(kept, _ZERO) / n
        if n <= ddof:
            return Moments(n, skipped, mean, None, None, None)
        devs = [v - mean for v in kept]
        m2 = sum((d * d for d in devs), _ZERO)
        stdev = (m2 / (n - ddof)).sqrt()
        pop_var = m2 / n
        if pop_var == 0:
            return Moments(n, skipped, mean, stdev, None, None)
        pop_sd = pop_var.sqrt()
        m3 = sum((d * d * d for d in devs), _ZERO) / n
        m4 = sum((d * d * d * d for d in devs), _ZERO) / n
        skew = m3 / (pop_sd ** 3)
        kurt = m4 / (pop_var * pop_var)
        return Moments(n, skipped, mean, stdev, skew, kurt)


# -- equity, drawdown --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Drawdown:
    """One peak-to-trough decline in an equity curve.

    All three indices point into the curve returned by :func:`equity_curve`,
    which is one element longer than the return series it came from.
    ``recovery_index`` is ``None`` for a decline that had not recovered by the
    end of the sample — the most important one to report and the one most
    often dropped, because it has no end date to put in a table.
    """

    peak_index: int
    trough_index: int
    recovery_index: Optional[int]
    peak_value: Decimal
    trough_value: Decimal
    depth: Decimal

    @property
    def recovered(self) -> bool:
        return self.recovery_index is not None

    @property
    def length(self) -> int:
        """Observations from peak to trough."""
        return self.trough_index - self.peak_index

    @property
    def recovery_length(self) -> Optional[int]:
        """Observations from trough back to the old high."""
        if self.recovery_index is None:
            return None
        return self.recovery_index - self.trough_index


def equity_curve(
    values: _Series, *, initial: Decimal = _ONE, compound: bool = True
) -> List[Decimal]:
    """Cumulative equity from a series of period returns.

    The result has one more element than the input: index 0 is ``initial``,
    before anything happened. Missing returns are treated as no change and are
    not counted as observations by the statistics elsewhere in this module.

    ``compound=False`` adds returns to the initial capital instead of
    multiplying, which is the right convention for a series of returns on a
    fixed notional rather than on a growing balance.
    """
    if initial <= 0:
        raise ValueError("initial capital must be positive")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        out = [Decimal(initial)]
        level = Decimal(initial)
        for v in values:
            if v is not None:
                if compound:
                    level = level * (_ONE + Decimal(v))
                else:
                    level = level + Decimal(initial) * Decimal(v)
            out.append(level)
        return out


def drawdowns(curve: Sequence[Decimal]) -> List[Drawdown]:
    """Every peak-to-trough decline in an equity curve, in time order.

    A decline opens when the curve falls below its running maximum and closes
    when it regains it. A decline still open at the end of the sample is
    returned with ``recovery_index`` set to ``None`` rather than being
    silently closed at the final observation.
    """
    if len(curve) < 2:
        return []
    out: List[Drawdown] = []
    peak_i = 0
    peak = Decimal(curve[0])
    trough_i: Optional[int] = None
    trough = peak
    for i in range(1, len(curve)):
        v = Decimal(curve[i])
        if v >= peak:
            if trough_i is not None:
                out.append(_make_drawdown(peak_i, peak, trough_i, trough, i))
                trough_i = None
            peak_i, peak = i, v
        elif trough_i is None or v < trough:
            trough_i, trough = i, v
    if trough_i is not None:
        out.append(_make_drawdown(peak_i, peak, trough_i, trough, None))
    return out


def _make_drawdown(
    peak_i: int, peak: Decimal, trough_i: int, trough: Decimal,
    recovery_i: Optional[int],
) -> Drawdown:
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        depth = (peak - trough) / peak if peak != 0 else _ZERO
    return Drawdown(peak_i, trough_i, recovery_i, peak, trough, depth)


def max_drawdown(curve: Sequence[Decimal]) -> Optional[Drawdown]:
    """The deepest decline in a curve, or ``None`` if it never fell.

    ``None`` means the sample contains no drawdown at all, which is a fact
    about the sample. It is not a maximum drawdown of zero, and reporting it
    as one puts an infinity in the denominator of :func:`calmar_ratio`.
    """
    dds = drawdowns(curve)
    if not dds:
        return None
    return max(dds, key=lambda d: d.depth)


# -- ratios ------------------------------------------------------------------


def sharpe_ratio(
    values: _Series, *, risk_free: Decimal = _ZERO, ddof: int = 1
) -> Optional[Decimal]:
    """Mean excess return over its standard deviation, **per period**.

    ``risk_free`` is a per-period rate, not an annual one. The result is not
    annualised: see :func:`annualise_sharpe`, which requires you to state the
    calendar. ``None`` if the series has no dispersion or too few observations
    for ``ddof``.
    """
    m = moments(values, ddof=ddof)
    if m.mean is None or m.stdev is None or m.stdev == 0:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return (m.mean - Decimal(risk_free)) / m.stdev


def annualise_sharpe(
    sharpe: Decimal, periods_per_year: Decimal
) -> Decimal:
    """Scale a per-period Sharpe ratio by the square root of the calendar.

    Deliberately a separate call. See :func:`mdnorm.features.periods_per_year`
    for the factor; there is no default, because the same series of minute
    bars annualises to two different numbers on a 24/7 venue and a six-hour
    equity session, and both look reasonable.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return Decimal(sharpe) * Decimal(periods_per_year).sqrt()


def sortino_ratio(
    values: _Series, *, target: Decimal = _ZERO, ddof: int = 1
) -> Optional[Decimal]:
    """Mean excess return over downside deviation, **per period**.

    Downside deviation counts only observations below ``target``, dividing by
    ``n - ddof`` where ``n`` is the whole sample — not by the number of losing
    periods. Dividing by the losses alone is a common variant and it makes a
    strategy with two bad days look identical to one with twenty.

    ``None`` when nothing fell below the target. That is not an infinite
    Sortino ratio; it is a sample with no downside in it yet.
    """
    if ddof < 0:
        raise ValueError("ddof must not be negative")
    kept, _ = _clean(values)
    n = len(kept)
    if n == 0 or n <= ddof:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        tgt = Decimal(target)
        mean = sum(kept, _ZERO) / n
        below = [(v - tgt) for v in kept if v < tgt]
        if not below:
            return None
        dd = (sum((d * d for d in below), _ZERO) / (n - ddof)).sqrt()
        if dd == 0:
            return None
        return (mean - tgt) / dd


def calmar_ratio(
    values: _Series, *, periods_per_year: Decimal, compound: bool = True
) -> Optional[Decimal]:
    """Annualised return over the deepest drawdown.

    ``periods_per_year`` is required for the same reason it is required
    everywhere else here. ``None`` when the curve never declined, when the
    sample is empty, or when equity reached zero — in the last case the
    annualised return is not defined and returning a large negative number
    would be a guess.
    """
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    kept, _ = _clean(values)
    n = len(kept)
    if n == 0:
        return None
    curve = equity_curve(values, compound=compound)
    worst = max_drawdown(curve)
    if worst is None or worst.depth == 0:
        return None
    final = curve[-1]
    if final <= 0:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        years = Decimal(n) / Decimal(periods_per_year)
        growth = final / curve[0]
        try:
            annual = (growth.ln() / years).exp() - _ONE
        except (InvalidOperation, ValueError):
            return None
        return annual / worst.depth


def hit_rate(values: _Series) -> Optional[Decimal]:
    """Fraction of moving periods that were positive.

    Flat periods are excluded from both the numerator and the denominator, and
    a series that never moved returns ``None``. Counting zeros as losses is
    the other convention and it makes an instrument that is closed half the
    time look like it loses half the time.
    """
    kept, _ = _clean(values)
    wins = sum(1 for v in kept if v > 0)
    losses = sum(1 for v in kept if v < 0)
    if wins + losses == 0:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return Decimal(wins) / Decimal(wins + losses)


def profit_factor(values: _Series) -> Optional[Decimal]:
    """Gross gains over gross losses.

    ``None`` when the sample contains no losing period, which is a statement
    about the sample length rather than an infinite profit factor.
    """
    kept, _ = _clean(values)
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        gains = sum((v for v in kept if v > 0), _ZERO)
        losses = sum((-v for v in kept if v < 0), _ZERO)
        if losses == 0:
            return None
        return gains / losses


def turnover(
    weights: Sequence[Mapping[str, Decimal]]
) -> List[Optional[Decimal]]:
    """One-sided turnover at each rebalance: half the sum of absolute changes.

    The first element is ``None``, because there is no previous allocation to
    compare against — the initial purchase is not a rebalance, and counting it
    as one puts a spike at the start of every turnover series. An instrument
    absent from either allocation is treated as a zero weight in that one, so
    entering and leaving the book both register.
    """
    out: List[Optional[Decimal]] = []
    prev: Optional[Mapping[str, Decimal]] = None
    for w in weights:
        if prev is None:
            out.append(None)
        else:
            with localcontext() as ctx:
                ctx.prec = _PRECISION
                names = set(prev) | set(w)
                total = sum(
                    (abs(Decimal(w.get(k, _ZERO)) - Decimal(prev.get(k, _ZERO)))
                     for k in names),
                    _ZERO,
                )
                out.append(total / 2)
        prev = w
    return out


# -- how much of it is selection ---------------------------------------------


def _psr_variance(sharpe: float, skewness: float, kurtosis: float) -> float:
    """Variance term shared by PSR and the minimum track record length."""
    return 1.0 - skewness * sharpe + (kurtosis - 1.0) / 4.0 * sharpe * sharpe


def probabilistic_sharpe_ratio(
    sharpe: Decimal,
    *,
    observations: int,
    skewness: Decimal = _ZERO,
    kurtosis: Decimal = Decimal(3),
    benchmark: Decimal = _ZERO,
) -> Decimal:
    """Probability that the true Sharpe ratio exceeds ``benchmark``.

    ``sharpe`` and ``benchmark`` are **per period**, matching ``observations``.
    ``kurtosis`` is non-excess, so leave it at 3 for a normal assumption.

    This is the honest reading of a Sharpe ratio from a short sample. A ratio
    of 1.0 from sixty observations and the same ratio from six hundred are the
    same number and not the same evidence, and the difference is exactly what
    this returns. Negative skew and fat tails both push it down — which is why
    strategies that sell insurance score worse here than their headline ratio
    suggests.
    """
    if observations < 2:
        raise ValueError("observations must be at least 2")
    sr = float(sharpe)
    var = _psr_variance(sr, float(skewness), float(kurtosis))
    if var <= 0:
        raise ValueError(
            "the skewness and kurtosis given make the Sharpe estimator "
            "variance non-positive; check that kurtosis is non-excess"
        )
    z = (sr - float(benchmark)) * math.sqrt(observations - 1) / math.sqrt(var)
    return _dec(_phi(z))


def min_track_record_length(
    sharpe: Decimal,
    *,
    skewness: Decimal = _ZERO,
    kurtosis: Decimal = Decimal(3),
    benchmark: Decimal = _ZERO,
    confidence: Decimal = Decimal("0.95"),
) -> Optional[Decimal]:
    """Observations needed before ``sharpe`` clears ``benchmark`` at ``confidence``.

    The number of periods of track record required, at the observation
    frequency, for the ratio to be statistically distinguishable from the
    benchmark. Compare it with the sample you actually have: a strategy whose
    minimum track record is nine years and whose backtest is eighteen months
    has not been demonstrated, however good the ratio looks.

    ``None`` when ``sharpe`` does not exceed ``benchmark`` at all, in which
    case no amount of further data makes it do so.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be strictly between 0 and 1")
    sr = float(sharpe)
    bench = float(benchmark)
    if sr <= bench:
        return None
    var = _psr_variance(sr, float(skewness), float(kurtosis))
    if var <= 0:
        raise ValueError(
            "the skewness and kurtosis given make the Sharpe estimator "
            "variance non-positive; check that kurtosis is non-excess"
        )
    z = _phi_inv(float(confidence))
    return _dec(1.0 + var * (z / (sr - bench)) ** 2)


def expected_max_sharpe(trials: int, variance: Decimal) -> Decimal:
    """The best Sharpe ratio you would expect from ``trials`` worthless attempts.

    Given ``trials`` independent strategies whose true Sharpe ratios are all
    zero, and whose estimated ratios vary with ``variance``, this is roughly
    the highest estimate among them. It is the number a search produces from
    nothing, and it grows with the size of the search: it is the reason a
    grid over four parameters yields a good-looking result on any data at all.

    ``variance`` is the variance of the estimated Sharpe ratios across the
    trials — see :func:`trial_variance` — at the observation frequency.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if variance < 0:
        raise ValueError("variance must not be negative")
    if trials == 1 or variance == 0:
        return _ZERO
    n = float(trials)
    a = _phi_inv(1.0 - 1.0 / n)
    b = _phi_inv(1.0 - 1.0 / (n * math.e))
    return _dec(math.sqrt(float(variance)) * ((1.0 - _EULER) * a + _EULER * b))


def trial_variance(sharpes: Sequence[Decimal], *, ddof: int = 1) -> Optional[Decimal]:
    """Variance of the Sharpe ratios produced by a search.

    Feed this every candidate the search evaluated, not the survivors. A
    variance computed from the top ten of two hundred is itself selected, and
    it understates the dispersion that produced them.
    """
    m = moments(list(sharpes), ddof=ddof)
    if m.stdev is None:
        return None
    with localcontext() as ctx:
        ctx.prec = _PRECISION
        return m.stdev * m.stdev


def deflated_sharpe_ratio(
    sharpe: Decimal,
    *,
    observations: int,
    trials: int,
    variance: Decimal,
    skewness: Decimal = _ZERO,
    kurtosis: Decimal = Decimal(3),
) -> Decimal:
    """Probability that the best result of a search is better than the search.

    The probabilistic Sharpe ratio measured against what ``trials`` attempts
    would have produced by chance, rather than against zero. A value near 0.5
    means the winning strategy is indistinguishable from the best of that many
    coin flips; the headline ratio may still be large.

    All inputs are at the observation frequency. ``trials`` is the number of
    configurations evaluated — the whole search, including the ones discarded
    early, because those are what made the maximum a maximum.
    """
    benchmark = expected_max_sharpe(trials, variance)
    return probabilistic_sharpe_ratio(
        sharpe,
        observations=observations,
        skewness=skewness,
        kurtosis=kurtosis,
        benchmark=benchmark,
    )


# -- the report --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SharpeReport:
    """A Sharpe ratio with the things a Sharpe ratio leaves out.

    ``warnings`` is the part worth reading. Everything in it is derivable from
    the other fields, and none of it survives being copied into a slide.
    """

    observations: int
    skipped: int
    sharpe: Optional[Decimal]
    periods_per_year: Optional[Decimal]
    sharpe_annualised: Optional[Decimal]
    skewness: Optional[Decimal]
    kurtosis: Optional[Decimal]
    probabilistic: Optional[Decimal]
    min_track_record: Optional[Decimal]
    deflated: Optional[Decimal]
    trials: Optional[int]
    warnings: Tuple[str, ...] = field(default=())

    @property
    def demonstrated(self) -> Optional[bool]:
        """Whether the sample is at least as long as the minimum track record.

        ``None`` when there is no minimum track record to compare against,
        which happens when the Sharpe ratio is not above the benchmark.
        """
        if self.min_track_record is None:
            return None
        return Decimal(self.observations) >= self.min_track_record


def sharpe_report(
    values: _Series,
    *,
    risk_free: Decimal = _ZERO,
    periods_per_year: Optional[Decimal] = None,
    confidence: Decimal = Decimal("0.95"),
    trials: Optional[int] = None,
    trial_sharpe_variance: Optional[Decimal] = None,
    ddof: int = 1,
) -> SharpeReport:
    """Everything this module can say about one series of returns.

    Pass ``trials`` and ``trial_sharpe_variance`` together to get a deflated
    ratio; pass neither and the report covers the single-strategy case. Giving
    only one of them is an error rather than a silent omission, because the
    deflated figure is the one that changes the conclusion and its absence
    should not be something you discover later.
    """
    if (trials is None) != (trial_sharpe_variance is None):
        raise ValueError(
            "trials and trial_sharpe_variance must be given together"
        )
    m = moments(values, ddof=ddof)
    warns: List[str] = []
    if m.skipped:
        warns.append(
            f"{m.skipped} of {m.skipped + m.observations} observation(s) were "
            f"missing and excluded"
        )
    if m.mean is None or m.stdev is None or m.stdev == 0:
        if m.observations == 0:
            warns.append("no observations")
        else:
            warns.append(
                "returns have no dispersion over this sample; Sharpe is undefined"
            )
        return SharpeReport(
            m.observations, m.skipped, None, periods_per_year, None,
            m.skewness, m.kurtosis, None, None, None, trials, tuple(warns),
        )

    with localcontext() as ctx:
        ctx.prec = _PRECISION
        sr = (m.mean - Decimal(risk_free)) / m.stdev

    annualised = None
    if periods_per_year is None:
        warns.append(
            "Sharpe is per period; supply periods_per_year to annualise it "
            "(there is no safe default)"
        )
    else:
        annualised = annualise_sharpe(sr, periods_per_year)

    skew = m.skewness if m.skewness is not None else _ZERO
    kurt = m.kurtosis if m.kurtosis is not None else Decimal(3)

    psr: Optional[Decimal] = None
    mintrl: Optional[Decimal] = None
    deflated: Optional[Decimal] = None
    try:
        psr = probabilistic_sharpe_ratio(
            sr, observations=m.observations, skewness=skew, kurtosis=kurt
        )
        mintrl = min_track_record_length(
            sr, skewness=skew, kurtosis=kurt, confidence=confidence
        )
    except ValueError as exc:
        warns.append(f"probabilistic Sharpe not computed: {exc}")

    if mintrl is not None and Decimal(m.observations) < mintrl:
        warns.append(
            f"sample is shorter than the minimum track record length: "
            f"{m.observations} observation(s) against "
            f"{mintrl.quantize(Decimal('1'))} required at "
            f"{confidence} confidence"
        )

    if trials is not None and trial_sharpe_variance is not None:
        try:
            deflated = deflated_sharpe_ratio(
                sr,
                observations=m.observations,
                trials=trials,
                variance=trial_sharpe_variance,
                skewness=skew,
                kurtosis=kurt,
            )
        except ValueError as exc:
            warns.append(f"deflated Sharpe not computed: {exc}")
        else:
            if deflated < Decimal("0.95"):
                warns.append(
                    f"after {trials} trial(s) the deflated Sharpe is "
                    f"{deflated.quantize(Decimal('0.001'))}; the result is not "
                    f"clearly better than the best of that many attempts on noise"
                )
    elif m.observations:
        warns.append(
            "no trial count supplied; this figure is not adjusted for how many "
            "configurations were tried before it was chosen"
        )

    return SharpeReport(
        m.observations, m.skipped, sr, periods_per_year, annualised,
        m.skewness, m.kurtosis, psr, mintrl, deflated, trials, tuple(warns),
    )
