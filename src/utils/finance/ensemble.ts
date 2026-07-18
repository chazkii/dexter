/**
 * Mirrors `research/models/ensemble.py`.
 *
 * Polymarket weighted ensemble forecast engine.
 *
 * Pure math — no API calls, no side effects. All functions are exported
 * individually so they can be unit-tested in isolation.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface MarketInput {
  question: string;
  probability: number;           // raw [0,1]
  volume24hUsd: number;          // 24h USD volume
  ageDays?: number;              // days since market opened (undefined → assume 21+ = mature)
  priceSpikeDetected?: boolean;  // true if history heuristics flag a recent spike
  transitoryMove?: boolean;      // true if history heuristics flag a likely reversing move
  signalTier?: 'macro' | 'geopolitical' | 'electoral'; // default 'geopolitical'
  deltaYes: number;              // estimated asset return if YES (decimal, e.g. 0.06)
  deltaNo: number;               // estimated asset return if NO (decimal, e.g. -0.04)
  /** P1b — Days remaining until market resolution (≥ 0). Undefined ⇒ neutral (no boost/penalty). */
  daysToExpiry?: number;
  /** P5 — Requested short-horizon expiry target in days. Undefined ⇒ neutral (legacy behaviour). */
  requestedHorizonDays?: number;
  /** P1d — Bid-ask spread on YES token in [0, 1]. Undefined ⇒ no penalty. */
  bidAskSpread?: number;
  /** P1e — Per-hour drift in pp (positive = momentum, negative = fading). Undefined ⇒ no penalty. */
  priceVelocityPpH?: number;
  /** P3 — Preferred per-hour drift signal in log-odds units. Undefined ⇒ fall back to pp-space drift. */
  priceVelocityLogitPerHour?: number;
  /** P1e — Largest single-hour |Δp| over the prior 24h window. Undefined ⇒ no penalty. */
  maxHourlyJump?: number;
  /** P3 — Preferred single-hour jump signal in log-odds units. Undefined ⇒ fall back to raw-probability jumps. */
  maxHourlyLogitJump?: number;
  /** P2 — Semantic classification of the market question. 'ambiguous' applies a 40% quality discount. */
  marketSemantics?: string;
  /** P4 — Stable multi-snapshot path; rewards persistent markets with a modest quality boost. */
  stablePath?: boolean;
}

export interface OtherSignals {
  sentimentScore?: number;    // -1 to +1 (-1=bearish, 0=neutral, +1=bullish)
  fundamentalReturn?: number; // analyst 1yr return scaled to horizon (decimal)
  optionsSkew?: number;       // -1/0/+1 (bearish/neutral/bullish put-call skew)
  markovReturn?: number;      // Markov expected return over the horizon (decimal)
  horizonDays?: number;       // for scaling fundamentalReturn (default 7)
}

export interface EnsembleOptions {
  /**
   * Apply bounded adaptive reweighting using the current signal strengths and
   * cross-signal agreement. Default false to preserve legacy behaviour.
   */
  adaptiveWeighting?: boolean;
}

export interface EnsembleResult {
  forecastReturn: number;      // E[r_forecast] as decimal
  forecastPrice: number;       // S_current * (1 + forecastReturn)
  ciLow95: number;             // lower bound of 95% CI
  ciHigh95: number;            // upper bound of 95% CI
  sigma: number;               // total standard deviation
  qualityScore: number;        // 0-100
  qualityGrade: 'A' | 'B' | 'C' | 'D';
  pmSignal: number;            // E[r_PM] the polymarket component
  pmEffectiveWeight: number;   // w_PM_eff (0-0.40 scaled by market quality)
  pmNormalizedWeight: number;  // final normalized PM share after any adaptive reweighting
  avgMarketQuality: number;    // w̄ mean quality weight
  warnings: string[];
}

const LEGACY_PRICE_VELOCITY_PPH_PENALTY_THRESHOLD = 2;
const LEGACY_MAX_HOURLY_JUMP_PENALTY_THRESHOLD = 0.08;
const LOGIT_PRICE_VELOCITY_PENALTY_THRESHOLD = 0.1;
const LOGIT_MAX_HOURLY_JUMP_PENALTY_THRESHOLD = 0.35;

// P3 — Microstructure-aware weighting constants.
/** Spread penalty is amplified by up to this fraction for young + illiquid markets. */
const SPREAD_THINNESS_AMPLIFICATION = 0.40;

/**
 * Dubach (2026) Phase 3 — Category-aware spread benchmarks.
 *
 * Different signal families carry structurally different expected bid-ask spreads:
 *   - electoral:    Polymarket election markets are the most actively traded; tight spreads
 *                   are the norm, so a 7 % spread is strongly adverse-selection-significant.
 *   - macro:        Policy/rate markets are well-arbitraged; 10 % spread is the neutral cutoff
 *                   (same as the legacy global default — no change for this family).
 *   - geopolitical: Conflict/diplomatic markets trade less frequently; a wider spread is
 *                   structurally expected and should not carry the same penalty weight.
 *
 * A market whose observed spread equals the benchmark receives a full (100 %) spread penalty.
 * Below the benchmark the penalty scales proportionally.
 */
export const TIER_SPREAD_BENCHMARKS: Record<'macro' | 'geopolitical' | 'electoral', number> = {
  electoral:    0.07,
  macro:        0.10,
  geopolitical: 0.14,
};

/**
 * Phase 3 — Returns the spread benchmark for a signal tier.
 * Exported for unit-testability. Unknown / undefined tier falls back to 'geopolitical'.
 */
export function tierSpreadBenchmark(tier?: MarketInput['signalTier']): number {
  return TIER_SPREAD_BENCHMARKS[tier ?? 'geopolitical'];
}
/** Probability below which a market is considered a longshot (or above 1-this = near-certain favourite). */
const LONGSHOT_PROBABILITY_THRESHOLD = 0.07;
/**
 * Maximum additional quality discount applied to a longshot/favourite with terrible microstructure.
 *
 * Dubach (2026) Phase 1 — longshot spread premium: low-probability contracts carry
 * materially wider spreads due to elevated adverse-selection risk. When observed spread
 * confirms this, trust should be discounted more aggressively than a generic extreme-probability
 * penalty would justify. Raised from 0.30 to 0.45 to match empirical severity.
 */
const MAX_LONGSHOT_MICRO_PENALTY = 0.45;
/**
 * Minimum longshot micro-penalty that triggers an explicit warning.
 * Scaled with MAX_LONGSHOT_MICRO_PENALTY to keep warning frequency proportional.
 */
const LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD = 0.08;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function baseLiquidityQuality(volume24hUsd: number): number {
  return Math.min(1, Math.log10(volume24hUsd + 1) / 6);
}

/**
 * Dubach (2026) Phase 1 — microstructure quality score for a longshot/near-certain contract.
 *
 * Longshot spread premium finding: low-probability contracts carry systematically wider
 * bid-ask spreads as compensation for elevated adverse-selection risk. When spread data
 * is available, it is the most informative live signal and should drive the score.
 * Age and volume are supporting signals only — volume in particular must NOT be the
 * sole basis of trust when spread and age are present.
 *
 * Weighting when spread is available:
 *   spread  0.60   (primary — live microstructure evidence)
 *   age     0.25   (secondary — structural consistency over time)
 *   volume  0.15   (tertiary — can be inflated; lower evidentiary weight)
 *
 * Fallback when spread is absent (age + volume only):
 *   age     0.60   (primary — more reliable structural signal)
 *   volume  0.40   (secondary)
 *
 * Returns a score in [0, 1]; higher = better microstructure.
 */
function longshotMicrostructureScore(m: Pick<MarketInput, 'ageDays' | 'volume24hUsd' | 'bidAskSpread' | 'signalTier'>): number {
  const wAge = Math.min(1, (m.ageDays ?? 21) / 21);
  const volQuality = baseLiquidityQuality(m.volume24hUsd);
  if (m.bidAskSpread !== undefined && Number.isFinite(m.bidAskSpread)) {
    // Phase 3: use the tier-specific benchmark so longshot quality is also category-aware.
    const spreadQuality = Math.max(0, 1 - m.bidAskSpread / tierSpreadBenchmark(m.signalTier));
    const rawScore = 0.60 * spreadQuality + 0.25 * wAge + 0.15 * volQuality;
    // Spread quality is a hard ceiling: a catastrophic spread (spreadQuality = 0) cannot be
    // rescued by mature age or high volume alone. Age/volume may lift score only up to the
    // spread quality itself, preserving spread as the primary adverse-selection signal.
    return Math.min(rawScore, spreadQuality);
  }
  // No spread data: age dominates volume (age is a structural signal; volume can be inflated).
  return 0.60 * wAge + 0.40 * volQuality;
}

type SignalKey = 'pm' | 'sentiment' | 'fundamental' | 'options' | 'markov';
type SignalEntry = { weight: number; signal: number };

function signalScale(key: SignalKey, horizonDays: number): number {
  switch (key) {
    case 'pm':
      return 0.05;
    case 'sentiment':
      return 0.04;
    case 'fundamental':
      return Math.max(0.01, 0.20 * (horizonDays / 365));
    case 'options':
      return 0.03;
    case 'markov':
      return 0.05;
  }
}

function normalizedSignalStrength(
  key: SignalKey,
  signal: number,
  horizonDays: number,
): number {
  const scale = signalScale(key, horizonDays);
  return clamp(Math.abs(signal) / scale, 0, 1);
}

function agreementScore(
  key: SignalKey,
  available: Partial<Record<SignalKey, SignalEntry>>,
): number {
  const current = available[key];
  if (!current) return 0.5;
  const currentDir = Math.sign(current.signal);
  if (currentDir === 0) return 0.5;

  let signedMass = 0;
  let totalMass = 0;
  for (const [otherKey, other] of Object.entries(available) as Array<[SignalKey, SignalEntry | undefined]>) {
    if (!other || otherKey === key) continue;
    signedMass += other.weight * Math.sign(other.signal);
    totalMass += other.weight;
  }
  if (totalMass === 0) return 0.5;

  const normalizedSupport = clamp(signedMass / totalMass, -1, 1);
  return clamp(0.5 + 0.5 * currentDir * normalizedSupport, 0, 1);
}

function adaptiveWeightMultiplier(
  key: SignalKey,
  available: Partial<Record<SignalKey, SignalEntry>>,
  horizonDays: number,
): number {
  const entry = available[key];
  if (!entry) return 1;
  const strength = normalizedSignalStrength(key, entry.signal, horizonDays);
  const agreement = agreementScore(key, available);
  return clamp(1 + 0.35 * (strength - 0.5) + 0.25 * (agreement - 0.5), 0.75, 1.25);
}

// ---------------------------------------------------------------------------
// Public functions
// ---------------------------------------------------------------------------

/**
 * Apply a yes-side bias correction for conditional return estimation.
 *
 * Reichenbach & Walther (2025) found systematic YES-overtrading across 124M
 * Polymarket trades. This function uses an *additive offset* (−β when p > 0.5)
 * because ensemble conditional returns are linear in p: small absolute shifts
 * in p map directly to small return shifts.
 *
 * The Markov distribution module uses a *multiplicative* correction (p × 0.95)
 * instead, because survival probabilities are log-spaced and a multiplicative
 * discount is more natural for interpolation across price levels.
 *
 * Both corrections target the same phenomenon (YES overpricing) but use the
 * form best suited to their downstream math.
 *
 * @param p    Raw YES probability [0, 1]
 * @param beta Additive discount when p > 0.5 (default 0.035 = 3.5pp)
 */
export function adjustYesBias(p: number, beta = 0.035): number {
  if (p > 0.5) {
    return clamp(p - beta, 0.01, 0.99);
  }
  return clamp(p, 0.01, 0.99);
}

/**
 * Multiplicative YES-bias discount factor.
 * Shared constant used by markov-distribution.ts (and any future module that
 * needs multiplicative rather than additive bias correction).
 *
 * 0.95 = 5% haircut on all raw Polymarket probabilities.
 * Rationale: Reichenbach & Walther (2025) report ~5% aggregate YES overpricing.
 */
export const YES_BIAS_MULTIPLIER = 0.95;

/**
 * P1a — Empirically calibrated YES-bias correction (longshot-aware).
 *
 * Replaces the flat additive shift used by `adjustYesBias` with a U-shaped
 * curve that addresses the empirically confirmed favourite-longshot bias:
 *
 *   p < 0.05  → strong longshot discount (× 0.70)        — longshots ~30% overpriced
 *   p ∈ [0.05, 0.15) → linear interpolation (× 0.70 → × 0.95)
 *   p ∈ [0.15, 0.85] → legacy mid-range behaviour (−3.5pp when p > 0.5, else unchanged)
 *   p > 0.85  → mild favourite haircut (−2.5pp)
 *
 * Output is clamped to [0.001, 0.999].
 *
 * Sources:
 *   - Reichenbach & Walther (2025): YES-overtrading on Polymarket
 *   - l-marque calibration study (GitHub, 2024): U-shaped Brier residuals
 *   - docs/polymarket-prediction-improvements-research-2026-07.md §2
 */
export function adjustYesBiasV2(p: number): number {
  if (!Number.isFinite(p) || p <= 0) return 0.001;
  if (p >= 1) return 0.999;

  let adjusted: number;
  if (p < 0.05) {
    adjusted = p * 0.70;
  } else if (p <= 0.15) {
    const t = (p - 0.05) / 0.10;          // 0 at p=0.05, 1 at p=0.15
    const mult = 0.70 + t * (0.95 - 0.70); // 0.70 → 0.95
    adjusted = p * mult;
  } else if (p <= 0.85) {
    adjusted = p > 0.50 ? p - 0.035 : p;
  } else {
    adjusted = p - 0.025;
  }
  return clamp(adjusted, 0.001, 0.999);
}

/**
 * W3 Idea 1a — Dubach (2026) depth-decay haircut.
 *
 * Reference: arXiv 2604.24366 §4 — empirical Polymarket depth shrinks
 * materially as a contract approaches resolution (in-category log-log slope
 * ≈ 0.55 on seconds-to-close). Our `volume24hUsd` proxy is backward-looking,
 * so this corrects a *liquidity* misestimate — independent of the
 * information-value `computeExpiryBoost` (which captures martingale collapse).
 *
 *   ≥ 30 d         →  1.00     (no penalty; far enough out)
 *   undefined/NaN  →  1.00     (back-compat)
 *   day-by-day     →  (d/30)^0.55, floored at 0.5
 *
 * Floor of 0.5 keeps the haircut from collapsing a market entirely.
 */
export function depthDecayHaircut(daysToExpiry: number | undefined): number {
  if (daysToExpiry === undefined || !Number.isFinite(daysToExpiry)) return 1.0;
  if (daysToExpiry >= 30) return 1.0;
  const ratio = Math.max(0, daysToExpiry) / 30;
  const slope = 0.55;
  return Math.max(0.5, Math.pow(ratio, slope));
}

/** Time-to-resolution boost: near-expiry markets carry sharper information. */
export function computeExpiryBoost(daysToExpiry: number): number {
  if (!Number.isFinite(daysToExpiry)) return 1.0;
  if (daysToExpiry <= 1) return 1.50;
  if (daysToExpiry <= 7) return 1.20;
  if (daysToExpiry <= 30) return 1.00;
  if (daysToExpiry <= 90) return 0.85;
  return 0.70;
}

/**
 * Composite quality weight for a single Polymarket market.
 *
 * Factors in market age, liquidity (log-volume), tier discount, a 50%
 * penalty when a whale-sized price spike is detected, and a 30% discount
 * on transitory moves (unless the stronger whale penalty already applies).
 */
export function computeMarketQualityWeight(m: MarketInput): number {
  const wAge = Math.min(1, (m.ageDays ?? 21) / 21);
  // W3 Idea 1a — Dubach 2026 depth-decay haircut applied to the liquidity
  // component only. volume24hUsd is a stale proxy near resolution.
  // wLiqRaw is kept separate so the spread-thinness amplifier (P1d) uses the
  // structural liquidity signal and is not distorted by near-expiry decay.
  const wLiqRaw = baseLiquidityQuality(m.volume24hUsd);
  const wLiq = wLiqRaw * depthDecayHaircut(m.daysToExpiry);
  const tau =
    m.signalTier === 'macro'
      ? 0.90
      : m.signalTier === 'electoral'
        ? 0.55
        : 0.75;
  const deltaWhale = m.priceSpikeDetected ? 1 : 0;
  const deltaTransitory = m.transitoryMove && !m.priceSpikeDetected ? 1 : 0;
  // Whale flag applies a 50% discount (not full elimination).
  // Transitory moves apply a 30% discount unless the stronger whale discount already dominates.
  let w = wAge * wLiq * tau * (1 - deltaWhale * 0.5) * (1 - deltaTransitory * 0.3);
  if (m.stablePath && !m.priceSpikeDetected && !m.transitoryMove) {
    w *= 1.1;
  }
  // P1b — time-to-resolution boost (multiplicative). Backward compat: omitted ⇒ 1.0.
  if (m.daysToExpiry !== undefined) {
    w *= computeExpiryBoost(m.daysToExpiry);
  }
  if (m.requestedHorizonDays !== undefined && m.daysToExpiry !== undefined) {
    const horizonGap = Math.abs(m.daysToExpiry - m.requestedHorizonDays);
    w *= Math.max(0.5, 1 - 0.25 * horizonGap);
  }
  // P1d (strengthened) — bid-ask spread × thinness compound quality discount.
  // When a market is also young or illiquid, the spread penalty is amplified because
  // thin-market quotes carry less reliable microstructure information.
  // Phase 3: spread fraction is normalised against the tier-specific benchmark so that
  // the same absolute spread carries different weight across contract families.
  if (m.bidAskSpread !== undefined && Number.isFinite(m.bidAskSpread)) {
    const rawSpreadFrac = m.bidAskSpread / tierSpreadBenchmark(m.signalTier);
    // thinness: 0 = mature & liquid, 1 = completely young & dry.
    // Uses raw (un-decayed) liquidity so near-expiry decay does not
    // artificially inflate thinness and compound the spread penalty.
    const thinness = 1 - Math.min(1, wAge * Math.sqrt(wLiqRaw));
    const amplification = 1 + SPREAD_THINNESS_AMPLIFICATION * thinness;
    w *= Math.max(0, 1 - rawSpreadFrac * amplification);
  }
  // P3 — prefer logit-space microstructure when present. Around p≈0.5, the old
  // 2pp/h and 8pp cutoffs map to ~0.08 and ~0.32 logit units; we round slightly
  // higher (0.10 / 0.35) to avoid over-penalizing ordinary center-book churn.
  const hasLogitVelocity = m.priceVelocityLogitPerHour !== undefined
    && Number.isFinite(m.priceVelocityLogitPerHour);
  const hasLogitJump = m.maxHourlyLogitJump !== undefined
    && Number.isFinite(m.maxHourlyLogitJump);

  if (hasLogitVelocity
    ? Math.abs(m.priceVelocityLogitPerHour!) > LOGIT_PRICE_VELOCITY_PENALTY_THRESHOLD
    : m.priceVelocityPpH !== undefined
      && Math.abs(m.priceVelocityPpH) > LEGACY_PRICE_VELOCITY_PPH_PENALTY_THRESHOLD) {
    w *= 0.80;
  }
  if (hasLogitJump
    ? m.maxHourlyLogitJump! > LOGIT_MAX_HOURLY_JUMP_PENALTY_THRESHOLD
    : m.maxHourlyJump !== undefined
      && m.maxHourlyJump > LEGACY_MAX_HOURLY_JUMP_PENALTY_THRESHOLD) {
    w *= 0.70;
  }
  // P2 — soft penalty for ambiguous or rule-heavy market semantics (not hard rejection).
  if (m.marketSemantics === 'ambiguous') {
    w *= 0.6;
  }
  // Dubach (2026) Phase 1 — longshot spread-premium penalty.
  // Extreme-probability markets (longshots / near-certain favourites) face elevated
  // adverse-selection risk. When their microstructure is also poor (wide spread, young,
  // thin volume), trust drops further. Markets with good microstructure are not penalised.
  // See longshotMicrostructureScore() for the spread-primary weighting rationale.
  const p = m.probability;
  if (p < LONGSHOT_PROBABILITY_THRESHOLD || p > 1 - LONGSHOT_PROBABILITY_THRESHOLD) {
    const microScore = longshotMicrostructureScore(m);
    w *= 1 - MAX_LONGSHOT_MICRO_PENALTY * (1 - microScore);
  }
  return Math.max(0, Math.min(1, w));
}

/**
 * Expected asset return conditioned on the adjusted YES probability.
 */
export function computeConditionalReturn(
  pAdjusted: number,
  deltaYes: number,
  deltaNo: number,
): number {
  return pAdjusted * deltaYes + (1 - pAdjusted) * deltaNo;
}

/**
 * Aggregate the Polymarket signal across all markets.
 *
 * Returns a quality-weighted average conditional return, the mean quality
 * weight, and any relevant warnings.
 */
export function computePolymarketSignal(markets: MarketInput[]): {
  signal: number;
  avgQuality: number;
  warnings: string[];
} {
  if (markets.length === 0) {
    return {
      signal: 0,
      avgQuality: 0,
      warnings: ['No Polymarket markets found — PM signal omitted'],
    };
  }

  const warnings: string[] = [];
  let weightedSum = 0;
  let totalWeight = 0;

  for (const m of markets) {
    const pAdj = adjustYesBias(m.probability);
    const w = computeMarketQualityWeight(m);
    const r = computeConditionalReturn(pAdj, m.deltaYes, m.deltaNo);
    weightedSum += w * r;
    totalWeight += w;

    if (Math.abs(m.probability - adjustYesBias(m.probability)) > 0.1) {
      warnings.push(
        `Market "${m.question}" has high YES bias (raw p=${m.probability.toFixed(3)})`,
      );
    }
    if (m.priceSpikeDetected) {
      warnings.push(
        `Market "${m.question}" has a price spike (possible whale activity) — quality discounted 50%`,
      );
    }
    if (m.transitoryMove) {
      warnings.push(
        `Market "${m.question}" shows a transitory historical move — quality discounted 30%`,
      );
    }
    if (m.marketSemantics === 'ambiguous') {
      warnings.push(
        `Market "${m.question}" has ambiguous resolution semantics — quality discounted 40%`,
      );
    }
    // P3b — Warn when a longshot/favourite has poor enough microstructure to meaningfully
    // reduce quality beyond the standard adjustYesBias probability correction.
    const mp = m.probability;
    if (mp < LONGSHOT_PROBABILITY_THRESHOLD || mp > 1 - LONGSHOT_PROBABILITY_THRESHOLD) {
      const microScore = longshotMicrostructureScore(m);
      const penalty = MAX_LONGSHOT_MICRO_PENALTY * (1 - microScore);
      if (penalty > LONGSHOT_MICRO_PENALTY_WARN_THRESHOLD) {
        const range = mp < LONGSHOT_PROBABILITY_THRESHOLD ? 'longshot' : 'near-certain favourite';
        warnings.push(
          `Market "${m.question}" is a ${range} (p=${mp.toFixed(3)}) with poor microstructure — ` +
          `quality further discounted ${Math.round(penalty * 100)}%`,
        );
      }
    }
  }

  const signal = totalWeight > 0 ? weightedSum / totalWeight : 0;
  const avgQuality = totalWeight / markets.length;

  return { signal, avgQuality, warnings };
}

/**
 * Combine the Polymarket signal with auxiliary signals into a single forecast
 * return, normalising weights so they sum to 1 across available signals.
 *
 * Base weights: PM=0.40, sentiment=0.20, fundamental=0.25, options=0.15.
 * The PM weight is further scaled by pmAvgQuality before normalisation.
 */
export function computeEnsemble(
  pmSignal: number,
  pmAvgQuality: number,
  others: OtherSignals,
  options: EnsembleOptions = {},
): { forecastReturn: number; weights: Record<string, number> } {
  const horizonDays = others.horizonDays ?? 7;

  // Effective PM weight scaled by market quality.
  const wPmEff = 0.40 * pmAvgQuality;

  // Available signals with their raw weights and returns.
  const available: Partial<Record<SignalKey, SignalEntry>> = {};

  // PM is always included.
  available.pm = { weight: wPmEff, signal: pmSignal };

  if (others.sentimentScore !== undefined && !Number.isNaN(others.sentimentScore)) {
    available.sentiment = {
      weight: 0.20,
      signal: others.sentimentScore * 0.04,
    };
  }

  if (others.fundamentalReturn !== undefined && !Number.isNaN(others.fundamentalReturn)) {
    available.fundamental = {
      weight: 0.25,
      signal: others.fundamentalReturn * (horizonDays / 365),
    };
  }

  if (others.optionsSkew !== undefined && !Number.isNaN(others.optionsSkew)) {
    available.options = {
      weight: 0.15,
      signal: others.optionsSkew * 0.03,
    };
  }

  if (others.markovReturn !== undefined && !Number.isNaN(others.markovReturn)) {
    available.markov = {
      weight: 0.20,
      signal: others.markovReturn,
    };
  }

  const effectiveAvailable = options.adaptiveWeighting
    ? Object.fromEntries(
        Object.entries(available).map(([key, entry]) => {
          const signalKey = key as SignalKey;
          const multiplier = adaptiveWeightMultiplier(signalKey, available, horizonDays);
          return [
            key,
            {
              ...entry!,
              weight: entry!.weight * multiplier,
            },
          ];
        }),
      ) as Partial<Record<SignalKey, SignalEntry>>
    : available;

  // Normalise weights to sum to 1.
  const totalRaw = Object.values(effectiveAvailable).reduce((acc, e) => acc + e.weight, 0);
  const weights: Record<string, number> = {};
  let forecastReturn = 0;

  if (totalRaw === 0) {
    // Degenerate case: all weights zero — equal-weight available signals.
    const n = Object.keys(effectiveAvailable).length;
    for (const [key, entry] of Object.entries(effectiveAvailable)) {
      const w = n > 0 ? 1 / n : 0;
      weights[key] = w;
      forecastReturn += w * entry.signal;
    }
  } else {
    for (const [key, entry] of Object.entries(effectiveAvailable)) {
      const w = entry.weight / totalRaw;
      weights[key] = w;
      forecastReturn += w * entry.signal;
    }
  }

  return { forecastReturn, weights };
}

/**
 * Estimate total forecast standard deviation combining market-level
 * uncertainty, sentiment uncertainty, and a 20% model-uncertainty buffer.
 */
export function computeVariance(
  markets: MarketInput[],
  pmWeight: number,
  sentWeight: number,
  sentSignal: number,
): number {
  if (markets.length === 0) {
    return 0.05; // default 5% uncertainty when no markets
  }

  // Total quality weight across markets (for normalisation).
  const weights = markets.map((m) => computeMarketQualityWeight(m));
  const totalWeight = weights.reduce((a, b) => a + b, 0);

  // Variance of the quality-weighted PM signal.
  let variancePmMarkets = 0;
  for (let i = 0; i < markets.length; i++) {
    const m = markets[i]!;
    const pAdj = adjustYesBias(m.probability);
    const normW = totalWeight > 0 ? weights[i]! / totalWeight : 0;
    const spread = m.deltaYes - m.deltaNo;
    variancePmMarkets += normW * normW * pAdj * (1 - pAdj) * spread * spread;
  }

  // Sentiment variance — assume ±4% std for full bullish/bearish unit.
  const varianceSent = (sentWeight * 0.04) ** 2;

  // Combine and apply 20% model-uncertainty buffer.
  const varianceCombined = pmWeight ** 2 * variancePmMarkets + varianceSent;
  return Math.sqrt(varianceCombined) * 1.2;
}

/**
 * 95% confidence interval around the forecast price using 1.96σ.
 */
export function computeCI(
  forecastPrice: number,
  sigma: number,
): { low: number; high: number } {
  return {
    low: forecastPrice * (1 - 1.96 * sigma),
    high: forecastPrice * (1 + 1.96 * sigma),
  };
}

/**
 * Composite quality score in [0, 100].
 *
 * Combines market breadth, average quality, forecast precision, signal
 * diversity, and absence of whale activity.
 */
export function computeQualityScore(
  markets: MarketInput[],
  avgQuality: number,
  sigma: number,
  signalsWithData: number,
  whaleCount: number,
): number {
  const s1 = 30 * Math.min(markets.length, 5) / 5;
  const s2 = 25 * avgQuality;
  const s3 = 20 * Math.max(0, 1 - sigma / 0.20);
  const s4 = 15 * (signalsWithData / 4);
  const s5 = markets.length > 0 ? 10 * (1 - whaleCount / markets.length) : 0;
  return Math.round(Math.min(100, Math.max(0, s1 + s2 + s3 + s4 + s5)));
}

/**
 * Convert a numeric quality score to a letter grade.
 */
export function scoreToGrade(score: number): 'A' | 'B' | 'C' | 'D' {
  if (score >= 80) return 'A';
  if (score >= 60) return 'B';
  if (score >= 40) return 'C';
  return 'D';
}

// ---------------------------------------------------------------------------
// Threshold-implied distribution forecast
// ---------------------------------------------------------------------------

/** A single point on a threshold ladder: upper-tail probability at a price level. */
export interface ThresholdLadderPoint {
  price: number;
  /** P(asset > price) at expiry, 0–1. Should be bias-corrected before use. */
  probability: number;
}

/**
 * Validates a threshold ladder for use as an authoritative price distribution.
 *
 * Requirements:
 *  - ≥ 2 distinct price levels
 *  - Prices strictly increasing
 *  - Probabilities non-increasing (allows ≤ 5pp upward reversal per step as
 *    noise; larger inversions suggest mixed-semantic markets → fall back)
 *
 * Returns `{ clean: true, warnings }` when safe to use; `{ clean: false, warnings }`
 * with a diagnostic message when the ladder should not be used.
 */
export function isCleanThresholdLadder(thresholds: ThresholdLadderPoint[]): {
  clean: boolean;
  warnings: string[];
} {
  if (thresholds.length < 2) {
    return {
      clean: false,
      warnings: ['Threshold ladder has fewer than 2 price levels; threshold-implied path not activated.'],
    };
  }

  const pts = [...thresholds].sort((a, b) => a.price - b.price);

  for (let i = 1; i < pts.length; i++) {
    if (pts[i]!.price <= pts[i - 1]!.price) {
      return {
        clean: false,
        warnings: ['Threshold ladder contains duplicate or non-increasing prices; threshold-implied path not activated.'],
      };
    }
  }

  const MONOTONE_TOLERANCE = 0.05;
  const warnings: string[] = [];

  for (let i = 1; i < pts.length; i++) {
    const delta = pts[i]!.probability - pts[i - 1]!.probability;
    if (delta > MONOTONE_TOLERANCE) {
      return {
        clean: false,
        warnings: [
          `Threshold ladder probability inversion at $${pts[i]!.price}: ` +
          `P(>${pts[i - 1]!.price}) = ${pts[i - 1]!.probability.toFixed(3)} < ` +
          `P(>${pts[i]!.price}) = ${pts[i]!.probability.toFixed(3)} — mixed semantics suspected; ` +
          `threshold-implied path not activated.`,
        ],
      };
    }
    if (delta > 0) {
      warnings.push(
        `Minor probability inversion at $${pts[i]!.price} (≤ 5pp noise); distribution smoothed.`,
      );
    }
  }

  return { clean: true, warnings };
}

/**
 * Derives an `EnsembleResult` directly from a threshold-implied price distribution.
 *
 * Decomposes the upper-tail CDF ladder into probability-weighted buckets:
 *   - Below t₀:      mid = t₀ − stride/2,     prob = 1 − p₀
 *   - Interior [i]:  mid = (tᵢ + tᵢ₊₁) / 2,  prob = pᵢ − pᵢ₊₁
 *   - Above tₙ₋₁:   mid = tₙ₋₁ + stride/2,   prob = pₙ₋₁
 *
 * The result can replace `rawPolymarketResult` in `polymarket-forecast.ts`
 * when a clean aligned ladder is present.
 */
export function computeThresholdImpliedRawForecast(
  thresholds: ThresholdLadderPoint[],
  currentPrice: number,
  horizonDays: number,
): EnsembleResult {
  const pts = [...thresholds].sort((a, b) => a.price - b.price);
  const n = pts.length;

  const avgStride =
    n >= 2 ? (pts[n - 1]!.price - pts[0]!.price) / (n - 1) : pts[0]!.price * 0.05;

  const buckets: Array<{ mid: number; prob: number }> = [];

  buckets.push({
    mid: Math.max(0, pts[0]!.price - avgStride / 2),
    prob: Math.max(0, 1 - pts[0]!.probability),
  });

  for (let i = 0; i < n - 1; i++) {
    buckets.push({
      mid: (pts[i]!.price + pts[i + 1]!.price) / 2,
      prob: Math.max(0, pts[i]!.probability - pts[i + 1]!.probability),
    });
  }

  buckets.push({
    mid: pts[n - 1]!.price + avgStride / 2,
    prob: Math.max(0, pts[n - 1]!.probability),
  });

  const totalProb = buckets.reduce((s, b) => s + b.prob, 0);
  const norm = totalProb > 0 ? buckets.map((b) => ({ ...b, prob: b.prob / totalProb })) : buckets;

  const expPrice = norm.reduce((s, b) => s + b.mid * b.prob, 0);
  const expPriceSq = norm.reduce((s, b) => s + b.mid * b.mid * b.prob, 0);
  const varPrice = Math.max(0, expPriceSq - expPrice * expPrice);

  const forecastReturn = currentPrice > 0 ? (expPrice - currentPrice) / currentPrice : 0;
  const forecastPrice = expPrice;

  const rawSigma = currentPrice > 0 ? Math.sqrt(varPrice) / currentPrice : 0.05;
  const sigmaFloor = 0.10 * Math.sqrt(Math.max(1, horizonDays) / 252);
  const sigma = Math.max(sigmaFloor, rawSigma);

  const { low: ciLow95, high: ciHigh95 } = computeCI(forecastPrice, sigma);

  const qualityScore = Math.round(
    Math.min(100, Math.max(0, 30 * Math.min(n, 5) / 5 + 20 * Math.max(0, 1 - sigma / 0.20))),
  );
  const qualityGrade = scoreToGrade(qualityScore);

  return {
    forecastReturn,
    forecastPrice,
    ciLow95,
    ciHigh95,
    sigma,
    qualityScore,
    qualityGrade,
    pmSignal: forecastReturn,
    pmEffectiveWeight: 1.0,
    pmNormalizedWeight: 1.0,
    avgMarketQuality: Math.min(1, n / 5),
    warnings: [],
  };
}

/**
 * End-to-end ensemble forecast.
 *
 * Given a current asset price, a set of Polymarket markets, and optional
 * auxiliary signals, returns a full EnsembleResult including forecast price,
 * confidence interval, quality score, and diagnostic metadata.
 */
export function runEnsemble(
  currentPrice: number,
  markets: MarketInput[],
  others: OtherSignals,
  options: EnsembleOptions = {},
): EnsembleResult {
  // Step 1: Aggregate Polymarket signal.
  const { signal: pmSignal, avgQuality, warnings } = computePolymarketSignal(markets);

  // Step 2: Blend with auxiliary signals.
  const { forecastReturn, weights } = computeEnsemble(pmSignal, avgQuality, others, options);

  // Step 3: Forecast price.
  const forecastPrice = currentPrice * (1 + forecastReturn);

  // Step 4: Uncertainty.
  const rawSigma = computeVariance(
    markets,
    weights['pm'] ?? 0,
    weights['sentiment'] ?? 0,
    others.sentimentScore ?? 0,
  );

  // Apply a minimum sigma floor based on horizon length.
  // Prediction-market variance only captures event-resolution uncertainty, not
  // general market volatility. A 10% annualised floor prevents implausibly tight
  // CIs when market probabilities are extreme (P≈0.03 makes P×(1-P) ≈ 0.03).
  const horizonFrac = Math.max(1, others.horizonDays ?? 7) / 252;
  const sigmaFloor = 0.10 * Math.sqrt(horizonFrac); // 10% annual floor scaled to horizon
  const sigma = Math.max(sigmaFloor, rawSigma);

  // Step 5: Confidence interval.
  const { low, high } = computeCI(forecastPrice, sigma);

  // Step 6: Count available signals.
  const signalsWithData =
    (markets.length > 0 ? 1 : 0) +
    (others.sentimentScore !== undefined && !Number.isNaN(others.sentimentScore) ? 1 : 0) +
    (others.fundamentalReturn !== undefined && !Number.isNaN(others.fundamentalReturn) ? 1 : 0) +
    (others.optionsSkew !== undefined && !Number.isNaN(others.optionsSkew) ? 1 : 0) +
    (others.markovReturn !== undefined && !Number.isNaN(others.markovReturn) ? 1 : 0);

  // Step 7: Whale count.
  const whaleCount = markets.filter((m) => m.priceSpikeDetected).length;

  // Step 8-9: Quality.
  const qualityScore = computeQualityScore(markets, avgQuality, sigma, signalsWithData, whaleCount);
  const qualityGrade = scoreToGrade(qualityScore);

  return {
    forecastReturn,
    forecastPrice,
    ciLow95: low,
    ciHigh95: high,
    sigma,
    qualityScore,
    qualityGrade,
    pmSignal,
    pmEffectiveWeight: 0.40 * avgQuality,
    pmNormalizedWeight: weights['pm'] ?? 0,
    avgMarketQuality: avgQuality,
    warnings,
  };
}
