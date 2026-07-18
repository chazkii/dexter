/**
 * Trump Pressure Index — Cramer-Short-native implementation
 *
 * Inspired by Deutsche Bank strategist Maximilian Uleer's "Pressure Index" that
 * tracks when political/market stress may force Trump to reverse policy (a "TACO
 * moment" — Trump Always Chickens Out).
 *
 * DB core inputs (60% weight):
 *   1. S&P 500 1-month change  (w=0.20)
 *   2. 10Y Treasury yield 1-month change  (w=0.15)
 *   3. Inflation expectations 1-month change  (w=0.15)
 *   4. Approval rating 1-month change  (w=0.10)
 *
 * Cramer-Short extensions (40% weight):
 *   5. Gas prices (UGA ETF) 1-month change  (w=0.12)
 *   6. Polymarket policy-reversal probability  (w=0.15)
 *   7. Social sentiment score  (w=0.13)
 *
 * Pipeline: Z-score normalization → weighted composite → 4-state Markov regime →
 *           Polymarket TACO probability anchor → formatted output
 *
 * References:
 *   - Uleer (2026, Deutsche Bank): Original 4-input pressure methodology
 *   - Nguyen (2018, IJFS): 4-state HMM for S&P 500
 *   - Welton & Ades (2005): Dirichlet priors for transition matrix estimation
 *   - Reichenbach & Walther (2025): YES-bias in Polymarket
 */

import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { YES_BIAS_MULTIPLIER } from '../../utils/finance/ensemble.js';
import { api } from './api.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Pressure regime states — DEFCON-like 4-level system */
export type PressureRegime = 'LOW' | 'MODERATE' | 'ELEVATED' | 'CRITICAL';

export const PRESSURE_REGIMES: PressureRegime[] = ['LOW', 'MODERATE', 'ELEVATED', 'CRITICAL'];
export const NUM_PRESSURE_STATES = PRESSURE_REGIMES.length;

export const PRESSURE_STATE_INDEX: Record<PressureRegime, number> = {
  LOW: 0,
  MODERATE: 1,
  ELEVATED: 2,
  CRITICAL: 3,
};

/** Regime emoji badges for display */
export const REGIME_BADGES: Record<PressureRegime, string> = {
  LOW: '🟢 LOW',
  MODERATE: '🟡 MODERATE',
  ELEVATED: '🟠 ELEVATED',
  CRITICAL: '🔴 CRITICAL',
};

/** Single input component with its value and Z-score */
export interface PressureComponent {
  name: string;
  /** Raw 1-month change value (%, bps, or points depending on input) */
  rawValue: number;
  /** Z-score normalized (positive = more pressure) */
  zScore: number;
  /** Weight in the composite (sums to 1.0) */
  weight: number;
  /** Weighted contribution to composite (zScore × weight) */
  contribution: number;
  /** Whether data was available or a fallback was used */
  dataQuality: 'live' | 'stale' | 'unavailable';
  /** When data was unavailable, the weight is redistributed */
  qualityPenalty: number;
}

/** Historical TACO event for comparison */
export interface TacoLandmark {
  date: string;
  event: string;
  pressureScore: number;
  regime: PressureRegime;
  outcome: string;
}

/** 4×4 stochastic matrix for pressure regime transitions */
export type PressureTransitionMatrix = number[][];

/** Full output of the Trump Pressure Index computation */
export interface TrumpPressureResult {
  /** Composite pressure score (approximately N(0,1) under normal conditions) */
  pressureScore: number;
  /** Current regime classification */
  regime: PressureRegime;
  /** Probability of a TACO event (policy reversal) — blend of Markov + Polymarket */
  tacoProb: number;
  /** Per-component breakdown */
  components: PressureComponent[];
  /** Closest historical TACO landmark for comparison */
  nearestLandmark: TacoLandmark | null;
  /** 30-day regime forecast from Markov chain */
  regimeForecast: Record<PressureRegime, number>;
  /** Whether TACO alert threshold (>2.0σ) is breached */
  alertTriggered: boolean;
  /** Model warnings */
  warnings: string[];
  /** Metadata for transparency */
  metadata: {
    dataTimestamp: string;
    componentsAvailable: number;
    componentsTotal: number;
    markovObservations: number;
    polymarketAnchors: number;
    /** Whether structural break was detected in regime transitions */
    structuralBreakDetected: boolean;
  };
}

// ---------------------------------------------------------------------------
// Constants — component weights
// ---------------------------------------------------------------------------

export interface ComponentWeight {
  name: string;
  key: string;
  weight: number;
  /** Direction: 1 = rising value means more pressure, -1 = falling means more pressure */
  direction: 1 | -1;
  group: 'db_core' | 'extension';
}

export const COMPONENT_WEIGHTS: ComponentWeight[] = [
  // DB Core (60%)
  { name: 'S&P 500',            key: 'spx',       weight: 0.20, direction: -1, group: 'db_core' },
  { name: '10Y Treasury',       key: 'yield10y',   weight: 0.15, direction:  1, group: 'db_core' },
  { name: 'Inflation Expect.',   key: 'inflation',  weight: 0.15, direction:  1, group: 'db_core' },
  { name: 'Approval Rating',    key: 'approval',   weight: 0.10, direction: -1, group: 'db_core' },
  // Cramer-Short Extensions (40%)
  { name: 'Gas Prices',         key: 'gas',        weight: 0.12, direction:  1, group: 'extension' },
  { name: 'Policy Reversal',    key: 'polymarket',  weight: 0.15, direction:  1, group: 'extension' },
  { name: 'Social Sentiment',   key: 'sentiment',   weight: 0.13, direction: -1, group: 'extension' },
];

// ---------------------------------------------------------------------------
// Historical TACO landmarks (known events)
// ---------------------------------------------------------------------------

export const TACO_LANDMARKS: TacoLandmark[] = [
  {
    date: '2025-04-09',
    event: 'Liberation Day tariff pause (90-day)',
    pressureScore: 2.8,
    regime: 'CRITICAL',
    outcome: 'S&P rallied +9.5% in single day after 90-day tariff pause announced',
  },
  {
    date: '2025-05-12',
    event: 'US-China tariff de-escalation deal',
    pressureScore: 2.3,
    regime: 'CRITICAL',
    outcome: 'Tariffs reduced from 145% to 30%; markets surged',
  },
  {
    date: '2025-07-10',
    event: 'UK trade deal concession',
    pressureScore: 1.5,
    regime: 'ELEVATED',
    outcome: 'First bilateral deal — pressure partially relieved',
  },
  {
    date: '2025-01-20',
    event: 'Inauguration — baseline (no pressure)',
    pressureScore: 0.0,
    regime: 'LOW',
    outcome: 'Starting point; markets at highs',
  },
  {
    date: '2026-01-15',
    event: 'Auto tariff exemption extension',
    pressureScore: 1.8,
    regime: 'ELEVATED',
    outcome: 'Auto sector tariffs delayed after industry lobbying',
  },
];

// ---------------------------------------------------------------------------
// Z-Score computation
// ---------------------------------------------------------------------------

/**
 * Compute a rolling Z-score from an array of historical values.
 * The latest value is standardized against the rolling window's mean and std.
 * Direction: if direction=-1, the sign is flipped (falling SPX = more pressure).
 */
export function computeZScore(
  values: number[],
  direction: 1 | -1 = 1,
  windowSize = 90,
): { zScore: number; mean: number; std: number } {
  if (values.length < 2) {
    return { zScore: 0, mean: values[0] ?? 0, std: 0 };
  }

  const window = values.slice(-windowSize);
  const mean = window.reduce((a, b) => a + b, 0) / window.length;
  const variance = window.reduce((a, v) => a + (v - mean) ** 2, 0) / window.length;
  const std = Math.sqrt(variance);

  if (std < 1e-10) return { zScore: 0, mean, std: 0 };

  const latest = values[values.length - 1];
  const zScore = direction * ((latest - mean) / std);

  return { zScore, mean, std };
}

/**
 * Compute 1-month percentage change from daily price data.
 * Returns the change as a decimal (e.g. 0.05 = +5%).
 */
export function computeMonthlyChange(prices: number[]): number | null {
  if (prices.length < 21) return null;
  const current = prices[prices.length - 1];
  const monthAgo = prices[prices.length - 21]; // ~21 trading days
  if (monthAgo === 0) return null;
  return (current - monthAgo) / monthAgo;
}

// ---------------------------------------------------------------------------
// Pressure regime classification
// ---------------------------------------------------------------------------

/**
 * Classify a composite pressure score into a 4-state regime.
 * Thresholds based on standard deviations:
 *   LOW: score < 0.5σ
 *   MODERATE: 0.5σ ≤ score < 1.5σ
 *   ELEVATED: 1.5σ ≤ score < 2.0σ
 *   CRITICAL: score ≥ 2.0σ
 */
export function classifyPressureRegime(score: number): PressureRegime {
  if (score >= 2.0) return 'CRITICAL';
  if (score >= 1.5) return 'ELEVATED';
  if (score >= 0.5) return 'MODERATE';
  return 'LOW';
}

// ---------------------------------------------------------------------------
// Markov transition matrix for pressure regimes
// ---------------------------------------------------------------------------

/**
 * Estimate a 4×4 transition matrix from a sequence of pressure regimes.
 * Uses Dirichlet smoothing α=0.1 (Welton & Ades 2005).
 */
export function estimatePressureTransitionMatrix(
  regimes: PressureRegime[],
  alpha = 0.1,
  minObservations = 20,
): PressureTransitionMatrix {
  if (regimes.length < minObservations) {
    return buildDefaultPressureMatrix();
  }

  const counts: number[][] = Array.from({ length: NUM_PRESSURE_STATES }, () =>
    Array(NUM_PRESSURE_STATES).fill(alpha),
  );

  for (let i = 0; i < regimes.length - 1; i++) {
    const from = PRESSURE_STATE_INDEX[regimes[i]];
    const to = PRESSURE_STATE_INDEX[regimes[i + 1]];
    counts[from][to] += 1;
  }

  return normalizePressureRows(counts);
}

/** Default transition matrix: 0.6 diagonal, uniform off-diagonal */
export function buildDefaultPressureMatrix(): PressureTransitionMatrix {
  const diagonal = 0.6;
  const offDiag = (1 - diagonal) / (NUM_PRESSURE_STATES - 1);
  return Array.from({ length: NUM_PRESSURE_STATES }, (_, i) =>
    Array.from({ length: NUM_PRESSURE_STATES }, (_, j) => (i === j ? diagonal : offDiag)),
  );
}

/** Normalize each row to sum to 1 */
export function normalizePressureRows(matrix: number[][]): PressureTransitionMatrix {
  return matrix.map(row => {
    const sum = row.reduce((a, b) => a + b, 0);
    return sum > 0 ? row.map(v => v / sum) : row;
  });
}

/**
 * Monte Carlo simulation of regime transitions.
 * Returns the probability distribution over regimes after `horizon` steps.
 */
export function monteCarloRegimeForecast(
  transitionMatrix: PressureTransitionMatrix,
  currentRegime: PressureRegime,
  horizon = 30,
  simulations = 1000,
): Record<PressureRegime, number> {
  const counts: Record<PressureRegime, number> = { LOW: 0, MODERATE: 0, ELEVATED: 0, CRITICAL: 0 };

  for (let sim = 0; sim < simulations; sim++) {
    let state = PRESSURE_STATE_INDEX[currentRegime];

    for (let step = 0; step < horizon; step++) {
      const row = transitionMatrix[state];
      const rand = Math.random();
      let cumulative = 0;
      for (let j = 0; j < NUM_PRESSURE_STATES; j++) {
        cumulative += row[j];
        if (rand < cumulative) {
          state = j;
          break;
        }
      }
    }

    counts[PRESSURE_REGIMES[state]] += 1;
  }

  // Normalize to probabilities
  for (const regime of PRESSURE_REGIMES) {
    counts[regime] = Math.round((counts[regime] / simulations) * 1000) / 1000;
  }

  return counts;
}

/**
 * Detect structural break between first and second half of regime sequence.
 * Reuses the Frobenius divergence approach from markov-distribution.ts.
 */
export function detectPressureStructuralBreak(
  regimes: PressureRegime[],
  threshold = 0.05,
  alpha = 0.1,
): { detected: boolean; divergence: number } {
  const mid = Math.floor(regimes.length / 2);
  if (mid < 10) return { detected: false, divergence: 0 };

  const firstHalf = regimes.slice(0, mid);
  const secondHalf = regimes.slice(mid);

  const firstMatrix = estimatePressureTransitionMatrix(firstHalf, alpha, 5);
  const secondMatrix = estimatePressureTransitionMatrix(secondHalf, alpha, 5);

  let divergence = 0;
  for (let i = 0; i < NUM_PRESSURE_STATES; i++) {
    for (let j = 0; j < NUM_PRESSURE_STATES; j++) {
      divergence += (firstMatrix[i][j] - secondMatrix[i][j]) ** 2;
    }
  }

  return { detected: divergence > threshold, divergence };
}

// ---------------------------------------------------------------------------
// TACO probability blender
// ---------------------------------------------------------------------------

/**
 * Blend Markov-derived reversal probability with Polymarket policy markets.
 *
 * TACO_prob = MARKOV_WEIGHT × P_markov + POLYMARKET_WEIGHT × P_polymarket
 *
 * P_markov: probability of transitioning from ELEVATED/CRITICAL to LOW/MODERATE
 *           within 30 days (from Monte Carlo simulation).
 * P_polymarket: average probability across policy-reversal markets (tariff pause,
 *               trade deal, etc.) with YES-bias correction.
 */
export const MARKOV_WEIGHT = 0.4;
export const POLYMARKET_WEIGHT = 0.6;

export function computeTacoProb(
  markovForecast: Record<PressureRegime, number>,
  currentRegime: PressureRegime,
  polymarketReversal: number | null,
): { tacoProb: number; markovComponent: number; polymarketComponent: number } {
  // P_markov: probability of de-escalation (moving to LOW or MODERATE from current state)
  let markovDeescalation: number;
  if (currentRegime === 'LOW') {
    markovDeescalation = 0;
  } else if (currentRegime === 'MODERATE') {
    markovDeescalation = markovForecast.LOW;
  } else {
    // ELEVATED or CRITICAL: probability of dropping to LOW or MODERATE
    markovDeescalation = markovForecast.LOW + markovForecast.MODERATE;
  }

  if (polymarketReversal === null) {
    // No Polymarket data — 100% Markov
    return {
      tacoProb: markovDeescalation,
      markovComponent: markovDeescalation,
      polymarketComponent: 0,
    };
  }

  const corrected = polymarketReversal * YES_BIAS_MULTIPLIER;
  const blended = MARKOV_WEIGHT * markovDeescalation + POLYMARKET_WEIGHT * corrected;

  return {
    tacoProb: Math.min(1, Math.max(0, blended)),
    markovComponent: markovDeescalation,
    polymarketComponent: corrected,
  };
}

// ---------------------------------------------------------------------------
// Data fetchers
// ---------------------------------------------------------------------------

const FRED_BASE = 'https://fred.stlouisfed.org/graph/fredgraph.csv';

/** Fetch a FRED series and return recent daily values */
async function fetchFredHistory(seriesId: string, days = 120): Promise<number[]> {
  const endDate = new Date().toISOString().slice(0, 10);
  const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  const url = `${FRED_BASE}?id=${seriesId}&cosd=${startDate}&coed=${endDate}`;
  const res = await fetch(url, { signal: AbortSignal.timeout(10_000) });
  if (!res.ok) throw new Error(`FRED ${seriesId} HTTP ${res.status}`);
  const text = await res.text();
  const lines = text.trim().split('\n');
  const values: number[] = [];
  for (let i = 1; i < lines.length; i++) {
    const val = lines[i].split(',')[1]?.trim();
    if (val && val !== '.' && val !== '') {
      const num = parseFloat(val);
      if (!isNaN(num)) values.push(num);
    }
  }
  return values;
}

/** Fetch stock price history from Financial Datasets API, with FRED fallback */
async function fetchStockHistory(ticker: string, days = 120): Promise<number[]> {
  const endDate = new Date().toISOString().slice(0, 10);
  const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);
  // Primary: Financial Datasets API
  try {
    const { data } = await api.get('/prices/', {
      ticker,
      interval: 'day',
      start_date: startDate,
      end_date: endDate,
    });
    const rawData = data as { prices?: Array<{ close: number }> } | Array<{ close: number }>;
    const prices: Array<{ close: number }> =
      (Array.isArray(rawData) ? rawData : (rawData.prices ?? [])) as Array<{ close: number }>;
    const result = prices.map(p => p.close).filter(v => typeof v === 'number' && !isNaN(v));
    if (result.length >= 21) return result;
  } catch {
    // fall through to FRED fallback
  }
  // Fallback: FRED series for known tickers
  const fredFallback: Record<string, string> = {
    SPY: 'SP500',      // S&P 500 index (close proxy for SPY)
    UGA: 'GASDESW',    // US regular gas price (weekly, proxy for UGA)
  };
  const fredId = fredFallback[ticker.toUpperCase()];
  if (fredId) {
    try {
      return await fetchFredHistory(fredId, days);
    } catch {
      // FRED also failed
    }
  }
  return [];
}

/** Search Polymarket for policy-reversal markets and compute average reversal probability */
async function fetchPolicyReversalProb(): Promise<{ prob: number | null; count: number }> {
  try {
    // Dynamic import to avoid circular dependency
    const { fetchPolymarketMarkets } = await import('./polymarket.js');

    const queries = [
      'Trump tariff pause',
      'Trump trade deal',
      'Trump policy reversal',
      'tariff exemption',
    ];

    let totalProb = 0;
    let count = 0;

    for (const query of queries) {
      try {
        const results = await fetchPolymarketMarkets(query, 3);
        for (const market of results) {
          if (market.probability > 0 && market.probability < 1) {
            totalProb += market.probability;
            count++;
          }
        }
      } catch {
        // individual query failure — continue with others
      }
    }

    if (count === 0) return { prob: null, count: 0 };
    return { prob: totalProb / count, count };
  } catch {
    return { prob: null, count: 0 };
  }
}

/** Search Polymarket for Trump approval markets.
 *  Fallback: FRED consumer sentiment (UMCSENT) as proxy. */
async function fetchApprovalData(): Promise<number | null> {
  // Primary: Polymarket approval markets
  try {
    const { fetchPolymarketMarkets } = await import('./polymarket.js');
    const results = await fetchPolymarketMarkets('Trump approval rating', 5);

    // Look for markets about approval percentage
    for (const market of results) {
      const q = market.question.toLowerCase();
      if (q.includes('approval') && market.probability > 0) {
        // Probability that approval is above X% — use as proxy
        return market.probability * 100;
      }
    }
  } catch {
    // fall through to FRED fallback
  }
  // Fallback: FRED consumer sentiment index as proxy (UMCSENT, monthly)
  // Higher sentiment ≈ higher approval. Normalize to 0-100 scale.
  try {
    const vals = await fetchFredHistory('UMCSENT', 90);
    if (vals.length > 0) {
      // UMCSENT ranges roughly 50-110; normalize to 0-100
      const latest = vals[vals.length - 1];
      return Math.max(0, Math.min(100, latest));
    }
  } catch {
    // FRED also failed
  }
  return null;
}

/** Fetch social sentiment for Trump policy / market impact.
 *  Primary: socialSentimentTool. Fallback: GDELT news tone. */
async function fetchPolicySentiment(): Promise<number | null> {
  // Primary: social sentiment tool (Reddit, X/Twitter)
  try {
    const { socialSentimentTool } = await import('./social-sentiment.js');
    const result = await socialSentimentTool.invoke({ query: 'Trump tariffs economy market impact' });
    const match = typeof result === 'string' ? result.match(/Sentiment Score:\s*([-\d.]+)/) : null;
    if (match) return parseFloat(match[1]);
  } catch {
    // fall through to GDELT
  }
  // Fallback: GDELT news tone (free, no API key)
  try {
    const { fetchGdeltArticles } = await import('../osint/gdelt.js');
    const articles = await fetchGdeltArticles('Trump tariffs economy', { timespan: '7d', maxRecords: 50 });
    const tones = articles
      .filter(a => a.tone !== undefined && !isNaN(a.tone!))
      .map(a => a.tone!);
    if (tones.length >= 5) {
      // GDELT tone is roughly -100 to +100; normalize to -1..+1 range then scale to -100..+100
      const avgTone = tones.reduce((s, v) => s + v, 0) / tones.length;
      // GDELT tone typically ranges -10 to +10, so clamp and scale
      return Math.max(-100, Math.min(100, avgTone * 10));
    }
  } catch {
    // GDELT also failed
  }
  return null;
}

// ---------------------------------------------------------------------------
// Quality-adaptive weight redistribution
// ---------------------------------------------------------------------------

/**
 * Redistribute weights when some components are unavailable.
 * Unavailable weights are spread proportionally across available components
 * within the same group (db_core or extension), falling back to all available
 * if the entire group is missing.
 */
export function redistributeWeights(
  components: Array<{ key: string; available: boolean }>,
): Map<string, number> {
  const result = new Map<string, number>();
  const groups = new Map<string, { available: number; unavailable: number; keys: string[] }>();

  for (const cw of COMPONENT_WEIGHTS) {
    const comp = components.find(c => c.key === cw.key);
    const isAvailable = comp?.available ?? false;

    if (!groups.has(cw.group)) {
      groups.set(cw.group, { available: 0, unavailable: 0, keys: [] });
    }
    const g = groups.get(cw.group)!;
    g.keys.push(cw.key);
    if (isAvailable) {
      g.available += cw.weight;
    } else {
      g.unavailable += cw.weight;
    }

    result.set(cw.key, isAvailable ? cw.weight : 0);
  }

  // Redistribute within each group
  for (const [, g] of groups) {
    if (g.unavailable > 0 && g.available > 0) {
      const boost = g.unavailable / g.available;
      for (const key of g.keys) {
        const current = result.get(key) ?? 0;
        if (current > 0) {
          result.set(key, current * (1 + boost));
        }
      }
    }
  }

  // If an entire group is missing, redistribute to the other group
  const totalAvailable = Array.from(result.values()).reduce((a, b) => a + b, 0);
  if (totalAvailable > 0 && Math.abs(totalAvailable - 1.0) > 0.01) {
    for (const [key, val] of result) {
      result.set(key, val / totalAvailable);
    }
  }

  return result;
}

// ---------------------------------------------------------------------------
// Core computation pipeline
// ---------------------------------------------------------------------------

/**
 * Compute the Trump Pressure Index from all available data.
 * This is the main exported function — it orchestrates data fetching,
 * Z-score normalization, composite scoring, and Markov regime analysis.
 */
export async function computeTrumpPressureIndex(): Promise<TrumpPressureResult> {
  const warnings: string[] = [];
  const componentResults = new Map<string, { values: number[]; quality: 'live' | 'stale' | 'unavailable' }>();
  const warnAndFallback = <T>(label: string, fallback: T) => (err: unknown): T => {
    const msg = err instanceof Error ? err.message : String(err);
    warnings.push(`${label} unavailable: ${msg}`);
    console.warn(`[trump-pressure-index] ${label} unavailable: ${msg}`);
    return fallback;
  };

  // ── Fetch all data in parallel ────────────────────────────────────────
  const [
    spyPrices,
    ugaPrices,
    yield10y,
    breakeven5y,
    approvalData,
    policyReversal,
    sentimentScore,
  ] = await Promise.all([
    fetchStockHistory('SPY', 120).catch(warnAndFallback('SPY history', [] as number[])),
    fetchStockHistory('UGA', 120).catch(warnAndFallback('UGA history', [] as number[])),
    fetchFredHistory('DGS10', 120).catch(warnAndFallback('DGS10 history', [] as number[])),
    fetchFredHistory('T5YIE', 120).catch(warnAndFallback('T5YIE history', [] as number[])),
    fetchApprovalData().catch(warnAndFallback('approval data', null)),
    fetchPolicyReversalProb().catch(warnAndFallback('policy reversal probability', { prob: null as number | null, count: 0 })),
    fetchPolicySentiment().catch(warnAndFallback('policy sentiment', null)),
  ]);

  // ── Process each component ────────────────────────────────────────────

  // 1. S&P 500 (SPY)
  if (spyPrices.length >= 21) {
    const changes: number[] = [];
    for (let i = 21; i < spyPrices.length; i++) {
      changes.push((spyPrices[i] - spyPrices[i - 21]) / spyPrices[i - 21]);
    }
    componentResults.set('spx', { values: changes, quality: 'live' });
  } else {
    componentResults.set('spx', { values: [], quality: 'unavailable' });
    warnings.push('S&P 500 data unavailable — weight redistributed');
  }

  // 2. 10Y Treasury yield
  if (yield10y.length >= 21) {
    const changes: number[] = [];
    for (let i = 21; i < yield10y.length; i++) {
      changes.push(yield10y[i] - yield10y[i - 21]); // bps change
    }
    componentResults.set('yield10y', { values: changes, quality: 'live' });
  } else {
    componentResults.set('yield10y', { values: [], quality: 'unavailable' });
    warnings.push('10Y Treasury data unavailable (FRED) — weight redistributed');
  }

  // 3. Inflation expectations (5Y breakeven)
  if (breakeven5y.length >= 21) {
    const changes: number[] = [];
    for (let i = 21; i < breakeven5y.length; i++) {
      changes.push(breakeven5y[i] - breakeven5y[i - 21]);
    }
    componentResults.set('inflation', { values: changes, quality: 'live' });
  } else {
    componentResults.set('inflation', { values: [], quality: 'unavailable' });
    warnings.push('Inflation expectations data unavailable (FRED T5YIE) — weight redistributed');
  }

  // 4. Approval rating (Polymarket proxy)
  if (approvalData !== null) {
    // Use single-point data; Z-score will center on 0
    componentResults.set('approval', { values: [approvalData], quality: 'stale' });
  } else {
    componentResults.set('approval', { values: [], quality: 'unavailable' });
    warnings.push('Approval rating data unavailable — weight redistributed');
  }

  // 5. Gas prices (UGA ETF)
  if (ugaPrices.length >= 21) {
    const changes: number[] = [];
    for (let i = 21; i < ugaPrices.length; i++) {
      changes.push((ugaPrices[i] - ugaPrices[i - 21]) / ugaPrices[i - 21]);
    }
    componentResults.set('gas', { values: changes, quality: 'live' });
  } else {
    componentResults.set('gas', { values: [], quality: 'unavailable' });
    warnings.push('Gas price data (UGA) unavailable — weight redistributed');
  }

  // 6. Polymarket policy reversal
  if (policyReversal.prob !== null) {
    componentResults.set('polymarket', { values: [policyReversal.prob], quality: 'live' });
  } else {
    componentResults.set('polymarket', { values: [], quality: 'unavailable' });
    warnings.push('Polymarket policy markets unavailable — weight redistributed');
  }

  // 7. Social sentiment
  if (sentimentScore !== null) {
    componentResults.set('sentiment', { values: [sentimentScore], quality: 'live' });
  } else {
    componentResults.set('sentiment', { values: [], quality: 'unavailable' });
    warnings.push('Social sentiment data unavailable — weight redistributed');
  }

  // ── Compute Z-scores and weighted composite ────────────────────────────

  const availability = COMPONENT_WEIGHTS.map(cw => ({
    key: cw.key,
    available: (componentResults.get(cw.key)?.quality ?? 'unavailable') !== 'unavailable',
  }));

  const adjustedWeights = redistributeWeights(availability);

  const components: PressureComponent[] = [];
  let compositeScore = 0;

  for (const cw of COMPONENT_WEIGHTS) {
    const data = componentResults.get(cw.key);
    const adjWeight = adjustedWeights.get(cw.key) ?? 0;

    if (!data || data.quality === 'unavailable' || data.values.length === 0) {
      components.push({
        name: cw.name,
        rawValue: 0,
        zScore: 0,
        weight: adjWeight,
        contribution: 0,
        dataQuality: 'unavailable',
        qualityPenalty: 1,
      });
      continue;
    }

    const { zScore } = computeZScore(data.values, cw.direction);
    const contribution = zScore * adjWeight;
    compositeScore += contribution;

    components.push({
      name: cw.name,
      rawValue: data.values[data.values.length - 1],
      zScore,
      weight: adjWeight,
      contribution,
      dataQuality: data.quality,
      qualityPenalty: data.quality === 'stale' ? 0.5 : 0,
    });
  }

  // ── Regime classification ──────────────────────────────────────────────
  const regime = classifyPressureRegime(compositeScore);

  // ── Build historical regime sequence for Markov ────────────────────────
  // Use SPX monthly changes as a proxy for historical pressure levels
  const spxData = componentResults.get('spx');
  const historicalRegimes: PressureRegime[] = [];
  if (spxData && spxData.values.length > 0) {
    for (const val of spxData.values) {
      // Rough mapping: use SPX change as a pressure proxy
      const pseudoScore = -val / 0.05; // 5% drop ≈ 1σ pressure
      historicalRegimes.push(classifyPressureRegime(pseudoScore));
    }
  }

  const transitionMatrix = estimatePressureTransitionMatrix(historicalRegimes);
  const regimeForecast = monteCarloRegimeForecast(transitionMatrix, regime, 30, 1000);

  // ── Structural break detection ─────────────────────────────────────────
  const breakResult = detectPressureStructuralBreak(historicalRegimes);
  if (breakResult.detected) {
    warnings.push(`Structural break detected (divergence=${breakResult.divergence.toFixed(3)}) — regime transitions may be unstable`);
  }

  // ── TACO probability ───────────────────────────────────────────────────
  const { tacoProb, markovComponent, polymarketComponent } = computeTacoProb(
    regimeForecast,
    regime,
    policyReversal.prob,
  );

  // ── Alert check ────────────────────────────────────────────────────────
  const alertTriggered = compositeScore >= 2.0;
  if (alertTriggered) {
    warnings.unshift(`⚠️ TACO ALERT: Pressure at ${compositeScore.toFixed(1)}σ (CRITICAL). Historical pattern suggests policy reversal likely.`);
  }

  // ── Nearest landmark ───────────────────────────────────────────────────
  let nearestLandmark: TacoLandmark | null = null;
  let minDist = Infinity;
  for (const lm of TACO_LANDMARKS) {
    const dist = Math.abs(lm.pressureScore - compositeScore);
    if (dist < minDist) {
      minDist = dist;
      nearestLandmark = lm;
    }
  }

  return {
    pressureScore: Math.round(compositeScore * 100) / 100,
    regime,
    tacoProb: Math.round(tacoProb * 1000) / 1000,
    components,
    nearestLandmark,
    regimeForecast,
    alertTriggered,
    warnings,
    metadata: {
      dataTimestamp: new Date().toISOString(),
      componentsAvailable: availability.filter(a => a.available).length,
      componentsTotal: COMPONENT_WEIGHTS.length,
      markovObservations: historicalRegimes.length,
      polymarketAnchors: policyReversal.count,
      structuralBreakDetected: breakResult.detected,
    },
  };
}

// ---------------------------------------------------------------------------
// Output formatter
// ---------------------------------------------------------------------------

export function formatPressureResult(result: TrumpPressureResult): string {
  const lines: string[] = [];

  lines.push('# 🌮 Trump Pressure Index\n');
  lines.push(`**Regime:** ${REGIME_BADGES[result.regime]}`);
  lines.push(`**Pressure Score:** ${result.pressureScore.toFixed(2)}σ`);
  lines.push(`**TACO Probability:** ${(result.tacoProb * 100).toFixed(1)}%`);
  lines.push('');

  // Component breakdown table
  lines.push('## Component Breakdown\n');
  lines.push('| Component | Raw Value | Z-Score | Weight | Contribution | Quality |');
  lines.push('|-----------|-----------|---------|--------|-------------|---------|');
  for (const c of result.components) {
    const rawStr = c.dataQuality === 'unavailable' ? 'N/A' : c.rawValue.toFixed(4);
    const zStr = c.dataQuality === 'unavailable' ? '—' : c.zScore.toFixed(2);
    const contStr = c.dataQuality === 'unavailable' ? '—' : c.contribution.toFixed(3);
    const qualBadge = c.dataQuality === 'live' ? '🟢' : c.dataQuality === 'stale' ? '🟡' : '🔴';
    lines.push(`| ${c.name} | ${rawStr} | ${zStr} | ${(c.weight * 100).toFixed(0)}% | ${contStr} | ${qualBadge} |`);
  }
  lines.push('');

  // 30-day regime forecast
  lines.push('## 30-Day Regime Forecast\n');
  for (const r of PRESSURE_REGIMES) {
    const pct = (result.regimeForecast[r] * 100).toFixed(1);
    const bar = '█'.repeat(Math.round(result.regimeForecast[r] * 20));
    lines.push(`${REGIME_BADGES[r]}: ${pct}% ${bar}`);
  }
  lines.push('');

  // Nearest TACO landmark
  if (result.nearestLandmark) {
    lines.push('## Historical Comparison\n');
    lines.push(`Closest TACO event: **${result.nearestLandmark.event}** (${result.nearestLandmark.date})`);
    lines.push(`- Pressure then: ${result.nearestLandmark.pressureScore.toFixed(1)}σ (${result.nearestLandmark.regime})`);
    lines.push(`- Outcome: ${result.nearestLandmark.outcome}`);
    lines.push('');
  }

  // Warnings
  if (result.warnings.length > 0) {
    lines.push('## Warnings\n');
    for (const w of result.warnings) {
      lines.push(`- ${w}`);
    }
    lines.push('');
  }

  // Metadata
  lines.push('## Metadata\n');
  lines.push(`- Data: ${result.metadata.componentsAvailable}/${result.metadata.componentsTotal} components available`);
  lines.push(`- Markov observations: ${result.metadata.markovObservations}`);
  lines.push(`- Polymarket anchors: ${result.metadata.polymarketAnchors}`);
  lines.push(`- Structural break: ${result.metadata.structuralBreakDetected ? '⚠️ Yes' : 'No'}`);
  lines.push(`- Timestamp: ${result.metadata.dataTimestamp}`);

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Tool description
// ---------------------------------------------------------------------------

export const TRUMP_PRESSURE_DESCRIPTION = `
Computes the Trump Pressure Index — a composite metric that tracks when political and market stress may force Trump to reverse policy (a "TACO moment" — Trump Always Chickens Out).

Inspired by Deutsche Bank's methodology (Maximilian Uleer), extended with Polymarket prediction markets, gas prices, and social sentiment.

## When to Use
- User asks about Trump policy reversal likelihood
- Assessing tariff or trade policy risk
- Evaluating political pressure on markets
- TACO probability estimation
- "Will Trump back down on tariffs?"
- "How much political pressure is Trump under?"

## When NOT to Use
- General stock price queries (use get_market_data)
- Non-US political analysis
- Historical political data only (use web_search)

## Components (7 inputs)
1. S&P 500 monthly change (DB core)
2. 10Y Treasury yield change (DB core)
3. Inflation expectations change (DB core)
4. Approval rating change (DB core)
5. Gas prices monthly change (extension)
6. Polymarket policy-reversal probability (extension)
7. Social sentiment score (extension)

## Output
- Pressure score (σ units)
- Regime: LOW / MODERATE / ELEVATED / CRITICAL
- TACO probability (0-100%)
- Per-component breakdown
- 30-day regime forecast
- Historical TACO event comparison
- Alert when pressure ≥ 2.0σ
`.trim();

// ---------------------------------------------------------------------------
// Tool definition
// ---------------------------------------------------------------------------

const TrumpPressureInputSchema = z.object({
  query: z
    .string()
    .max(10_000)
    .optional()
    .describe('Optional context query (e.g. "tariff pressure", "market stress"). Currently unused but reserved for future filtering.'),
});
/**
 * Tool entrypoint for computing the current Trump policy pressure index.
 */
export const trumpPressureIndexTool = new DynamicStructuredTool({
  name: 'trump_pressure_index',
  description: TRUMP_PRESSURE_DESCRIPTION,
  schema: TrumpPressureInputSchema,
  func: async () => {
    try {
      const result = await computeTrumpPressureIndex();
      const formatted = formatPressureResult(result);
      return formatToolResult({ report: formatted, data: result }, []);
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      return formatToolResult(
        { error: `Trump Pressure Index computation failed: ${msg}` },
        [],
      );
    }
  },
});

// ---------------------------------------------------------------------------
// Alert helper — for watchlist briefing integration
// ---------------------------------------------------------------------------

/**
 * Check if the current pressure level triggers a TACO alert.
 * Designed to be called from watchlist/briefing workflows.
 * Returns null if no alert, or an alert message string if triggered.
 */
export async function checkTacoAlert(): Promise<string | null> {
  try {
    const result = await computeTrumpPressureIndex();
    if (result.alertTriggered) {
      return `⚠️ TACO ALERT: Trump Pressure Index at ${result.pressureScore.toFixed(1)}σ (${result.regime}). ` +
        `TACO probability: ${(result.tacoProb * 100).toFixed(0)}%. ` +
        (result.nearestLandmark
          ? `Comparable to: ${result.nearestLandmark.event} (${result.nearestLandmark.date}).`
          : 'No close historical match.');
    }
    return null;
  } catch {
    return null;
  }
}
