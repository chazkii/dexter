import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import {
  appendReplayCacheBundle,
  createArbiterReplayBundleFromArbitratorInput,
  type ArbiterReplayBundle,
} from './arbiter-replay.js';

export type ForecastMarketSemantics =
  | 'terminal'
  | 'barrier_touch'
  | 'range'
  | 'path_dependent'
  | 'ambiguous'
  | 'unknown';

export type ForecastArbiterVerdict =
  | 'LONG'
  | 'SHORT'
  | 'NO_TRADE'
  | 'CONDITIONAL_LONG'
  | 'CONDITIONAL_SHORT';

export type ForecastTrustPolicyLevel = 'full' | 'context-only' | 'abstain';

type Direction = 'long' | 'short' | 'neutral';

export interface ForecastMarketEvidence {
  marketId?: string;
  assetId?: string;
  question: string;
  probability?: number;
  semantics?: ForecastMarketSemantics;
  price?: number;
  volume24h?: number;
  endDate?: string | null;
  bid?: number;
  ask?: number;
}

export interface ForecastArbiterInput {
  ticker: string;
  horizon_days: number;
  current_price?: number;
  leverage?: number;
  markov?: {
    forecast_return?: number;
    p_up?: number;
    confidence?: number;
    structural_break?: boolean;
    flat_probability?: number;
    ci_low?: number;
    ci_high?: number;
    trusted_anchors?: number;
    total_anchors?: number;
    anchor_quality?: string;
    conformal?: {
      applied?: boolean;
      radius?: number;
      coverageEstimate?: number | null;
      mode?: 'normal' | 'break';
    };
    summary?: string;
  };
  polymarket?: {
    forecast_return?: number;
    raw_forecast_return?: number;
    blended_forecast_return?: number;
    confidence?: number;
    quality_score?: number;
    quality_grade?: string;
    querySet?: string[];
    markets?: ForecastMarketEvidence[];
    summary?: string;
  };
  whale?: {
    direction?: Direction;
    confidence?: number;
    summary?: string;
    source?: string;
    observationWindowStart?: string;
    observationWindowEnd?: string;
    txCount?: number;
    notionalUsd?: number | null;
    txHashes?: string[];
  };
}

export interface ForecastArbiterResult {
  ticker: string;
  horizonDays: number;
  currentPrice: number | null;
  leverage: number;
  verdict: ForecastArbiterVerdict;
  preferredDirection: Direction;
  confidence: 'low' | 'medium' | 'high';
  shouldEnterNow: boolean;
  semanticSummary: {
    primaryPolymarketSemantics: ForecastMarketSemantics;
    counts: Record<ForecastMarketSemantics, number>;
    barrierPrices: number[];
    reconciliation: string;
  };
  disagreement: {
    markovDirection: Direction;
    polymarketDirection: Direction;
    whaleDirection: Direction;
    isDivergent: boolean;
    summary: string;
  };
  leverageAssessment: {
    long: TradeScore;
    short: TradeScore;
    warning: string | null;
  };
  conditionalPlan: {
    longTrigger: string | null;
    shortTrigger: string | null;
    invalidation: string | null;
  };
  policy: {
    level: ForecastTrustPolicyLevel;
    horizonEligible: boolean;
    tradeEligible: boolean;
    reasons: string[];
  };
  rationale: string[];
  rawEvidence: {
    markov: ForecastArbiterInput['markov'] | null;
    polymarket: ForecastArbiterInput['polymarket'] | null;
    whale: ForecastArbiterInput['whale'] | null;
  };
}

interface TradeScore {
  directionalEdgePct: number;
  riskAdjustedScore: number;
  leveragePnlPct: number;
  rr: number | null;
  notes: string[];
}

interface ForecastTrustPolicy {
  level: ForecastTrustPolicyLevel;
  horizonEligible: boolean;
  tradeEligible: boolean;
  reasons: string[];
}

const SEMANTICS_VALUES: ForecastMarketSemantics[] = [
  'terminal',
  'barrier_touch',
  'range',
  'path_dependent',
  'ambiguous',
  'unknown',
];

function optionalNumber(options: { min?: number; max?: number; positive?: boolean } = {}) {
  let numberSchema = z.coerce.number().finite();
  if (options.positive) numberSchema = numberSchema.positive();
  if (options.min !== undefined) numberSchema = numberSchema.min(options.min);
  if (options.max !== undefined) numberSchema = numberSchema.max(options.max);
  return z.preprocess(
    (value) => value === null || value === '' ? undefined : value,
    numberSchema.optional(),
  ).optional();
}

function optionalBoolean() {
  return z.preprocess((value) => {
    if (value === null || value === '') return undefined;
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase();
      if (['true', 'yes', 'y', '1'].includes(normalized)) return true;
      if (['false', 'no', 'n', '0'].includes(normalized)) return false;
    }
    return value;
  }, z.boolean().optional()).optional();
}

function optionalString() {
  return z.preprocess((value) => {
    if (value === null || value === undefined) return undefined;
    if (typeof value === 'string') return value;
    return JSON.stringify(value);
  }, z.string().max(10_000).optional()).optional();
}

function optionalDirection() {
  return z.preprocess((value) => {
    if (value === null || value === '') return undefined;
    if (typeof value !== 'string') return value;
    const normalized = value.trim().toLowerCase();
    if (['long', 'bullish', 'buy', 'up'].includes(normalized)) return 'long';
    if (['short', 'bearish', 'sell', 'down'].includes(normalized)) return 'short';
    if (['neutral', 'none', 'no_signal', 'no signal', 'flat', 'unknown'].includes(normalized)) return 'neutral';
    return 'neutral';
  }, z.enum(['long', 'short', 'neutral']).optional()).optional();
}

function optionalSemantics() {
  return z.preprocess((value) => {
    if (value === null || value === '') return undefined;
    if (typeof value !== 'string') return value;
    const normalized = value.trim().toLowerCase().replace(/[-\s]+/g, '_');
    return SEMANTICS_VALUES.includes(normalized as ForecastMarketSemantics) ? normalized : 'unknown';
  }, z.enum(SEMANTICS_VALUES as [ForecastMarketSemantics, ...ForecastMarketSemantics[]]).optional()).optional();
}

function optionalConformalMode() {
  return z.preprocess((value) => {
    if (value === null || value === '') return undefined;
    if (typeof value !== 'string') return value;
    const normalized = value.trim().toLowerCase();
    return normalized === 'normal' || normalized === 'break' ? normalized : undefined;
  }, z.enum(['normal', 'break']).optional()).optional();
}

const schema = z.object({
  ticker: z.string().max(128).describe('Asset ticker, e.g. BTC, BTC-USD, ETH, SPY.'),
  horizon_days: z.coerce.number().int().min(1).max(365).default(1),
  current_price: optionalNumber({ positive: true }),
  leverage: optionalNumber({ positive: true, max: 125 }).default(1),
  markov: z.object({
    forecast_return: optionalNumber().describe('Markov expected return as a decimal, e.g. 0.004 for +0.4%.'),
    p_up: optionalNumber({ min: 0, max: 1 }),
    confidence: optionalNumber({ min: 0, max: 1 }),
    structural_break: optionalBoolean(),
    flat_probability: optionalNumber({ min: 0, max: 1 }),
    ci_low: optionalNumber({ positive: true }),
    ci_high: optionalNumber({ positive: true }),
    trusted_anchors: optionalNumber({ min: 0 }),
    total_anchors: optionalNumber({ min: 0 }),
    anchor_quality: optionalString(),
    conformal: z.object({
      applied: optionalBoolean(),
      radius: optionalNumber({ min: 0 }),
      coverageEstimate: z.preprocess(
        (value) => value === '' ? undefined : value,
        z.coerce.number().finite().min(0).max(1).nullable().optional(),
      ).optional(),
      mode: optionalConformalMode(),
    }).optional(),
    summary: optionalString(),
  }).optional(),
  polymarket: z.object({
    forecast_return: optionalNumber().describe('Polymarket forecast return as a decimal, e.g. -0.012 for -1.2%.'),
    raw_forecast_return: optionalNumber().describe('Raw Polymarket-only forecast return as a decimal.'),
    blended_forecast_return: optionalNumber().describe('Blended Polymarket-plus-auxiliary forecast return as a decimal.'),
    confidence: optionalNumber({ min: 0, max: 1 }),
    quality_score: optionalNumber({ min: 0, max: 100 }),
    quality_grade: optionalString(),
    querySet: z.array(z.coerce.string().max(10_000)).optional(),
    markets: z.array(z.object({
      marketId: optionalString(),
      assetId: optionalString(),
      question: z.coerce.string().max(10_000),
      probability: optionalNumber({ min: 0, max: 1 }),
      semantics: optionalSemantics(),
      price: optionalNumber({ positive: true }),
      volume24h: optionalNumber({ min: 0 }),
      endDate: optionalString(),
      bid: optionalNumber({ min: 0, max: 1 }),
      ask: optionalNumber({ min: 0, max: 1 }),
    })).optional().default([]),
    summary: optionalString(),
  }).optional(),
  whale: z.object({
    direction: optionalDirection(),
    confidence: optionalNumber({ min: 0, max: 1 }),
    summary: optionalString(),
    source: optionalString(),
    observationWindowStart: optionalString(),
    observationWindowEnd: optionalString(),
    txCount: optionalNumber({ min: 0 }),
    notionalUsd: optionalNumber({ min: 0 }),
    txHashes: z.array(z.coerce.string().max(256)).optional(),
  }).optional(),
});

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}

function round(value: number, digits = 4): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function directionFromReturn(value: number | undefined, threshold = 0.001): Direction {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) < threshold) {
    return 'neutral';
  }
  return value > 0 ? 'long' : 'short';
}

export function classifyPolymarketQuestion(question: string): ForecastMarketSemantics {
  const q = question.toLowerCase();

  // Rule-heavy conditional language makes resolution criteria ambiguous.
  if (/\b(according to|subject to|provided that|contingent on|as long as|unless)\b/.test(q)) {
    return 'ambiguous';
  }

  // Mixed terminal + barrier signals in the same question indicate unclear semantics.
  const hasBarrierSignal = /\b(hit|touch|reach|dip|tap)\b|\bdrop below\b|\bfall below\b|\btrade (above|below|at)\b/.test(q);
  const terminalScanQuestion = q
    .replace(/\b(drop|fall)\s+below\b/g, '')
    .replace(/\b(hit|touch|reach|dip|tap)\s+(above|below|over|under|at)\b/g, '')
    .replace(/\btrade (above|below|at)\b/g, '');
  const hasTerminalSignal = /\b(above|below|over|under|exceed|greater than|less than|settle|close|finish)\b/.test(terminalScanQuestion);
  if (hasTerminalSignal && hasBarrierSignal) return 'ambiguous';

  if (/\bbetween\b|\brange\b|\bwithin\b/.test(q)) return 'range';
  if (/\bstay\b|\bthrough\b|\bany time\b|\bintraday\b/.test(q)) return 'path_dependent';
  if (hasBarrierSignal) return 'barrier_touch';
  if (hasTerminalSignal) return 'terminal';

  return 'unknown';
}

export function extractPriceLevels(text: string): number[] {
  const levels: number[] = [];
  const pattern = /\$?\b(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([kK])?\b/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    const raw = match[1];
    if (!raw) continue;
    const parsed = Number.parseFloat(raw.replace(/,/g, ''));
    if (!Number.isFinite(parsed) || parsed < 10) continue;
    const level = match[2] ? parsed * 1_000 : parsed;
    levels.push(level);
  }
  return [...new Set(levels)];
}

function inferMarketSemantics(markets: ForecastMarketEvidence[] | undefined): {
  primary: ForecastMarketSemantics;
  counts: Record<ForecastMarketSemantics, number>;
  barrierPrices: number[];
} {
  const counts: Record<ForecastMarketSemantics, number> = {
    terminal: 0,
    barrier_touch: 0,
    range: 0,
    path_dependent: 0,
    ambiguous: 0,
    unknown: 0,
  };
  const barrierPrices: number[] = [];

  for (const market of markets ?? []) {
    const semantics = market.semantics ?? classifyPolymarketQuestion(market.question);
    counts[semantics]++;
    if (semantics === 'barrier_touch' || semantics === 'path_dependent') {
      const price = market.price ?? extractPriceLevels(market.question)[0];
      if (typeof price === 'number' && Number.isFinite(price)) {
        barrierPrices.push(price);
      }
    }
  }

  const primary = SEMANTICS_VALUES.reduce((best, key) =>
    counts[key] > counts[best] ? key : best, 'unknown');

  return { primary, counts, barrierPrices: [...new Set(barrierPrices)].sort((a, b) => a - b) };
}

function qualityToConfidence(input: ForecastArbiterInput['polymarket']): number {
  if (!input) return 0;
  if (typeof input.confidence === 'number' && Number.isFinite(input.confidence)) {
    return clamp(input.confidence, 0, 1);
  }
  if (typeof input.quality_score === 'number' && Number.isFinite(input.quality_score)) {
    return clamp(input.quality_score / 100, 0, 1);
  }
  return input.forecast_return === undefined ? 0 : 0.55;
}

function computeTradeScore(params: {
  direction: 'long' | 'short';
  markovReturn: number;
  polymarketReturn: number;
  markovWeight: number;
  polymarketWeight: number;
  polymarketSemantics: ForecastMarketSemantics;
  flatProbability: number;
  structuralBreak: boolean;
  hasWhaleInput: boolean;
  whaleDirection: Direction;
  whaleConfidence: number;
  leverage: number;
  currentPrice?: number;
  ciLow?: number;
  ciHigh?: number;
}): TradeScore {
  const side = params.direction === 'long' ? 1 : -1;
  const semanticWeight = params.polymarketSemantics === 'terminal'
    ? 1
    : params.polymarketSemantics === 'barrier_touch' || params.polymarketSemantics === 'path_dependent'
      ? 0.4
      : params.polymarketSemantics === 'ambiguous'
        ? 0.5
        : 0.7;
  const whaleBoost = params.whaleDirection === params.direction
    ? params.whaleConfidence * 0.0025
    : params.whaleDirection === 'neutral'
      ? 0
      : -params.whaleConfidence * 0.0025;

  const edge = side * (
    params.markovReturn * params.markovWeight
    + params.polymarketReturn * params.polymarketWeight * semanticWeight
  ) + whaleBoost;

  const flatPenalty = clamp(params.flatProbability - 0.55, 0, 0.4);
  const breakPenalty = params.structuralBreak ? 0.0025 : 0;
  const leveragePenalty = params.leverage >= 5 ? (params.leverage - 4) * 0.00035 : 0;
  const riskAdjustedScore = edge - flatPenalty * 0.006 - breakPenalty - leveragePenalty;

  let rr: number | null = null;
  if (params.currentPrice && params.currentPrice > 0) {
    const targetMove = Math.max(Math.abs(edge), 0.004);
    const adverseMove = params.direction === 'long' && params.ciLow
      ? Math.max(0.001, (params.currentPrice - params.ciLow) / params.currentPrice)
      : params.direction === 'short' && params.ciHigh
        ? Math.max(0.001, (params.ciHigh - params.currentPrice) / params.currentPrice)
        : Math.max(0.006, targetMove * 1.5);
    rr = round(targetMove / adverseMove, 2);
  }

  const notes: string[] = [];
  if (params.polymarketSemantics !== 'terminal') {
    notes.push('Prediction-market signal is not purely terminal, so it gets less directional weight.');
  }
  if (params.flatProbability >= 0.7) {
    notes.push('Flat/range-bound scenario dominates, reducing the value of an immediate directional entry.');
  }
  if (params.structuralBreak) {
    notes.push('Markov structural-break flag reduces conviction.');
  }
  if (params.leverage >= 5) {
    notes.push(`${params.leverage}x leverage magnifies small forecast errors.`);
  }
  if (!params.hasWhaleInput) {
    notes.push('Whale/on-chain input was not provided.');
  } else if (params.whaleDirection === 'neutral') {
    notes.push('Whale/on-chain input is neutral and does not confirm either side.');
  }

  return {
    directionalEdgePct: round(edge * 100, 3),
    riskAdjustedScore: round(riskAdjustedScore, 5),
    leveragePnlPct: round(edge * params.leverage * 100, 2),
    rr,
    notes,
  };
}

function confidenceFromScores(best: TradeScore, divergent: boolean, leverage: number): 'low' | 'medium' | 'high' {
  if (divergent || leverage >= 8 || best.riskAdjustedScore < 0.002) return 'low';
  if (best.riskAdjustedScore >= 0.008) return 'high';
  return 'medium';
}

function hasTerminalPolymarketSupport(markets: ForecastMarketEvidence[] | undefined): boolean {
  return (markets ?? []).some((market) => (market.semantics ?? classifyPolymarketQuestion(market.question)) === 'terminal');
}

function computeForecastTrustPolicy(params: {
  input: ForecastArbiterInput;
  leverage: number;
  semantic: ReturnType<typeof inferMarketSemantics>;
  bestScore: TradeScore;
  isDivergent: boolean;
  markovConfidence: number;
  structuralBreak: boolean;
  flatProbability: number;
}): ForecastTrustPolicy {
  const reasons: string[] = [];
  const conformal = params.input.markov?.conformal;
  const conformalApplied = conformal?.applied === true;
  const conformalRadius = conformal?.radius ?? null;
  const conformalCoverage = conformal?.coverageEstimate ?? null;
  const conformalBreakMode = conformal?.mode === 'break';
  const anchorQuality = params.input.markov?.anchor_quality?.trim().toLowerCase() ?? null;
  const trustedAnchors = params.input.markov?.trusted_anchors ?? null;
  const hasTerminalSupport = hasTerminalPolymarketSupport(params.input.polymarket?.markets);

  const weakConfidence = params.markovConfidence < 0.45;
  const severeConfidence = params.markovConfidence < 0.2;
  const weakConformal = conformalApplied
    && (
      conformalBreakMode
      || (typeof conformalCoverage === 'number' && conformalCoverage < 0.75)
      || (typeof conformalRadius === 'number' && conformalRadius >= 0.08)
    );
  const severeConformal = conformalApplied
    && (
      (typeof conformalCoverage === 'number' && conformalCoverage < 0.6)
      || (typeof conformalRadius === 'number' && conformalRadius >= 0.12)
      || (conformalBreakMode && typeof conformalCoverage === 'number' && conformalCoverage < 0.65)
    );
  const missingTrustedSupport = (
    (trustedAnchors !== null && trustedAnchors <= 0)
    || anchorQuality === 'none'
    || anchorQuality === 'weak'
    || (!hasTerminalSupport && params.structuralBreak && params.flatProbability >= 0.8 && severeConfidence)
  );

  if (params.isDivergent) {
    reasons.push('Markov and Polymarket disagree on direction, so this horizon is context-only until the signals realign.');
  }
  if (params.semantic.primary === 'barrier_touch' || params.semantic.primary === 'path_dependent') {
    reasons.push('Prediction-market support is barrier/path dependent rather than a clean terminal anchor.');
  } else if (params.semantic.primary === 'ambiguous') {
    reasons.push('Prediction-market questions have mixed or ambiguous resolution semantics, reducing directional weight.');
  }
  if (params.structuralBreak) {
    reasons.push('A structural-break flag is active, so regime trust is reduced.');
  }
  if (params.flatProbability >= 0.7) {
    reasons.push('Flat-probability is elevated, which weakens immediate directional edge.');
  }
  if (params.leverage >= 8) {
    reasons.push(`${params.leverage}x leverage is too unforgiving for the current forecast quality.`);
  }
  if (weakConfidence) {
    reasons.push('Markov prediction confidence is too weak to treat as a standalone trade trigger.');
  }
  if (weakConformal) {
    reasons.push('Conformal diagnostics are stressed, so keep the forecast as regime context rather than a full trade signal.');
  }
  if (missingTrustedSupport) {
    reasons.push('Trusted horizon support is missing, so the forecast should abstain instead of manufacturing a calibrated edge.');
  }

  let level: ForecastTrustPolicyLevel = 'full';
  if (
    missingTrustedSupport
    || (params.structuralBreak && severeConfidence && params.flatProbability >= 0.82)
    || (params.structuralBreak && severeConformal && !hasTerminalSupport)
  ) {
    level = 'abstain';
  } else if (
    params.isDivergent
    || params.flatProbability >= 0.7
    || params.leverage >= 8
    || weakConfidence
    || weakConformal
    || params.semantic.primary === 'barrier_touch'
    || params.semantic.primary === 'path_dependent'
    || params.semantic.primary === 'ambiguous'
  ) {
    level = 'context-only';
  }

  if (reasons.length === 0) {
    reasons.push(level === 'full'
      ? 'Evidence is aligned and regime diagnostics are healthy enough for full guidance.'
      : level === 'context-only'
        ? 'Use the forecast as context only until trust diagnostics improve.'
        : 'No calibrated edge is available for this horizon.');
  }

  return {
    level,
    horizonEligible: level !== 'abstain',
    tradeEligible: level === 'full',
    reasons,
  };
}

export function arbitrateForecast(input: ForecastArbiterInput): ForecastArbiterResult {
  const ticker = input.ticker.trim().toUpperCase();
  const leverage = clamp(input.leverage ?? 1, 1, 125);
  const markovReturn = input.markov?.forecast_return ?? 0;
  const polymarketReturn = input.polymarket?.forecast_return ?? 0;
  const markovConfidence = clamp(input.markov?.confidence ?? (input.markov?.forecast_return === undefined ? 0 : 0.45), 0, 1);
  const polymarketConfidence = qualityToConfidence(input.polymarket);
  const markovDirection = directionFromReturn(markovReturn);
  const polymarketDirection = directionFromReturn(polymarketReturn);
  const hasWhaleInput = input.whale !== undefined;
  const whaleDirection = input.whale?.direction ?? 'neutral';
  const whaleConfidence = clamp(input.whale?.confidence ?? 0, 0, 1);
  const whaleSummary = hasWhaleInput ? whaleDirection : 'unavailable';
  const semantic = inferMarketSemantics(input.polymarket?.markets);
  const structuralBreak = input.markov?.structural_break ?? false;
  const flatProbability = input.markov?.flat_probability ?? 0;
  const isDivergent = markovDirection !== 'neutral'
    && polymarketDirection !== 'neutral'
    && markovDirection !== polymarketDirection;

  const markovWeight = markovConfidence * (structuralBreak ? 0.6 : 1);
  const polymarketWeight = polymarketConfidence;
  const long = computeTradeScore({
    direction: 'long',
    markovReturn,
    polymarketReturn,
    markovWeight,
    polymarketWeight,
    polymarketSemantics: semantic.primary,
    flatProbability,
    structuralBreak,
    hasWhaleInput,
    whaleDirection,
    whaleConfidence,
    leverage,
    currentPrice: input.current_price,
    ciLow: input.markov?.ci_low,
    ciHigh: input.markov?.ci_high,
  });
  const short = computeTradeScore({
    direction: 'short',
    markovReturn,
    polymarketReturn,
    markovWeight,
    polymarketWeight,
    polymarketSemantics: semantic.primary,
    flatProbability,
    structuralBreak,
    hasWhaleInput,
    whaleDirection,
    whaleConfidence,
    leverage,
    currentPrice: input.current_price,
    ciLow: input.markov?.ci_low,
    ciHigh: input.markov?.ci_high,
  });

  const bestDirection: Direction = long.riskAdjustedScore > short.riskAdjustedScore
    ? 'long'
    : short.riskAdjustedScore > long.riskAdjustedScore
      ? 'short'
      : 'neutral';
  const bestScore = bestDirection === 'short' ? short : long;
  const policy = computeForecastTrustPolicy({
    input,
    leverage,
    semantic,
    bestScore,
    isDivergent,
    markovConfidence,
    structuralBreak,
    flatProbability,
  });

  const immediateEntryBlocked = isDivergent
    || leverage >= 8
    || flatProbability >= 0.7
    || bestScore.riskAdjustedScore < 0.0015
    || !policy.tradeEligible;

  let verdict: ForecastArbiterVerdict = 'NO_TRADE';
  if (!immediateEntryBlocked && bestDirection === 'long') verdict = 'LONG';
  if (!immediateEntryBlocked && bestDirection === 'short') verdict = 'SHORT';
  if (immediateEntryBlocked && bestDirection === 'long' && bestScore.riskAdjustedScore > 0) verdict = 'CONDITIONAL_LONG';
  if (immediateEntryBlocked && bestDirection === 'short' && bestScore.riskAdjustedScore > 0) verdict = 'CONDITIONAL_SHORT';
  if (isDivergent && leverage >= 8 && flatProbability >= 0.65) verdict = 'NO_TRADE';
  if (policy.level === 'abstain') verdict = 'NO_TRADE';

  const barrier = semantic.barrierPrices[0] ?? null;
  const formattedBarrier = barrier !== null ? `$${barrier.toLocaleString('en-US')}` : null;
  const longTrigger = formattedBarrier
    ? `Wait for a sweep/touch of ${formattedBarrier} followed by reclaim above that level before considering LONG.`
    : 'Wait for Markov and Polymarket to align bullish or for price to reclaim the nearest failed breakdown level.';
  const shortTrigger = formattedBarrier
    ? `Wait for an accepted break below ${formattedBarrier} and failed retest before considering SHORT.`
    : 'Wait for Markov and Polymarket to align bearish or for price to reject the nearest resistance level.';

  const rationale: string[] = [];
  if (isDivergent) rationale.push('Markov and Polymarket point in opposite directions, so the arbiter rejects a one-model trade call.');
  if (semantic.primary === 'barrier_touch' || semantic.primary === 'path_dependent') {
    rationale.push('The leading Polymarket evidence appears path-dependent/barrier-like, which can be true even if terminal Markov drift is flat or positive.');
  } else if (semantic.primary === 'ambiguous') {
    rationale.push('Polymarket questions have mixed or rule-heavy resolution criteria; treating the signal as soft directional context only.');
  } else if (semantic.primary !== 'terminal') {
    rationale.push('The leading Polymarket evidence has unclear semantics, so the directional signal is treated as soft context.');
  }
  if (flatProbability >= 0.7) rationale.push('Markov assigns high probability to a flat/range outcome, which weakens both immediate LONG and SHORT setups.');
  if (leverage >= 8) rationale.push(`${leverage}x leverage makes a normal intraday move large enough to dominate the expected edge.`);
  if (!hasWhaleInput) {
    rationale.push('Whale/on-chain data was not provided, so it cannot break the model tie.');
  } else if (whaleDirection === 'neutral') {
    rationale.push('Whale data is neutral, so it does not break the model tie.');
  }
  if (rationale.length === 0) rationale.push('Evidence is sufficiently aligned after leverage/risk adjustment.');

  const reconciliation = semantic.primary === 'barrier_touch' || semantic.primary === 'path_dependent'
    ? 'Barrier/touch markets describe whether a level is hit at any point; Markov usually describes terminal distribution at the horizon. Both can be true, so use conditional triggers rather than forcing a side.'
    : semantic.primary === 'ambiguous'
      ? 'Markets have mixed or rule-heavy resolution criteria; interpret as soft directional context rather than a clean terminal probability.'
      : 'Signals are comparable as terminal directional forecasts.';

  return {
    ticker,
    horizonDays: input.horizon_days,
    currentPrice: input.current_price ?? null,
    leverage,
    verdict,
    preferredDirection: verdict === 'NO_TRADE' ? 'neutral' : bestDirection,
    confidence: confidenceFromScores(bestScore, isDivergent, leverage),
    shouldEnterNow: verdict === 'LONG' || verdict === 'SHORT',
    semanticSummary: {
      primaryPolymarketSemantics: semantic.primary,
      counts: semantic.counts,
      barrierPrices: semantic.barrierPrices,
      reconciliation,
    },
    disagreement: {
      markovDirection,
      polymarketDirection,
      whaleDirection,
      isDivergent,
      summary: isDivergent
        ? `Divergence: Markov is ${markovDirection}, Polymarket is ${polymarketDirection}, whales are ${whaleSummary}.`
        : `No strong directional conflict: Markov is ${markovDirection}, Polymarket is ${polymarketDirection}, whales are ${whaleSummary}.`,
    },
    leverageAssessment: {
      long,
      short,
      warning: leverage >= 5
        ? `At ${leverage}x, a 1% asset move is approximately ${leverage}% position P&L before fees/funding.`
        : null,
    },
    conditionalPlan: {
      longTrigger,
      shortTrigger,
      invalidation: formattedBarrier
        ? `If price chops around ${formattedBarrier} without reclaim/rejection confirmation, keep the setup as no-trade.`
        : 'If confirmation does not appear, preserve the raw forecasts but avoid a directional recommendation.',
    },
    policy,
    rationale,
    rawEvidence: {
      markov: input.markov ?? null,
      polymarket: input.polymarket ?? null,
      whale: input.whale ?? null,
    },
  };
}

export const FORECAST_ARBITRATOR_DESCRIPTION = `
**forecast_arbitrator** — Meta-decision layer for divergent Markov, Polymarket, whale/on-chain, and leverage-aware forecasts.

Use this after gathering the raw forecast tools when:
- Markov and Polymarket disagree on direction.
- The user asks for trade direction, entry, stop, target, or leveraged setup.
- Polymarket evidence is a touch/barrier market while Markov is a terminal distribution.

This tool does not replace raw evidence. It preserves Markov, Polymarket, and whale inputs in its output, classifies forecast semantics, scores LONG vs SHORT under leverage, and may recommend NO_TRADE or conditional triggers instead of forcing a side.
`.trim();

type ForecastArbitratorToolDependencies = {
  recordReplayBundleCapture?: (bundle: ArbiterReplayBundle) => void;
};

export function createForecastArbitratorTool(
  dependencies: ForecastArbitratorToolDependencies = {},
) {
  const recordReplayBundleCapture = dependencies.recordReplayBundleCapture
    ?? ((bundle: ArbiterReplayBundle) => {
      appendReplayCacheBundle(bundle);
    });

  /** Creates the forecast arbitrator tool for comparing market-derived forecasts. */
  return new DynamicStructuredTool({
    name: 'forecast_arbitrator',
    description: FORECAST_ARBITRATOR_DESCRIPTION,
    schema,
    func: async (input) => {
      const capturedAt = new Date().toISOString();
      const result = arbitrateForecast(input);
      recordReplayBundleCapture(createArbiterReplayBundleFromArbitratorInput({
        capturedAt,
        input,
      }));
      return formatToolResult({ result });
    },
  });
}

export const forecastArbitratorTool = createForecastArbitratorTool();
