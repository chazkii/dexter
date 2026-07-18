/**
 * Polymarket CLOB API client + microstructure helpers.
 *
 * Implements the velocity / spike / spread fetches outlined in
 * docs/polymarket-prediction-improvements-research-2026-07.md §3.3 and §3.4.
 *
 * Pure-math helpers (`computePriceVelocityPpH`, `computePriceVelocityLogitPerHour`,
 * `computeMaxHourlyJump`, `computeMaxHourlyLogitJump`, `parseClobPriceHistory`)
 * are exported so they can be unit-tested without live HTTP. The fetcher
 * (`fetchClobSpread`, `fetchClobPriceHistory`) is intentionally tiny — it is
 * exercised in integration tests only.
 */

const CLOB_BASE = 'https://clob.polymarket.com';
const POLYMARKET_LOGIT_EPSILON = 1e-6;

export interface ClobPricePoint {
  tSec: number;
  p: number;
}

interface RawHistoryPoint {
  t?: unknown;
  p?: unknown;
}

interface RawHistoryResponse {
  history?: unknown;
}

/** Parse the `/prices-history` JSON shape into a sorted, finite, in-range series. */
export function parseClobPriceHistory(raw: unknown): ClobPricePoint[] {
  if (raw === null || typeof raw !== 'object') return [];
  const hist = (raw as RawHistoryResponse).history;
  if (!Array.isArray(hist)) return [];
  const out: ClobPricePoint[] = [];
  for (const item of hist as RawHistoryPoint[]) {
    if (item === null || typeof item !== 'object') continue;
    const t = Number(item.t);
    const p = Number(item.p);
    if (!Number.isFinite(t) || !Number.isFinite(p)) continue;
    if (p < 0 || p > 1) continue;
    out.push({ tSec: t, p });
  }
  out.sort((a, b) => a.tSec - b.tSec);
  return out;
}

function getRecentHistoryWindow(
  history: readonly ClobPricePoint[],
  lookbackHours: number,
): ClobPricePoint[] {
  if (history.length < 2) return [];
  const newestSec = history[history.length - 1]!.tSec;
  const cutoff = newestSec - lookbackHours * 3600;
  return history.filter((pt) => pt.tSec >= cutoff);
}

function computeProjectedVelocityPerHour(
  history: readonly ClobPricePoint[],
  lookbackHours: number,
  project: (point: ClobPricePoint) => number | null,
): number {
  const window = getRecentHistoryWindow(history, lookbackHours);
  if (window.length < 2) return 0;
  let n = 0;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (const pt of window) {
    const x = pt.tSec / 3600;
    const y = project(pt);
    if (y === null || !Number.isFinite(y)) continue;
    n += 1;
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  if (n < 2) return 0;
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

function computeProjectedMaxJump(
  history: readonly ClobPricePoint[],
  windowHours: number,
  project: (point: ClobPricePoint) => number | null,
): number {
  if (history.length < 2) return 0;
  const newestSec = history[history.length - 1]!.tSec;
  const cutoff = newestSec - windowHours * 3600;
  let maxAbs = 0;
  for (let i = 1; i < history.length; i += 1) {
    const cur = history[i]!;
    if (cur.tSec < cutoff) continue;
    const prev = history[i - 1]!;
    const curProjected = project(cur);
    const prevProjected = project(prev);
    if (
      curProjected === null ||
      prevProjected === null ||
      !Number.isFinite(curProjected) ||
      !Number.isFinite(prevProjected)
    ) {
      continue;
    }
    const d = Math.abs(curProjected - prevProjected);
    if (d > maxAbs) maxAbs = d;
  }
  return maxAbs;
}

export function probabilityToBoundedLogit(probability: number): number | null {
  if (!Number.isFinite(probability) || probability < 0 || probability > 1) return null;
  if (probability === 0) probability = POLYMARKET_LOGIT_EPSILON;
  if (probability === 1) probability = 1 - POLYMARKET_LOGIT_EPSILON;
  return Math.log(probability / (1 - probability));
}

/**
 * Linear-regression slope over the last `lookbackHours` of price history,
 * expressed in **percentage points per hour** (so a 0.01 → 0.04 ramp over
 * 3 hours yields ≈ 1.0, not 0.01).
 */
export function computePriceVelocityPpH(
  history: readonly ClobPricePoint[],
  lookbackHours = 6,
): number {
  return computeProjectedVelocityPerHour(history, lookbackHours, (pt) => pt.p * 100);
}

/**
 * Linear-regression slope over the last `lookbackHours` of price history,
 * expressed in **log-odds per hour** using a bounded logit transform so
 * 0 / 1 endpoint prints remain finite.
 */
export function computePriceVelocityLogitPerHour(
  history: readonly ClobPricePoint[],
  lookbackHours = 6,
): number {
  return computeProjectedVelocityPerHour(history, lookbackHours, (pt) => probabilityToBoundedLogit(pt.p));
}

/**
 * Maximum absolute hour-over-hour price change within the last `windowHours`.
 *
 * Used to detect whale prints / sharp regime shifts. Returned as a raw delta
 * in [0, 1] (i.e. 0.13 = 13 percentage-points).
 */
export function computeMaxHourlyJump(
  history: readonly ClobPricePoint[],
  windowHours = 24,
): number {
  return computeProjectedMaxJump(history, windowHours, (pt) => pt.p);
}

/**
 * Maximum absolute adjacent logit jump within the last `windowHours`.
 *
 * Returned in bounded log-odds units so endpoint prints remain finite while
 * preserving the existing raw-probability primitive for current callers.
 */
export function computeMaxHourlyLogitJump(
  history: readonly ClobPricePoint[],
  windowHours = 24,
): number {
  return computeProjectedMaxJump(history, windowHours, (pt) => probabilityToBoundedLogit(pt.p));
}

// ---------------------------------------------------------------------------
// HTTP fetchers (kept thin; integration-tested only).
// ---------------------------------------------------------------------------

export interface ClobSpreadResponse {
  /** Spread in dollars on the YES token (e.g. 0.025 = 2.5pp). */
  spread: number;
}

export async function fetchClobSpread(tokenId: string): Promise<number | null> {
  try {
    const r = await fetch(`${CLOB_BASE}/spread/${encodeURIComponent(tokenId)}`);
    if (!r.ok) return null;
    const j = (await r.json()) as Partial<ClobSpreadResponse>;
    const s = Number(j?.spread);
    return Number.isFinite(s) && s >= 0 && s <= 1 ? s : null;
  } catch {
    return null;
  }
}

export async function fetchClobPriceHistory(
  market: string,
  interval: '1h' | '6h' | '1d' = '1h',
): Promise<ClobPricePoint[]> {
  try {
    const r = await fetch(
      `${CLOB_BASE}/prices-history?market=${encodeURIComponent(market)}&interval=${interval}`,
    );
    if (!r.ok) return [];
    return parseClobPriceHistory(await r.json());
  } catch {
    return [];
  }
}
