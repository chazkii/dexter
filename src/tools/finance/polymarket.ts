import { MS_PER_DAY } from '../../utils/time.js';
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { polymarketBreaker } from '../../utils/circuit-breaker.js';
import { resolveTickerSearchIdentity } from './asset-resolver.js';
import {
  appendSnapshotRecords,
  createSnapshotRecord,
  type PolymarketSnapshotRecord,
} from './polymarket-snapshots.js';
import {
  computeMaxHourlyJump,
  computeMaxHourlyLogitJump,
  computePriceVelocityPpH,
  computePriceVelocityLogitPerHour,
  fetchClobPriceHistory,
  fetchClobSpread,
} from './polymarket-clob.js';
import { hasEnv } from '../../utils/env.js';

// ---------------------------------------------------------------------------
// Description (injected into system prompt)
// ---------------------------------------------------------------------------

export const POLYMARKET_DESCRIPTION = `
Searches Polymarket prediction markets for crowd-sourced probability estimates on real-world events. Returns market-implied probabilities that reflect the collective wisdom of traders betting real money.

## When to Use

- Gauging crowd probability for macroeconomic events: Fed rate cuts, recessions, GDP outcomes
- Assessing geopolitical risk for a thesis: wars, elections, policy changes, trade tariffs
- Validating or stress-testing an investment hypothesis against market-implied odds
- Finding what events the crowd believes are most likely in a given time horizon
- Checking probability of a company-specific event: bankruptcy, acquisition, regulatory outcome
- Generating contrarian ideas when market prices diverge from prediction market probabilities

## When NOT to Use

- Real-time stock prices or financial metrics (use get_market_data or get_financials)
- Detailed company fundamentals (use get_financials)
- News or breaking events (use web_search)
- Prediction markets are not infallible — treat probabilities as one data point, not ground truth

## Query Guidelines (critical for good results)

The Polymarket search API uses simple text matching against market question titles.
Short, specific queries work far better than long compound strings.

**✅ Effective queries (short, topic-focused):**
- Company name only: \`"NVIDIA"\`, \`"Apple"\`, \`"Tesla"\`
- Company + topic: \`"NVIDIA earnings"\`, \`"Apple revenue"\`
- Macro topic: \`"Fed rate cut"\`, \`"US recession"\`, \`"FOMC"\`
- Event keyword: \`"tariff"\`, \`"oil price"\`, \`"FDA approval"\`
- Crypto: \`"Bitcoin ETF"\`, \`"crypto regulation"\`

**❌ Ineffective queries (too long, use ticker symbols, include years):**
- \`"NVDA earnings beat 2026"\` → use \`"NVIDIA earnings"\` instead
- \`"Fed rate cut Q2 2026"\` → use \`"Fed rate cut"\` instead
- \`"chip export controls 2026"\` → use \`"chip export controls"\` instead

**Key rules:**
- Use the company's full name, not the ticker symbol (\`"NVIDIA"\` not \`"NVDA"\`)
- Omit year/quarter suffixes — the API searches active markets, which are current
- 2–3 words is usually optimal; never more than 4
- Returns top markets sorted by 24h trading volume (most liquid = most reliable signal)
- YES price = implied probability (e.g. 0.72 = 72% chance the event happens)
- Combine with financial analysis: e.g. if recession probability is 35%, stress-test DCF with lower growth assumptions
`.trim();

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface PolymarketMarket {
  id: string;
  conditionId?: string;
  question: string;
  outcomes: string;
  outcomePrices: string;
  clobTokenIds?: string | string[];
  endDateIso?: string;
  /** Gamma API returns ISO date string for market creation time. */
  createdAt?: string;
  volume24hr?: number;
  volumeNum?: number;
  liquidityNum?: number;
  active: boolean;
  closed: boolean;
  enableOrderBook?: boolean;
  description?: string;
}

interface PolymarketEvent {
  id: string;
  title: string;
  endDate?: string;
  markets?: PolymarketMarket[];
  volume24hr?: number;
}

interface FormattedMarket {
  marketId: string;
  assetId?: string;
  question: string;
  probabilities: Record<string, string>;
  endDate: string | null;
  volume24h: string;
  liquidity: string;
  /** Days since the market was created (undefined if createdAt missing from API). */
  ageDays: number | undefined;
  active: boolean;
  closed: boolean;
  enableOrderBook?: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const GAMMA_BASE = 'https://gamma-api.polymarket.com';

// ---------------------------------------------------------------------------
// Retry with exponential backoff
// ---------------------------------------------------------------------------

/** Delays in ms between retry attempts (1s, 2s, 4s). */
export let RETRY_DELAYS = [1_000, 2_000, 4_000];

/** Override retry delays (for testing). Call with `[]` to disable waits. */
export function setRetryDelays(delays: number[]): void {
  RETRY_DELAYS = delays;
}

/**
 * Retries `fn` up to `maxRetries` times with exponential backoff.
 * Only retries on transient errors (network, 5xx, timeout).
 * 4xx errors are not retried.
 */
export async function fetchWithRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delays: number[] = RETRY_DELAYS,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      // Don't retry on 4xx client errors
      if (err instanceof Error && /\b4\d{2}\b/.test(err.message)) throw err;
      if (attempt < maxRetries) {
        const delay = delays[attempt] ?? 4_000;
        if (delay > 0) await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError;
}

// ---------------------------------------------------------------------------
// TTL cache for search results (5-minute default)
// ---------------------------------------------------------------------------

const CACHE_TTL_MS = 5 * 60 * 1_000; // 5 minutes
const CACHE_MAX_ENTRIES = 64;

interface CacheEntry<T> {
  data: T;
  expiresAt: number;
}

export interface FetchPolymarketMarketsOptions {
  snapshotFilePath?: string;
  capturedAt?: string;
  enrichMicrostructure?: boolean;
}

const searchCache = new Map<string, CacheEntry<FormattedMarket[]>>();

function cacheKey(query: string, limit: number): string {
  return `${query.toLowerCase().trim()}:${limit}`;
}

function getCached(key: string): FormattedMarket[] | undefined {
  const entry = searchCache.get(key);
  if (!entry) return undefined;
  if (Date.now() > entry.expiresAt) {
    searchCache.delete(key);
    return undefined;
  }
  return entry.data;
}

function setCache(key: string, data: FormattedMarket[]): void {
  // Evict oldest entries if over capacity
  if (searchCache.size >= CACHE_MAX_ENTRIES) {
    const oldest = searchCache.keys().next().value;
    if (oldest !== undefined) searchCache.delete(oldest);
  }
  searchCache.set(key, { data, expiresAt: Date.now() + CACHE_TTL_MS });
}

/** Clear the search cache. Exported for testing. */
export function clearPolymarketCache(): void {
  searchCache.clear();
}

// ---------------------------------------------------------------------------
// Degradation warnings collector
// ---------------------------------------------------------------------------

/** Collects non-fatal warnings during a search pass. */
let _searchWarnings: string[] = [];

/** Returns and clears accumulated search warnings. Exported for testing. */
export function drainSearchWarnings(): string[] {
  const w = _searchWarnings;
  _searchWarnings = [];
  return w;
}

// ---------------------------------------------------------------------------
// Market age computation
// ---------------------------------------------------------------------------

/** Compute days since creation from an ISO date string. Returns undefined if invalid. */
function computeAgeDays(createdAt: string | undefined): number | undefined {
  if (!createdAt) return undefined;
  const ms = Date.now() - new Date(createdAt).getTime();
  if (isNaN(ms) || ms < 0) return undefined;
  return Math.floor(ms / MS_PER_DAY);
}

// ---------------------------------------------------------------------------
// Client-side text filtering (API keyword param is unreliable)
// ---------------------------------------------------------------------------

const TEXT_FILTER_STOP_WORDS = new Set([
  'the', 'and', 'for', 'are', 'not', 'will', 'can', 'has', 'was',
  'how', 'what', 'that', 'this', 'its', 'from', 'with',
]);

const WEAK_QUERY_WORDS = new Set([
  'price', 'prices', 'market', 'markets', 'commodity', 'commodities',
  'forecast', 'forecasts', 'current', 'target', 'targets',
]);

/**
 * Returns true if the Polymarket question (or event title) contains at least
 * one significant word from the search query.
 *
 * The Gamma API `keyword` parameter is non-functional — it always returns
 * the highest-volume markets globally regardless of the query. This function
 * provides all client-side relevance filtering after every fetch.
 *
 * Exported for testing.
 */
export function questionMatchesQuery(text: string, query: string): boolean {
  const words = query
    .toLowerCase()
    .split(/[\s\-_/]+/)
    .map((w) => w.replace(/[^a-z0-9]/g, ''))
    .filter((w) => w.length >= 3 && !TEXT_FILTER_STOP_WORDS.has(w));

  if (words.length === 0) return true;

  const anchorWords = words.filter((word) => !WEAK_QUERY_WORDS.has(word));
  if (words.length > 0 && anchorWords.length === 0) return false;

  const candidateWords = anchorWords.length > 0 ? anchorWords : words;
  const lower = text.toLowerCase();
  return candidateWords.some((word) => lower.includes(word));
}

function extractCanonicalNames(ticker: string): string[] {
  const normalized = ticker.trim().toUpperCase().replace(/-USD$/, '');
  const known: Record<string, string[]> = {
    BTC: ['bitcoin', 'btc'],
    ETH: ['ethereum', 'eth'],
    SOL: ['solana', 'sol'],
    DOGE: ['dogecoin', 'doge'],
    XRP: ['ripple', 'xrp'],
    ADA: ['cardano', 'ada'],
  };
  return known[normalized] ?? resolveTickerSearchIdentity(normalized).canonicalNames;
}

const CRYPTO_BARRIER_PATTERN = /\b(?:reach|hit|surpass|touch)\b|\b(?:go|move)\s+to\b|\bgo\s+(?:above|below|over|under)\b|\b(?:remain|trade|stay)\s+(?:above|below|over|under)\b|\b(?:dip|drop|fall|sink|decline|decrease)s?\s+to\b/i;

// Month names for date-anchored trade pattern validation
const MONTH_NAMES = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'];

// Strict narrow exemption: only accept date-anchored trade phrases with actual dates
// Rejects non-date anchors like "at expiry", "at close", "at open"
const DATE_ANCHORED_TRADE_PATTERN = new RegExp(
  `\\btrade\\s+(?:above|below|over|under)\\b.*\\b(?:on|at)\\s+(?:\\d{1,2}[\\/\\-]|(?:${MONTH_NAMES.join('|')})\\b)`,
  'i'
);

export function scoreAnchorMarketRelevance(
  question: string,
  ticker: string,
  horizonDays?: number,
  endDate?: string | null,
): number {
  const lower = question.toLowerCase();
  let score = 0;
  const identity = resolveTickerSearchIdentity(ticker);
  const isCrypto = /(BTC|ETH|SOL|DOGE|XRP|ADA)[-\/]?USD/i.test(ticker)
    || /^(BTC|ETH|SOL|DOGE|XRP|ADA)$/i.test(ticker)
    || /CRYPTO/i.test(ticker);
  const hasDollarPrice = /\$[\d,]+(?:\.\d+)?(?:[KkMm])?/.test(question);

  if (hasDollarPrice) score += 3;
  if (/\b(above|below|less than|more than|greater than|under|over)\b/.test(lower)) score += 2;
  if (/\b(on|at)\b/.test(lower)) score += 1;

  const names = extractCanonicalNames(ticker);
  const hasCanonicalMatch = names.some((name) => lower.includes(name));
  if (identity.strictQuestionMatch && !hasCanonicalMatch) return 0;
  if (isCrypto && (!hasCanonicalMatch || !hasDollarPrice)) return 0;
  // Narrow exemption: accept date-anchored trade phrases, reject other barrier patterns
  if (isCrypto && CRYPTO_BARRIER_PATTERN.test(lower) && !DATE_ANCHORED_TRADE_PATTERN.test(lower)) return 0;
  if (hasCanonicalMatch) score += 2;

  if (/\b(reach|hit|dip to|between|up or down|first)\b/.test(lower)) score -= 3;
  if (/\b(election|sports|game|match|player|team|president|candidate)\b/.test(lower)) score -= 5;

  if (horizonDays != null && endDate) {
    const endMs = Date.parse(endDate);
    if (Number.isFinite(endMs)) {
      const daysUntilResolution = Math.abs((endMs - Date.now()) / MS_PER_DAY - horizonDays);
      score += Math.max(0, 4 - Math.min(4, daysUntilResolution));
    }
  }

  // Crypto-specific: barrier/path patterns ("reach $X", "dip to $Y") are
  // not terminal anchors — they describe a price path reaching/touching a
  // level, not the price being above/below at a specific future date.
  // Reject them outright so they do not enter the ranked candidate pool.
  if (isCrypto) {
    if (/\babove\b.*\$\d|\bbelow\b.*\$\d/.test(lower) && /\b(on|at)\b/.test(lower)) score += 2;
  }

  return score;
}

// ---------------------------------------------------------------------------
// Tag-slug map  (verified against live Gamma API — /events endpoint only)
// ---------------------------------------------------------------------------
//
// IMPORTANT: tag_slug ONLY works on the /events endpoint.
//            The /markets endpoint ignores it entirely.
//            The `keyword` parameter on BOTH endpoints is non-functional —
//            it always returns the same top-volume global markets.
//
// Slugs were validated by probing GET /events?tag_slug=X and confirming
// at least 3 relevant events are returned.

const TAG_SLUG_PATTERNS: Array<{ patterns: string[]; slugs: string[] }> = [
  { patterns: ['bitcoin', 'btc'],
    slugs: ['bitcoin', 'crypto-prices', 'crypto'] },
  { patterns: ['ethereum', 'eth'],
    slugs: ['ethereum', 'crypto-prices', 'crypto'] },
  { patterns: ['solana', 'sol', 'crypto', 'defi', 'nft', 'web3'],
    slugs: ['crypto-prices', 'crypto'] },
  { patterns: ['fed', 'fomc', 'federal reserve', 'rate cut', 'rate hike',
               'interest rate', 'basis point'],
    slugs: ['fed-rates', 'fed', 'economic-policy'] },
  { patterns: ['recession', 'gdp', 'inflation', 'cpi', 'unemployment', 'economic'],
    slugs: ['economy', 'business', 'economic-policy'] },
  { patterns: ['tariff', 'trade war', 'trade deal', 'import duty'],
    slugs: ['tariffs', 'politics', 'world'] },
  { patterns: ['oil', 'opec', 'crude', 'energy', 'wti', 'brent', 'petroleum'],
    slugs: ['commodities', 'world', 'business'] },
  { patterns: ['gold', 'silver', 'copper', 'platinum', 'palladium', 'precious metal', 'metal'],
    slugs: ['commodities', 'business'] },
  { patterns: ['wheat', 'corn', 'soybean', 'coffee', 'sugar', 'grain', 'natural gas'],
    slugs: ['commodities'] },
  { patterns: ['fda', 'drug approval', 'clinical trial', 'pharma', 'pfizer', 'moderna', 'eli lilly'],
    slugs: ['science', 'health'] },
  { patterns: ['nvidia', 'apple', 'microsoft', 'google', 'amazon', 'meta',
               'tesla', 'broadcom', 'qualcomm', 'intel', 'spacex'],
    slugs: ['big-tech', 'tech', 'business'] },
  { patterns: ['earnings', 'revenue', 'eps', 'quarterly results'],
    slugs: ['business', 'finance'] },
  { patterns: ['ai regulation', 'artificial intelligence', 'chatgpt', 'openai', 'antitrust'],
    slugs: ['tech', 'science'] },
  { patterns: ['middle east', 'ukraine', 'russia', 'china', 'taiwan', 'war', 'conflict', 'sanctions', 'geopolitical'],
    slugs: ['world', 'politics'] },
  { patterns: ['election', 'president', 'senate', 'congress', 'trump', 'white house'],
    slugs: ['elections', 'us-politics', 'politics'] },
  { patterns: ['ipo', 'initial public offering'],
    slugs: ['ipos', 'ipo', 'business'] },
];

/** Returns an ordered list of tag slugs to try for the given query. Exported for testing. */
export function inferTagSlugs(query: string): string[] {
  const lower = query.toLowerCase();
  for (const { patterns, slugs } of TAG_SLUG_PATTERNS) {
    if (patterns.some((p) => lower.includes(p))) return slugs;
  }
  return [];
}

// ---------------------------------------------------------------------------
// Core fetch helpers
// ---------------------------------------------------------------------------

function parseStringArrayField(raw: string | readonly string[] | undefined): string[] {
  const parsed = typeof raw === 'string'
    ? (() => {
        try {
          return JSON.parse(raw) as unknown;
        } catch {
          return [];
        }
      })()
    : raw;

  return Array.isArray(parsed)
    ? parsed
        .filter((item): item is string | number => typeof item === 'string' || typeof item === 'number')
        .map((item) => String(item))
    : [];
}

function formatVolume(n: number | undefined): string {
  if (!n) return '$0';
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${Math.round(n)}`;
}

function formatMarket(m: PolymarketMarket): FormattedMarket | null {
  const outcomes = parseStringArrayField(m.outcomes);
  const prices = parseStringArrayField(m.outcomePrices);
  const clobTokenIds = parseStringArrayField(m.clobTokenIds);
  if (!outcomes.length || !prices.length) return null;

  const probabilities: Record<string, string> = {};
  outcomes.forEach((outcome, i) => {
    const pct = parseFloat(prices[i] ?? '0') * 100;
    probabilities[outcome] = `${pct.toFixed(1)}%`;
  });

  return {
    marketId: m.conditionId ?? m.id,
    assetId: clobTokenIds.find((tokenId) => typeof tokenId === 'string' && tokenId.trim().length > 0),
    question: m.question,
    probabilities,
    endDate: m.endDateIso ?? null,
    volume24h: formatVolume(m.volume24hr),
    liquidity: formatVolume(m.liquidityNum),
    ageDays: computeAgeDays(m.createdAt),
    active: m.active,
    closed: m.closed,
    enableOrderBook: m.enableOrderBook,
  };
}

/**
 * Fetches events from the Gamma API using a specific tag slug and applies
 * client-side text filtering. This is the ONLY reliable search method —
 * the `keyword` param on both /events and /markets endpoints is non-functional.
 *
 * @param tagSlug   - Verified Polymarket tag slug (e.g. 'commodities', 'fed-rates')
 * @param textFilter - Query string used for client-side relevance filtering
 * @param limit     - Max markets to return after filtering
 */
async function searchEventsByTag(
  tagSlug: string,
  textFilter: string,
  limit: number,
): Promise<FormattedMarket[]> {
  try {
    return await fetchWithRetry(async () => {
      const params = new URLSearchParams({
        limit: String(Math.min(limit * 8, 80)), // fetch wide — text filter reduces count
        active: 'true',
        closed: 'false',
        order: 'volume24hr',
        ascending: 'false',
        tag_slug: tagSlug,
      });
      const res = await fetch(`${GAMMA_BASE}/events?${params}`, {
        headers: { 'Accept': 'application/json' },
        signal: AbortSignal.timeout(12_000),
      });
      if (!res.ok) throw new Error(`Gamma API ${res.status} for tag_slug=${tagSlug}`);
      const events: PolymarketEvent[] = await res.json() as PolymarketEvent[];

      const results: FormattedMarket[] = [];
      for (const event of events) {
        if (!event.markets?.length) continue;
        const titleMatches = questionMatchesQuery(event.title ?? '', textFilter);
        const sorted = [...event.markets]
          .filter((m) => m.active && !m.closed)
          .sort((a, b) => (b.volume24hr ?? 0) - (a.volume24hr ?? 0))
          .slice(0, 4); // up to 4 markets per event
        for (const m of sorted) {
          if (!titleMatches && !questionMatchesQuery(m.question, textFilter)) continue;
          const fmt = formatMarket(m);
          if (fmt) results.push(fmt);
          if (results.length >= limit) return results;
        }
      }
      return results;
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    _searchWarnings.push(`Tag "${tagSlug}" fetch failed after retries: ${msg}`);
    return [];
  }
}

/**
 * Primary search function. Uses tag slugs + client-side text filtering.
 * Results are cached for CACHE_TTL_MS (5 min) to avoid redundant API calls.
 *
 * Strategy:
 *  1. Check TTL cache → return immediately if fresh
 *  2. Infer tag slugs from query → fetch /events?tag_slug=X (works reliably)
 *  3. If tags infer 0 results or no tags found, do a broad top-events fetch
 *     and rely entirely on the client-side text filter.
 */
async function searchEvents(query: string, limit: number): Promise<FormattedMarket[]> {
  const key = cacheKey(query, limit);
  const cached = getCached(key);
  if (cached) return cached;

  // Reset warnings for this search pass
  _searchWarnings = [];

  const slugs = inferTagSlugs(query);

  if (slugs.length > 0) {
    // Fetch all tag buckets in parallel, merge, text-filter, deduplicate
    const settled = await Promise.allSettled(
      slugs.map((slug) => searchEventsByTag(slug, query, limit)),
    );

    const seen = new Set<string>();
    const combined: FormattedMarket[] = [];
    for (const r of settled) {
      if (r.status !== 'fulfilled') continue;
      for (const m of r.value) {
        if (!seen.has(m.question)) { seen.add(m.question); combined.push(m); }
        if (combined.length >= limit) break;
      }
      if (combined.length >= limit) break;
    }
    if (combined.length > 0) {
      setCache(key, combined);
      return combined;
    }
  }

  // Broad fallback: no tags inferred or tags returned 0 results.
  // Fetch top events globally and rely on text filter.
  try {
    const results = await fetchWithRetry(async () => {
      const params = new URLSearchParams({
        limit: String(limit * 10),
        active: 'true',
        closed: 'false',
        order: 'volume24hr',
        ascending: 'false',
      });
      const res = await fetch(`${GAMMA_BASE}/events?${params}`, {
        headers: { 'Accept': 'application/json' },
        signal: AbortSignal.timeout(15_000),
      });
      if (!res.ok) {
        polymarketBreaker.onFailure();
        throw new Error(`Polymarket API error: ${res.status}`);
      }
      polymarketBreaker.onSuccess();

      const events: PolymarketEvent[] = await res.json() as PolymarketEvent[];
      const out: FormattedMarket[] = [];
      const seen = new Set<string>();
      for (const event of events) {
        if (!event.markets?.length) continue;
        const titleMatches = questionMatchesQuery(event.title ?? '', query);
        const sorted = [...event.markets]
          .filter((m) => m.active && !m.closed)
          .sort((a, b) => (b.volume24hr ?? 0) - (a.volume24hr ?? 0))
          .slice(0, 2);
        for (const m of sorted) {
          if (!titleMatches && !questionMatchesQuery(m.question, query)) continue;
          const fmt = formatMarket(m);
          if (fmt && !seen.has(fmt.question)) {
            seen.add(fmt.question);
            out.push(fmt);
          }
          if (out.length >= limit) return out;
        }
      }
      return out;
    });

    if (results.length > 0) setCache(key, results);
    return results;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    _searchWarnings.push(`Global fallback fetch failed: ${msg}`);
    if (slugs.length === 0) throw err; // only re-throw if we had no tag results at all
    return [];
  }
}

function formatResults(markets: FormattedMarket[], query: string, warnings: string[]): string {
  if (markets.length === 0) {
    return `No active Polymarket prediction markets found for "${query}". Try broader keywords.`;
  }

  const lines: string[] = [
    `Polymarket — prediction market probabilities for: "${query}"`,
    `(Prices reflect crowd-sourced odds from real-money traders)\n`,
  ];

  for (const m of markets) {
    lines.push(`▸ ${m.question}`);
    for (const [outcome, prob] of Object.entries(m.probabilities)) {
      lines.push(`    ${outcome}: ${prob}`);
    }
    const meta: string[] = [];
    if (m.endDate) meta.push(`expires ${m.endDate}`);
    if (m.volume24h !== '$0') meta.push(`24h vol ${m.volume24h}`);
    if (m.liquidity !== '$0') meta.push(`liquidity ${m.liquidity}`);
    if (m.ageDays !== undefined) meta.push(`age ${m.ageDays}d`);
    if (meta.length) lines.push(`    ${meta.join(' · ')}`);
    lines.push('');
  }

  if (warnings.length > 0) {
    lines.push('⚠️  Data quality warnings:');
    for (const w of warnings) lines.push(`  • ${w}`);
    lines.push('');
  }

  lines.push('Source: polymarket.com — probabilities are market-implied, not guaranteed.');
  return lines.join('\n').trim();
}

const anchorTrace = hasEnv('DEBUG_ANCHORS')
  ? (...args: unknown[]) => console.error('[ANCHOR_TRACE]', ...args)
  : (..._args: unknown[]) => {};

// ---------------------------------------------------------------------------
// Anchor-acquisition search with optional end-date filtering
// ---------------------------------------------------------------------------

/**
 * Search Polymarket events with an end-date window constraint.
 * Used exclusively by anchor acquisition to find markets resolving within
 * a target horizon. The Gamma API supports `end_date_min` / `end_date_max`
 * on the /events endpoint, which filters to events with end dates inside
 * the specified range.
 */
async function searchEventsForAnchors(
  query: string,
  limit: number,
  dateFilter?: { end_date_min: string; end_date_max: string },
): Promise<FormattedMarket[]> {
  const keySuffix = dateFilter ? `:${dateFilter.end_date_min}:${dateFilter.end_date_max}` : '';
  const key = `${query.toLowerCase().trim()}:${limit}${keySuffix}`;
  const cached = getCached(key);
  if (cached) return cached;

  const slugs = inferTagSlugs(query);
  const uniqueSlugs = Array.from(new Set(slugs));
  const fetchLimit = Math.min(limit * 8, 80);

  const fetchEvents = async (tagSlugOverride?: string, extraParams?: URLSearchParams): Promise<FormattedMarket[]> => {
    try {
      return await fetchWithRetry(async () => {
        const resolvedTagSlug = tagSlugOverride ?? uniqueSlugs[0];
        const params = new URLSearchParams({
          limit: String(fetchLimit),
          active: 'true',
          closed: 'false',
          order: 'volume24hr',
          ascending: 'false',
          ...(resolvedTagSlug ? { tag_slug: resolvedTagSlug } : {}),
        });
        if (extraParams) {
          for (const [k, v] of extraParams) params.set(k, v);
        }
        anchorTrace('search_events_for_anchors_request', {
          query,
          limit,
          slugs,
          params: Object.fromEntries(params.entries()),
        });
        const res = await fetch(`${GAMMA_BASE}/events?${params}`, {
          headers: { 'Accept': 'application/json' },
          signal: AbortSignal.timeout(12_000),
        });
        if (!res.ok) throw new Error(`Gamma API ${res.status}`);
        const events: PolymarketEvent[] = await res.json() as PolymarketEvent[];
        anchorTrace('search_events_for_anchors_response', {
          query,
          limit,
          eventCount: events.length,
          eventTitles: events.slice(0, 20).map((event) => ({
            title: event.title ?? null,
            marketCount: event.markets?.length ?? 0,
          })),
        });
        const results: FormattedMarket[] = [];
        const seen = new Set<string>();
        for (const event of events) {
          if (!event.markets?.length) continue;
          const titleMatches = questionMatchesQuery(event.title ?? '', query);
          const sorted = [...event.markets]
            .filter((m) => m.active && !m.closed)
            .sort((a, b) => (b.volume24hr ?? 0) - (a.volume24hr ?? 0))
            .slice(0, 4);
          for (const m of sorted) {
            if (!titleMatches && !questionMatchesQuery(m.question, query)) continue;
            const fmt = formatMarket(m);
            if (fmt && !seen.has(fmt.question)) {
              seen.add(fmt.question);
              results.push(fmt);
            }
            if (results.length >= limit) return results;
          }
        }
        anchorTrace('search_events_for_anchors_results', {
          query,
          limit,
          resultCount: results.length,
          results: results.map((market) => ({
            question: market.question,
            volume24h: market.volume24h,
            ageDays: market.ageDays ?? null,
            endDate: market.endDate ?? null,
          })),
        });
        return results;
      });
    } catch {
      return [];
    }
  };

  let results: FormattedMarket[] = [];

  if (dateFilter) {
    const dateParams = new URLSearchParams({
      end_date_min: dateFilter.end_date_min,
      end_date_max: dateFilter.end_date_max,
    });
    const slugsToTry = uniqueSlugs.length > 0 ? uniqueSlugs : [undefined];
    const seen = new Set<string>();
    const perSlugCounts: Array<{ tagSlug: string | null; resultCount: number }> = [];

    for (const tagSlug of slugsToTry) {
      const slugResults = await fetchEvents(tagSlug, dateParams);
      perSlugCounts.push({ tagSlug: tagSlug ?? null, resultCount: slugResults.length });
      for (const market of slugResults) {
        if (seen.has(market.question)) continue;
        seen.add(market.question);
        results.push(market);
        if (results.length >= limit) break;
      }
      if (results.length >= limit) break;
    }

    anchorTrace('search_events_for_anchors_date_filter_attempt', {
      query,
      limit,
      dateFilter,
      slugsTried: slugsToTry.map((tagSlug) => tagSlug ?? null),
      perSlugCounts,
      resultCount: results.length,
    });
  }

  if (results.length === 0 && !dateFilter) {
    anchorTrace('search_events_for_anchors_fallback', {
      query,
      limit,
      dateFilter: dateFilter ?? null,
      reason: dateFilter ? 'date_filter_empty' : 'no_date_filter',
    });
    results = await fetchEvents();
  }

  if (results.length > 0) setCache(key, results);
  return results;
}

export async function fetchPolymarketAnchorMarketsWithQueries(
  queries: string[],
  limit: number,
  options: {
    ticker: string;
    horizonDays?: number;
    endDateFilter?: { end_date_min: string; end_date_max: string };
    enrichMicrostructure?: boolean;
  },
): Promise<PolymarketMarketResult[]> {
  const dedupedQueries = Array.from(new Set(queries.filter(Boolean)));
  const seen = new Set<string>();
  const scored: Array<{ market: PolymarketMarketResult; score: number }> = [];

  const settled = await Promise.allSettled(
    dedupedQueries.map((query) => (
      options.endDateFilter
        ? searchEventsForAnchors(query, Math.max(limit * 3, 24), options.endDateFilter)
        : searchEvents(query, Math.max(limit * 3, 24))
    )),
  );

  settled.forEach((result) => {
    if (result.status !== 'fulfilled') return;
    for (const market of result.value.map(toStructuredMarketResult)) {
      if (seen.has(market.question)) continue;
      seen.add(market.question);
      const score = scoreAnchorMarketRelevance(
        market.question,
        options.ticker,
        options.horizonDays,
        market.endDate,
      );
      if (score > 0) scored.push({ market, score });
    }
  });

  const selected = scored
    .sort((a, b) => {
      if (a.score !== b.score) return b.score - a.score;
      return b.market.volume24h - a.market.volume24h;
    })
    .slice(0, limit)
    .map(({ market }) => market);

  return options.enrichMicrostructure
    ? enrichStructuredMarketsMicrostructure(selected)
    : selected;
}

// ---------------------------------------------------------------------------
// Structured fetch (for programmatic use by polymarket-injector)
// ---------------------------------------------------------------------------

/** Structured Polymarket market result — numeric values, suitable for the injector. */
export interface PolymarketMarketResult {
  marketId?: string;
  assetId?: string;
  question: string;
  /** YES probability [0, 1] */
  probability: number;
  /** 24-hour trading volume in USD */
  volume24h: number;
  /** Days since market was created (undefined if unavailable from API). */
  ageDays: number | undefined;
  endDate?: string | null;
  active?: boolean;
  closed?: boolean;
  enableOrderBook?: boolean;
  bidAskSpread?: number;
  priceVelocityPpH?: number;
  priceVelocityLogitPerHour?: number;
  maxHourlyJump?: number;
  maxHourlyLogitJump?: number;
}

function toStructuredMarketResult(m: FormattedMarket): PolymarketMarketResult {
  return {
    marketId: m.marketId,
    assetId: m.assetId,
    question: m.question,
    probability: parseYesProbability(m.probabilities),
    volume24h: parseVolumeStr(m.volume24h),
    ageDays: m.ageDays,
    endDate: m.endDate,
    active: m.active,
    closed: m.closed,
    enableOrderBook: m.enableOrderBook,
  };
}

async function enrichStructuredMarketMicrostructure(
  market: PolymarketMarketResult,
): Promise<PolymarketMarketResult> {
  const tokenId = typeof market.assetId === 'string' && market.assetId.trim().length > 0
    ? market.assetId
    : undefined;
  if (!tokenId) return market;

  const [bidAskSpread, history] = await Promise.all([
    market.enableOrderBook === false ? Promise.resolve(null) : fetchClobSpread(tokenId),
    fetchClobPriceHistory(tokenId, '1h'),
  ]);

  if (bidAskSpread === null && history.length < 2) return market;

  return {
    ...market,
    ...(bidAskSpread !== null ? { bidAskSpread } : {}),
    ...(history.length >= 2
      ? {
          priceVelocityPpH: computePriceVelocityPpH(history),
          priceVelocityLogitPerHour: computePriceVelocityLogitPerHour(history),
          maxHourlyJump: computeMaxHourlyJump(history),
          maxHourlyLogitJump: computeMaxHourlyLogitJump(history),
        }
      : {}),
  };
}

async function enrichStructuredMarketsMicrostructure(
  markets: PolymarketMarketResult[],
): Promise<PolymarketMarketResult[]> {
  const enriched = await Promise.allSettled(markets.map(enrichStructuredMarketMicrostructure));
  return enriched.map((result, index) => result.status === 'fulfilled' ? result.value : markets[index]!);
}

function parseYesProbability(probs: Record<string, string>): number {
  const key = Object.keys(probs).find((k) => k.toLowerCase() === 'yes') ?? Object.keys(probs)[0];
  if (!key) return 0.5;
  return Math.min(1, Math.max(0, parseFloat(probs[key].replace('%', '')) / 100));
}

function parseVolumeStr(s: string): number {
  const n = parseFloat(s.replace(/[$,]/g, ''));
  if (isNaN(n)) return 0;
  if (s.includes('M')) return n * 1_000_000;
  if (s.includes('K')) return n * 1_000;
  return n;
}

/**
 * Idea 2 — Curate Polymarket markets suitable for use as jump-diffusion
 * event sources.  A market qualifies when it satisfies all of:
 *
 *   1. Resolves before `horizonDate` (settlement falls inside the forecast).
 *   2. 24h volume ≥ `minVolume24h` (default $5,000) — proxy for liquidity.
 *   3. Age ≥ `minAgeDays` (default 2) — filters brand-new low-info markets.
 *   4. Probability strictly in (0, 1) — degenerate quotes give no information.
 *
 * The returned shape is intentionally minimal so callers can pair it with
 * `buildJumpEventSpec()` without leaking Polymarket-specific fields into
 * the trajectory module.
 *
 * Probabilities returned here are still in the **Q-measure** — apply the
 * Q→P transformation before computing the daily hazard rate.
 */
export interface JumpEventMarket {
  /** Polymarket market id or question slug (used as JumpEventSpec.id). */
  id: string;
  /** Q-measure YES probability in (0, 1). */
  probability: number;
  /** Days from today to settlement (≥ 1). */
  daysToSettlement: number;
  /** Original question text — handy for provenance/debug logs. */
  question: string;
  /** P2a — heuristic direction inferred from question wording. */
  jumpDirection: JumpDirection;
}

/** P2a — Direction of the implied jump if the YES outcome materialises. */
export type JumpDirection = 'up' | 'down' | 'unknown';

/**
 * Heuristic classifier — returns the *most likely* asset-price direction
 * if the prediction-market YES outcome resolves true.
 *
 * Polymarket questions about war, sanctions, recession, defaults,
 * crashes, etc. are downside catalysts. Questions about rate cuts,
 * trade deals, regulatory approvals, breakthroughs are upside catalysts.
 *
 * Keyword lists are intentionally short and high-precision; ambiguous
 * questions return 'unknown' so the MC engine falls back to the
 * direction-neutral prior.
 */
const _DOWN_KEYWORDS = [
  'crash', 'attack', 'war', 'invasion', 'recession', 'default',
  'collapse', 'bankrupt', 'fail', 'drop', 'sanction', 'plunge',
  'tariff', 'shutdown', 'crisis', 'sell-off', 'selloff', 'meltdown',
];
const _UP_KEYWORDS = [
  'rate cut', 'cut rates', 'reach', 'hit', 'breakthrough', 'approve',
  'approved', 'rally', 'surge', 'deal', 'agreement', 'signed',
  'announce', 'launch', 'merger', 'acquisition',
];

export function classifyJumpDirection(question: string): JumpDirection {
  const q = question.toLowerCase();
  const hasDown = _DOWN_KEYWORDS.some((kw) => q.includes(kw));
  const hasUp = _UP_KEYWORDS.some((kw) => q.includes(kw));
  if (hasDown && !hasUp) return 'down';
  if (hasUp && !hasDown) return 'up';
  return 'unknown';
}

export interface ExtractJumpEventOptions {
  /** End-of-forecast horizon date.  Markets resolving after this are dropped. */
  horizonDate: Date;
  /** Optional liquidity floor (USD).  Default 5,000. */
  minVolume24h?: number;
  /** Minimum market age in days.  Default 2. */
  minAgeDays?: number;
  /** Reference "now" — defaults to `new Date()`. */
  now?: Date;
}

export function extractJumpEventMarkets(
  markets: readonly PolymarketMarketResult[],
  options: ExtractJumpEventOptions,
): JumpEventMarket[] {
  const minVol = options.minVolume24h ?? 5_000;
  const minAge = options.minAgeDays ?? 2;
  const now = options.now ?? new Date();
  const horizonMs = options.horizonDate.getTime();
  const out: JumpEventMarket[] = [];
  for (const m of markets) {
    if (m.probability <= 0 || m.probability >= 1) continue;
    if (m.volume24h < minVol) continue;
    if (m.ageDays === undefined || m.ageDays < minAge) continue;
    if (!m.endDate) continue;
    const endMs = Date.parse(m.endDate);
    if (!Number.isFinite(endMs) || endMs > horizonMs) continue;
    // P1c — drop markets settling in <24h (too noisy, often whale-driven).
    const rawDaysToSettle = (endMs - now.getTime()) / MS_PER_DAY;
    if (rawDaysToSettle < 1) continue;
    const daysToSettlement = Math.max(1, Math.ceil(rawDaysToSettle));
    out.push({
      id: m.marketId ?? m.question,
      probability: m.probability,
      daysToSettlement,
      question: m.question,
      jumpDirection: classifyJumpDirection(m.question),
    });
  }
  return out;
}

/**
 * Fetches Polymarket markets for `query` and returns structured numeric results.
 * Used by `polymarket-injector` for pre-query context injection.
 * Uses tag-based search exclusively — the Gamma API keyword param is non-functional.
 */
export async function fetchPolymarketMarkets(
  query: string,
  limit: number,
  options: FetchPolymarketMarketsOptions = {},
): Promise<PolymarketMarketResult[]> {
  const markets = await searchEvents(query, limit);
  const structured = markets.map(toStructuredMarketResult);
  const enriched = options.enrichMicrostructure
    ? await enrichStructuredMarketsMicrostructure(structured)
    : structured;

  if (options.snapshotFilePath) {
    const snapshotRecords = enriched
      .map((market) => createSnapshotRecord(market, options.capturedAt))
      .filter((record): record is PolymarketSnapshotRecord => record !== null);

    try {
      appendSnapshotRecords(options.snapshotFilePath, snapshotRecords);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.warn(`[polymarket] Failed to write snapshot records: ${message}`);
    }
  }

  return enriched;
}

export async function fetchPolymarketAnchorMarkets(
  query: string,
  limit: number,
  options: {
    ticker: string;
    horizonDays?: number;
    endDateFilter?: { end_date_min: string; end_date_max: string };
    enrichMicrostructure?: boolean;
  },
): Promise<PolymarketMarketResult[]> {
  return fetchPolymarketAnchorMarketsWithQueries([query], limit, options);
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

/** Searches and analyzes Polymarket prediction markets. */
export const polymarketTool = new DynamicStructuredTool({
  name: 'polymarket_search',
  description: 'Search Polymarket prediction markets for crowd-sourced probability estimates on macro, geopolitical, and financial events.',
  schema: z.object({
    query: z.string().max(10_000).describe(
      'Natural language search query, e.g. "Fed rate cut 2026", "US recession", "tariffs", "OPEC oil production"',
    ),
    limit: z.number().optional().default(8).describe(
      'Max number of markets to return (default 8)',
    ),
  }),
  func: async ({ query, limit = 8 }, _config?: unknown) => {
    if (polymarketBreaker.isOpen()) {
      return formatToolResult({ error: 'Polymarket API is temporarily unavailable (circuit open). Try again in a few minutes.' });
    }
    try {
      const markets = await searchEvents(query, limit);
      const warnings = drainSearchWarnings();
      return formatToolResult({ result: formatResults(markets, query, warnings) });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return formatToolResult({ error: `Polymarket search failed: ${msg}` });
    }
  },
});
