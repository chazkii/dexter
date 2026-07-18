/**
 * Robinhood API HTTP client — wraps the unofficial Robinhood Trade private REST API.
 *
 * IMPORTANT: This is a reverse-engineered private API with no official documentation
 * or ToS protection. Endpoints may break without notice. Use as a fallback only.
 *
 * Public endpoints (quotes, fundamentals) do not require authentication.
 * Base URL: https://api.robinhood.com
 *
 * Reference: github.com/sanko/Robinhood (reverse-engineered API docs)
 */

const RH_BASE_URL = 'https://api.robinhood.com';

export interface RobinhoodQuote {
  symbol: string;
  ask_price: string | null;
  bid_price: string | null;
  ask_size: number | null;
  bid_size: number | null;
  last_trade_price: string | null;
  last_trade_size: number | null;
  last_trade_condition: string | null;
  updated_at: string;
  previous_close: string | null;
  adjusted_previous_close: string | null;
  trading_halted: boolean;
  market_state: string | null;
  volume: number | null;
  // Optional fields present in some responses
  last_extended_hours_trade_price?: string | null;
  instrument?: string;
  instrument_id?: string;
  state?: string;
  has_traded?: boolean;
}

export interface RobinhoodFundamentals {
  symbol: string;
  open: string | null;
  high: string | null;
  low: string | null;
  volume: number | null;
  average_volume: number | null;
  average_volume_2_weeks?: number | null;
  average_volume_30_days?: number | null;
  high_52_weeks: string | null;
  low_52_weeks: string | null;
  market_cap: string | null;
  pe_ratio: string | null;
  earnings_per_share?: string | null;
  dividend_yield: number | null;
  dividend_per_share?: number | null;
  shares_outstanding: number | null;
  description: string | null;
  // Optional fields present in some responses
  pb_ratio?: string | null;
  sector?: string | null;
  industry?: string | null;
  num_employees?: number | null;
  year_founded?: number | null;
  ceo?: string | null;
  headquarters_city?: string | null;
  headquarters_state?: string | null;
  float?: number | null;
}

async function rhFetch<T>(path: string): Promise<T> {
  const url = `${RH_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);
  try {
    const res = await fetch(url, {
      signal: controller.signal,
      headers: {
        Accept: 'application/json',
        'User-Agent': 'Robinhood/8.15.0 (Android 11)',
      },
    });
    if (!res.ok) {
      throw new Error(`[Robinhood API] ${res.status} ${res.statusText}`);
    }
    return res.json() as Promise<T>;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('[Robinhood API] request timed out after 15s');
    }
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

/** Fetch a real-time quote for a single ticker. Returns null on failure. */
export async function getQuote(ticker: string): Promise<RobinhoodQuote | null> {
  try {
    const data = await rhFetch<RobinhoodQuote | { detail: string }>(
      `/quotes/${ticker.toUpperCase()}/`,
    );
    // Robinhood returns { detail: "Not found" } for unknown tickers
    if ('detail' in data) return null;
    return data;
  } catch {
    return null;
  }
}

/** Fetch real-time quotes for multiple tickers in a single request.
 *
 * Endpoint: GET /quotes/?symbols=TICK1,TICK2,...
 * Returns empty array on failure or if no tickers are found.
 */
export async function getQuotes(tickers: string[]): Promise<RobinhoodQuote[]> {
  if (tickers.length === 0) return [];
  const symbols = tickers.map((t) => t.toUpperCase()).join(',');
  try {
    const data = await rhFetch<{ results: RobinhoodQuote[] } | { detail: string }>(
      `/quotes/?symbols=${encodeURIComponent(symbols)}`,
    );
    if ('detail' in data) return [];
    return data.results;
  } catch {
    return [];
  }
}

/** Fetch fundamental metrics for a single ticker. Returns null on failure. */
export async function getFundamentals(ticker: string): Promise<RobinhoodFundamentals | null> {
  try {
    const data = await rhFetch<RobinhoodFundamentals | { detail: string }>(
      `/fundamentals/${ticker.toUpperCase()}/`,
    );
    if ('detail' in data) return null;
    return data;
  } catch {
    return null;
  }
}

/** Fetch fundamental metrics for multiple tickers in a single request.
 *
 * Endpoint: GET /fundamentals/?symbols=TICK1,TICK2,...
 * Returns empty array on failure or if no tickers are found.
 */
export async function getFundamentalsBatch(tickers: string[]): Promise<RobinhoodFundamentals[]> {
  if (tickers.length === 0) return [];
  const symbols = tickers.map((t) => t.toUpperCase()).join(',');
  try {
    const data = await rhFetch<{ results: RobinhoodFundamentals[] } | { detail: string }>(
      `/fundamentals/?symbols=${encodeURIComponent(symbols)}`,
    );
    if ('detail' in data) return [];
    return data.results;
  } catch {
    return [];
  }
}
