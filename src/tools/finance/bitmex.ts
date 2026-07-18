import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';

const BITMEX_BASE = 'https://www.bitmex.com';

type BitmexInstrument = {
  symbol: string;
  rootSymbol?: string;
  underlying?: string;
  state?: string;
  markPrice?: number;
  lastPrice?: number;
  bidPrice?: number;
  askPrice?: number;
  volume24h?: number;
  foreignNotional24h?: number;
  homeNotional24h?: number;
  turnover24h?: number;
  initMargin?: number;
  fundingRate?: number;
  indicativeFundingRate?: number;
};

type BitmexBucketRow = {
  close?: number;
};

function toBitmexRoot(ticker: string): string {
  const normalized = ticker.trim().toUpperCase().replace('/', '');
  if (normalized === 'BTC' || normalized === 'BTCUSD' || normalized === 'BTCUSDT') return 'XBT';
  if (normalized === 'BTC-USD' || normalized === 'BTC-USDT') return 'XBT';
  if (normalized === 'XAUUSD' || normalized === 'XAU-USD') return 'XAU';
  if (normalized === 'XAGUSD' || normalized === 'XAG-USD') return 'XAG';
  return normalized
    .replace(/[-_]?USD[T]?$/, '')
    .replace(/[-_]?XBT$/, '');
}

function bitmexLiquidity(instrument: BitmexInstrument): number {
  for (const key of ['foreignNotional24h', 'homeNotional24h', 'turnover24h'] as const) {
    const value = Number(instrument[key]);
    if (Number.isFinite(value) && value > 0) return value;
  }
  const volume = Number(instrument.volume24h ?? 0);
  const price = Number(instrument.markPrice ?? instrument.lastPrice ?? 0);
  return Number.isFinite(volume * price) ? volume * price : 0;
}

function toBitmexInstrumentSummary(instrument: BitmexInstrument) {
  const markPrice = Number(instrument.markPrice ?? instrument.lastPrice ?? 0);
  const bidPrice = Number(instrument.bidPrice);
  const askPrice = Number(instrument.askPrice);
  const spreadPct = Number.isFinite(bidPrice) && Number.isFinite(askPrice) && markPrice > 0
    ? ((askPrice - bidPrice) / markPrice) * 100
    : null;
  const maxLeverage = Number(instrument.initMargin) > 0
    ? 1 / Number(instrument.initMargin)
    : null;

  return {
    symbol: instrument.symbol,
    rootSymbol: instrument.rootSymbol ?? null,
    underlying: instrument.underlying ?? null,
    state: instrument.state ?? null,
    markPrice: Number.isFinite(markPrice) && markPrice > 0 ? markPrice : null,
    lastPrice: instrument.lastPrice ?? null,
    bidPrice: Number.isFinite(bidPrice) ? bidPrice : null,
    askPrice: Number.isFinite(askPrice) ? askPrice : null,
    spreadPct,
    volume24h: instrument.volume24h ?? null,
    liquidity24h: bitmexLiquidity(instrument),
    maxLeverage,
    fundingRate: instrument.fundingRate ?? null,
    indicativeFundingRate: instrument.indicativeFundingRate ?? null,
  };
}

export function toBitmexSymbolCandidates(ticker: string): string[] {
  const root = toBitmexRoot(ticker);
  const candidates = [
    `${root}USDT`,
    `${root}USD`,
    `${root}_USDT`,
  ];
  if (root === 'XBT') {
    candidates.unshift('XBTUSD', 'XBTUSDT');
  }
  return [...new Set(candidates)];
}

export async function fetchBitmexActiveInstruments(): Promise<BitmexInstrument[]> {
  try {
    const res = await fetch(`${BITMEX_BASE}/api/v1/instrument/active`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return [];
    return await res.json() as BitmexInstrument[];
  } catch {
    return [];
  }
}

export function selectBitmexInstrument(
  instruments: BitmexInstrument[],
  ticker: string,
): BitmexInstrument | null {
  const root = toBitmexRoot(ticker);
  const candidates = new Set(toBitmexSymbolCandidates(ticker));
  const matching = instruments
    .filter((instrument) => instrument.state === undefined || instrument.state === 'Open')
    .filter((instrument) =>
      candidates.has(instrument.symbol)
      || instrument.rootSymbol?.toUpperCase() === root
      || instrument.underlying?.toUpperCase() === root
    )
    .filter((instrument) => Number.isFinite(Number(instrument.markPrice ?? instrument.lastPrice)));

  const [best] = matching.sort((a, b) => bitmexLiquidity(b) - bitmexLiquidity(a));
  return best ?? null;
}

export async function resolveBitmexHistoricalSymbol(ticker: string): Promise<string | null> {
  const instruments = await fetchBitmexActiveInstruments();
  return selectBitmexInstrument(instruments, ticker)?.symbol ?? null;
}

async function fetchBitmexDailyClosesBySymbol(
  symbol: string,
  days: number,
): Promise<number[]> {
  const params = new URLSearchParams({
    binSize: '1d',
    partial: 'false',
    symbol,
    count: String(Math.min(Math.max(days, 1), 750)),
    reverse: 'true',
  });

  try {
    const res = await fetch(`${BITMEX_BASE}/api/v1/trade/bucketed?${params}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return [];
    const rows = await res.json() as BitmexBucketRow[];
    return rows
      .slice()
      .reverse()
      .map((row) => Number(row.close))
      .filter((price) => Number.isFinite(price) && price > 0);
  } catch {
    return [];
  }
}

export async function fetchBitmexDailyCloses(
  ticker: string,
  days = 120,
): Promise<number[]> {
  const resolvedSymbol = await resolveBitmexHistoricalSymbol(ticker);
  const candidateSymbols = resolvedSymbol
    ? [resolvedSymbol, ...toBitmexSymbolCandidates(ticker).filter((symbol) => symbol !== resolvedSymbol)]
    : toBitmexSymbolCandidates(ticker);

  for (const symbol of candidateSymbols) {
    const closes = await fetchBitmexDailyClosesBySymbol(symbol, days);
    if (closes.length > 0) {
      return closes;
    }
  }

  return [];
}

export const BITMEX_MARKET_DESCRIPTION = `
Fetch active BitMEX instrument data and optional 1d bucketed historical closes.

Use this for BitMEX-specific trading setup work when you need:
- Active contract selection by ticker/root, e.g. SOLUSD, SOLUSDT, HYPEUSDT, BTC-USD/XBTUSD
- Current mark, bid, ask, spread, funding, max leverage, and 24h liquidity
- BitMEX bucketed daily closes for markov_distribution historicalPrices

For forecasting, pass the returned historicalCloses into markov_distribution and then use forecast_arbitrator.
`.trim();

/** Retrieves BitMEX market data for crypto derivatives. */
export const bitmexMarketTool = new DynamicStructuredTool({
  name: 'bitmex_market',
  description: BITMEX_MARKET_DESCRIPTION,
  schema: z.object({
    tickers: z.array(z.string().max(128)).min(1).describe('BitMEX symbols or asset tickers, e.g. ["SOLUSD", "HYPEUSDT", "BTC-USD"].'),
    days: z.number().int().min(1).max(750).optional().default(120).describe('Number of 1d bucketed closes to fetch for each resolved BitMEX instrument.'),
  }),
  func: async (input) => {
    const instruments = await fetchBitmexActiveInstruments();
    const results = await Promise.all(input.tickers.map(async (ticker) => {
      const instrument = selectBitmexInstrument(instruments, ticker);
      if (!instrument) {
        return {
          ticker,
          matched: false,
          error: `No active BitMEX instrument matched ${ticker}`,
        };
      }

      const historicalCloses = await fetchBitmexDailyClosesBySymbol(instrument.symbol, input.days);
      return {
        ticker,
        matched: true,
        instrument: toBitmexInstrumentSummary(instrument),
        historicalCloses,
        historicalDays: historicalCloses.length,
      };
    }));

    return formatToolResult({
      exchange: 'bitmex',
      results,
    }, [
      `${BITMEX_BASE}/api/v1/instrument/active`,
      `${BITMEX_BASE}/api/v1/trade/bucketed`,
    ]);
  },
});
