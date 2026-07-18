import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { api } from './api.js';
import { formatToolResult } from '../types.js';
import { hasEnv } from '../../utils/env.js';
import { buildPortfolioRiskReport } from '../../utils/finance/portfolio-stats.js';

export const PORTFOLIO_RISK_DESCRIPTION = `
Computes portfolio risk metrics — VaR, CVaR (Expected Shortfall), Sharpe ratio,
annualised volatility, max drawdown, and correlation matrix — for a set of tickers.

When to use this tool:
- User asks about portfolio risk, risk metrics, or position sizing
- User wants to know VaR (Value at Risk) or drawdown for their watchlist
- User asks "how correlated are my holdings?" or "what is my Sharpe ratio?"
- User wants to assess concentration risk across positions
- A portfolio_risk skill step calls this tool

When NOT to use this tool:
- For a single stock's current price or fundamentals (use get_financials instead)
- For screening stocks by financial ratios (use stock_screener)

Inputs:
- tickers: list of ticker symbols to analyse
- watchlist_entries: watchlist entries supplied by the caller; pass these when the user asks about their watchlist
- lookback_days: historical window for price data (default 252 ≈ 1 trading year)
- confidence_level: VaR / CVaR confidence (default 0.95)
- risk_free_rate: annual risk-free rate used for Sharpe (default 0.05)

Output:
- Per-ticker: volatility, Sharpe, VaR, CVaR, maxDrawdown
- Correlation matrix across all tickers
- Equal-weighted portfolio-level aggregates
`.trim();

const PortfolioRiskInputSchema = z.object({
  tickers: z
    .array(z.string().max(128))
    .optional()
    .describe('Ticker symbols to analyse. Pass explicitly unless watchlist_entries is supplied.'),
  watchlist_entries: z
    .array(z.object({
      ticker: z.string().max(128),
      costBasis: z.number().optional(),
      shares: z.number().optional(),
      addedAt: z.string().max(64).optional(),
    }))
    .optional()
    .describe('Watchlist entries supplied by the CLI/controller layer. Used when tickers is omitted.'),
  lookback_days: z
    .number()
    .int()
    .min(20)
    .max(1260)
    .default(252)
    .describe('Number of trading days of history to pull (default 252 ≈ 1 year).'),
  confidence_level: z
    .number()
    .min(0.5)
    .max(0.999)
    .default(0.95)
    .describe('Confidence level for VaR / CVaR (default 0.95).'),
  risk_free_rate: z
    .number()
    .min(0)
    .max(1)
    .default(0.05)
    .describe('Annual risk-free rate used for Sharpe ratio (default 0.05 = 5%).'),
});

export interface PortfolioRiskWatchlistEntry {
  ticker: string;
  costBasis?: number;
  shares?: number;
  addedAt?: string;
}

export interface PortfolioRiskToolOptions {
  watchlistEntries?: PortfolioRiskWatchlistEntry[];
}

/** Return a YYYY-MM-DD date that is `days` calendar days in the past. */
function dateNDaysAgo(days: number): string {
  const d = new Date();
  // Buffer for weekends + holidays so we get enough trading days.
  d.setDate(d.getDate() - Math.ceil(days * 1.45));
  return d.toISOString().slice(0, 10);
}

function todayStr(): string {
  return new Date().toISOString().slice(0, 10);
}

function normalizeTickers(input: { tickers?: string[]; watchlist_entries?: PortfolioRiskWatchlistEntry[] }): string[] {
  const rawTickers = input.tickers?.length
    ? input.tickers
    : input.watchlist_entries?.map((entry) => entry.ticker) ?? [];

  return [...new Set(
    rawTickers
      .map((ticker) => ticker.trim().toUpperCase())
      .filter((ticker) => ticker.length > 0),
  )];
}

/**
 * Create the portfolio-risk LangChain tool with optional watchlist injection.
 */
export function createPortfolioRiskTool(options: PortfolioRiskToolOptions = {}): DynamicStructuredTool {
  /** Creates the portfolio risk analysis tool. */
  return new DynamicStructuredTool({
    name: 'portfolio_risk',
    description: PORTFOLIO_RISK_DESCRIPTION,
    schema: PortfolioRiskInputSchema,
    func: async (input) => {
      if (!hasEnv('FINANCIAL_DATASETS_API_KEY')) {
        return formatToolResult({
          error: 'FINANCIAL_DATASETS_API_KEY is not set. Portfolio risk analysis requires historical price data.',
        });
      }

      const effectiveInput = !input.tickers?.length && !input.watchlist_entries?.length && options.watchlistEntries?.length
        ? { ...input, watchlist_entries: options.watchlistEntries }
        : input;
      const tickers = normalizeTickers(effectiveInput);
      if (tickers.length === 0) {
        return formatToolResult({
          error: 'No tickers provided. Pass tickers explicitly, or pass watchlist_entries from the controller layer.',
        });
      }

      const startDate = dateNDaysAgo(effectiveInput.lookback_days);
      const endDate = todayStr();
      const sourceUrls: string[] = [];

      // Fetch close-price history for each ticker in parallel
      const pricesByTicker: Record<string, number[]> = {};
      const errors: string[] = [];

      await Promise.all(
        tickers.map(async (ticker) => {
          try {
            const { data, url } = await api.get('/prices/', {
              ticker,
              interval: 'day',
              start_date: startDate,
              end_date: endDate,
            });
            sourceUrls.push(url);
            const prices: number[] = ((data as { prices?: unknown[] }).prices ?? [])
              .map((p) => (p as { close: number }).close)
              .filter((v): v is number => typeof v === 'number' && isFinite(v));
            if (prices.length < 20) {
              errors.push(`${ticker}: insufficient price history (${prices.length} days)`);
              return;
            }
            pricesByTicker[ticker] = prices;
          } catch (err) {
            errors.push(
              `${ticker}: price fetch failed — ${err instanceof Error ? err.message : String(err)}`,
            );
          }
        }),
      );

      const validTickers = Object.keys(pricesByTicker);
      if (validTickers.length === 0) {
        return formatToolResult({ error: 'Could not fetch price data for any ticker.', errors });
      }

      // Equal weights
      const w = 1 / validTickers.length;
      const weights: Record<string, number> = Object.fromEntries(
        validTickers.map((t) => [t, w]),
      );

      const report = buildPortfolioRiskReport(
        pricesByTicker,
        weights,
        effectiveInput.confidence_level,
        effectiveInput.risk_free_rate,
      );

      const result = errors.length > 0 ? { ...report, warnings: errors } : report;
      return formatToolResult(result, sourceUrls);
    },
  });
}

/**
 * Default portfolio-risk tool using equal weights and optional watchlist tickers.
 */
export const portfolioRiskTool = createPortfolioRiskTool();
