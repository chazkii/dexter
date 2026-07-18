import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { quoteSummary as directQuoteSummary } from './yahoo-client.js';

interface YahooFinancialData {
  targetHighPrice?: unknown;
  targetLowPrice?: unknown;
  targetMeanPrice?: unknown;
  targetMedianPrice?: unknown;
  recommendationMean?: unknown;
  recommendationKey?: unknown;
  numberOfAnalystOpinions?: unknown;
}

interface YahooIncomeStatementRecord {
  endDate?: unknown;
  totalRevenue?: unknown;
  grossProfit?: unknown;
  operatingIncome?: unknown;
  netIncome?: unknown;
  ebit?: unknown;
}

interface YahooQuoteSummary {
  financialData?: YahooFinancialData;
  recommendationTrend?: { trend?: unknown[] };
  upgradeDowngradeHistory?: { history?: unknown[] };
  incomeStatementHistory?: { incomeStatementHistory?: YahooIncomeStatementRecord[] };
}

export type QuoteSummaryFn = (ticker: string, opts: { modules: string[] }) => Promise<YahooQuoteSummary>;

const YAHOO_SOURCE_URL = (ticker: string) =>
  `https://finance.yahoo.com/quote/${ticker}/analysis`;

// ---------------------------------------------------------------------------
// makeYahooTools — factory for dependency injection (enables unit testing
// without module-level mocking)
// ---------------------------------------------------------------------------

/**
 * Build Yahoo Finance tools with an injectable quoteSummary implementation for tests.
 */
export function makeYahooTools(quoteSummary: QuoteSummaryFn) {
  /** Fetches Yahoo Finance analyst price targets. */
  const getYahooAnalystTargets = new DynamicStructuredTool({
    name: 'get_yahoo_analyst_targets',
    description:
      'Fetches analyst consensus price targets and recommendation ratings from Yahoo Finance. ' +
      'Returns targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, ' +
      'recommendationKey (buy/hold/sell), recommendationMean score, and numberOfAnalystOpinions. ' +
      'Covers international tickers (e.g. VWS.CO, AZN.L, SAP.DE) not available in other sources.',
    schema: z.object({
      ticker: z.string().max(128).describe(
        "Stock ticker symbol, including exchange suffix for international stocks (e.g. 'VWS.CO', 'AZN.L', 'SAP.DE', 'AAPL').",
      ),
    }),
    func: async (input) => {
      const ticker = input.ticker.trim();
      try {
        const result = await quoteSummary(ticker, { modules: ['financialData'] });
        const fd = result.financialData;
        if (!fd) {
          return formatToolResult({ error: `No financial data returned by Yahoo Finance for ${ticker}` }, []);
        }
        const data = {
          ticker,
          targetHighPrice: fd.targetHighPrice ?? null,
          targetLowPrice: fd.targetLowPrice ?? null,
          targetMeanPrice: fd.targetMeanPrice ?? null,
          targetMedianPrice: fd.targetMedianPrice ?? null,
          recommendationMean: fd.recommendationMean ?? null,
          recommendationKey: fd.recommendationKey ?? null,
          numberOfAnalystOpinions: fd.numberOfAnalystOpinions ?? null,
        };
        return formatToolResult(data, [YAHOO_SOURCE_URL(ticker)]);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return formatToolResult({ error: message }, []);
      }
    },
  });

  /** Fetches Yahoo Finance analyst recommendation trends. */
  const getYahooAnalystRecommendations = new DynamicStructuredTool({
    name: 'get_yahoo_analyst_recommendations',
    description:
      'Fetches analyst buy/sell/hold recommendation trend from Yahoo Finance. ' +
      'Returns monthly counts (strongBuy, buy, hold, sell, strongSell) for the current month ' +
      'and the prior 3 months. Covers international tickers.',
    schema: z.object({
      ticker: z.string().max(128).describe(
        "Stock ticker symbol, including exchange suffix for international stocks (e.g. 'VWS.CO', 'AZN.L').",
      ),
    }),
    func: async (input) => {
      const ticker = input.ticker.trim();
      try {
        const result = await quoteSummary(ticker, { modules: ['recommendationTrend'] });
        const trend = result.recommendationTrend?.trend;
        if (!trend || trend.length === 0) {
          return formatToolResult({ error: `No recommendation trend data returned by Yahoo Finance for ${ticker}` }, []);
        }
        return formatToolResult(trend, [YAHOO_SOURCE_URL(ticker)]);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return formatToolResult({ error: message }, []);
      }
    },
  });

  /** Fetches Yahoo Finance upgrade and downgrade history. */
  const getYahooUpgradeDowngradeHistory = new DynamicStructuredTool({
    name: 'get_yahoo_upgrade_downgrade_history',
    description:
      'Fetches recent analyst rating changes (upgrades, downgrades, reiterations) from Yahoo Finance. ' +
      'Returns firm name, toGrade, fromGrade, action, and date for the most recent analyst actions. ' +
      'Covers international tickers.',
    schema: z.object({
      ticker: z.string().max(128).describe(
        "Stock ticker symbol, including exchange suffix for international stocks (e.g. 'VWS.CO', 'AZN.L').",
      ),
    }),
    func: async (input) => {
      const ticker = input.ticker.trim();
      try {
        const result = await quoteSummary(ticker, { modules: ['upgradeDowngradeHistory'] });
        const history = result.upgradeDowngradeHistory?.history;
        if (!history || history.length === 0) {
          return formatToolResult({ error: `No upgrade/downgrade history returned by Yahoo Finance for ${ticker}` }, []);
        }
        // Return the 10 most recent entries to keep context size manageable
        return formatToolResult(history.slice(0, 10), [YAHOO_SOURCE_URL(ticker)]);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return formatToolResult({ error: message }, []);
      }
    },
  });

  /** Fetches Yahoo Finance income statements. */
  const getYahooIncomeStatements = new DynamicStructuredTool({
    name: 'get_yahoo_income_statements',
    description:
      'Fetches historical income statements from Yahoo Finance. Free, no API key required. ' +
      'Works for international tickers (e.g. VWS.CO, AZN.L, SAP.DE). ' +
      'Returns totalRevenue, netIncome, grossProfit, operatingIncome, ebit per annual period. ' +
      'Used as a fallback when Financial Modeling Prep is unavailable or requires a paid plan.',
    schema: z.object({
      ticker: z.string().max(128).describe(
        "Stock ticker symbol, including exchange suffix for international stocks (e.g. 'VWS.CO', 'AZN.L', 'AAPL').",
      ),
      limit: z.number().default(4).describe('Number of periods to return (default: 4).'),
    }),
    func: async (input) => {
      const ticker = input.ticker.trim();
      try {
        const result = await quoteSummary(ticker, { modules: ['incomeStatementHistory'] });
        const records = result.incomeStatementHistory?.incomeStatementHistory ?? [];

        const data: Record<string, unknown>[] = [];
        for (const r of records.slice(0, input.limit)) {
          const entry: Record<string, unknown> = { date: r.endDate };
          // Only include fields that carry real values for this ticker
          if (r.totalRevenue) entry.totalRevenue = r.totalRevenue;
          if (r.grossProfit) entry.grossProfit = r.grossProfit;
          if (r.operatingIncome) entry.operatingIncome = r.operatingIncome;
          if (r.netIncome !== null && r.netIncome !== undefined) entry.netIncome = r.netIncome;
          if (r.ebit) entry.ebit = r.ebit;
          // Keep record only if it has at least one meaningful metric
          if (entry.totalRevenue !== undefined || entry.netIncome !== undefined) data.push(entry);
        }

        if (data.length === 0) {
          return formatToolResult(
            { error: `No income statement data available for ${ticker} on Yahoo Finance.` },
            [],
          );
        }

        return formatToolResult(data, [`https://finance.yahoo.com/quote/${ticker}/financials`]);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return formatToolResult({ error: message }, []);
      }
    },
  });

  return { getYahooAnalystTargets, getYahooAnalystRecommendations, getYahooUpgradeDowngradeHistory, getYahooIncomeStatements };
}

const _tools = makeYahooTools(directQuoteSummary);

/**
 * Yahoo Finance analyst price-target fallback tool using the direct HTTP client.
 */
export const getYahooAnalystTargets = _tools.getYahooAnalystTargets;
/**
 * Yahoo Finance analyst recommendation fallback tool using the direct HTTP client.
 */
export const getYahooAnalystRecommendations = _tools.getYahooAnalystRecommendations;
/**
 * Yahoo Finance upgrade/downgrade history fallback tool using the direct HTTP client.
 */
export const getYahooUpgradeDowngradeHistory = _tools.getYahooUpgradeDowngradeHistory;
/**
 * Yahoo Finance income-statement fallback tool using the direct HTTP client.
 */
export const getYahooIncomeStatements = _tools.getYahooIncomeStatements;
