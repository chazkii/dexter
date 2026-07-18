import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { getQuote, getFundamentals, getQuotes, getFundamentalsBatch } from './robinhood-client.js';

const RobinhoodQuoteInputSchema = z.object({
  ticker: z
    .union([
      z.string().max(128).describe("Stock ticker symbol, e.g. 'AAPL'."),
      z.array(z.string().max(128)).describe("List of stock ticker symbols for batch lookup, e.g. ['AAPL', 'MSFT']."),
    ])
    .describe('A single ticker or a list of tickers to fetch quotes for.'),
});

const RobinhoodFundamentalsInputSchema = z.object({
  ticker: z
    .union([
      z.string().max(128).describe("Stock ticker symbol, e.g. 'AAPL'."),
      z.array(z.string().max(128)).describe("List of stock ticker symbols for batch lookup, e.g. ['AAPL', 'MSFT']."),
    ])
    .describe('A single ticker or a list of tickers to fetch fundamentals for.'),
});

export function makeRobinhoodTools(
  quoteFn: typeof getQuote = getQuote,
  fundamentalsFn: typeof getFundamentals = getFundamentals,
  quotesBatchFn: typeof getQuotes = getQuotes,
  fundamentalsBatchFn: typeof getFundamentalsBatch = getFundamentalsBatch,
) {
  /** Fetches a Robinhood quote for a stock symbol. */
  const getRobinhoodQuote = new DynamicStructuredTool({
    name: 'get_robinhood_quote',
    description:
      "Fetches a real-time quote from Robinhood: last trade price, bid/ask, open/high/low/close, volume, and 52-week range. No API key required. " +
      'Use as a fallback when get_stock_price (Financial Datasets) fails. Accepts a single ticker or a list of tickers.',
    schema: RobinhoodQuoteInputSchema,
    func: async (input) => {
      const tickers = Array.isArray(input.ticker)
        ? input.ticker.map((t) => t.trim().toUpperCase())
        : [input.ticker.trim().toUpperCase()];

      if (tickers.length === 1) {
        const ticker = tickers[0];
        const quote = await quoteFn(ticker);
        if (!quote) {
          return formatToolResult(
            { error: `Robinhood has no quote data for ${ticker}.` },
            [],
          );
        }
        const data = {
          symbol: quote.symbol,
          lastTradePrice: quote.last_trade_price,
          bidPrice: quote.bid_price,
          askPrice: quote.ask_price,
          bidSize: quote.bid_size,
          askSize: quote.ask_size,
          volume: quote.volume,
          adjustedPreviousClose: quote.adjusted_previous_close,
          tradingHalted: quote.trading_halted,
          previousClose: quote.previous_close,
          high52Week: null,
          low52Week: null,
        };
        return formatToolResult(data, [`https://robinhood.com/stocks/${ticker}`]);
      }

      // Batch lookup
      const quotes = await quotesBatchFn(tickers);
      if (quotes.length === 0) {
        return formatToolResult(
          { error: `Robinhood has no quote data for any of: ${tickers.join(', ')}.` },
          [],
        );
      }

      const results = quotes.map((quote) => ({
        symbol: quote.symbol,
        lastTradePrice: quote.last_trade_price,
        bidPrice: quote.bid_price,
        askPrice: quote.ask_price,
        bidSize: quote.bid_size,
        askSize: quote.ask_size,
        volume: quote.volume,
        adjustedPreviousClose: quote.adjusted_previous_close,
        tradingHalted: quote.trading_halted,
        previousClose: quote.previous_close,
        high52Week: null,
        low52Week: null,
      }));

      const foundSymbols = new Set(quotes.map((q) => q.symbol));
      const missing = tickers.filter((t) => !foundSymbols.has(t));
      if (missing.length > 0) {
        return formatToolResult(
          { results, missing },
          [`https://robinhood.com`],
        );
      }

      return formatToolResult({ results }, [`https://robinhood.com`]);
    },
  });

  /** Fetches Robinhood fundamentals for a stock symbol. */
  const getRobinhoodFundamentals = new DynamicStructuredTool({
    name: 'get_robinhood_fundamentals',
    description:
      "Fetches basic fundamental metrics from Robinhood: market cap, P/E ratio, EPS, dividend yield, shares outstanding, and 52-week high/low. No API key required. " +
      'Use as a fallback for basic metrics when other sources fail. Accepts a single ticker or a list of tickers.',
    schema: RobinhoodFundamentalsInputSchema,
    func: async (input) => {
      const tickers = Array.isArray(input.ticker)
        ? input.ticker.map((t) => t.trim().toUpperCase())
        : [input.ticker.trim().toUpperCase()];

      if (tickers.length === 1) {
        const ticker = tickers[0];
        const fundamentals = await fundamentalsFn(ticker);
        if (!fundamentals) {
          return formatToolResult(
            { error: `Robinhood has no fundamentals data for ${ticker}.` },
            [],
          );
        }
        const data = {
          symbol: fundamentals.symbol,
          marketCap: fundamentals.market_cap,
          adjustedMarketCap: fundamentals.market_cap ? parseFloat(fundamentals.market_cap) : null,
          priceEarningsRatio: fundamentals.pe_ratio,
          earningsPerShare: fundamentals.earnings_per_share ?? null,
          dividendsYield: fundamentals.dividend_yield,
          dividendsPerShare: fundamentals.dividend_per_share ?? null,
          sharesOutstanding: fundamentals.shares_outstanding,
          high52Week: fundamentals.high_52_weeks,
          low52Week: fundamentals.low_52_weeks,
          volume: fundamentals.volume,
          averageVolume: fundamentals.average_volume,
          description: fundamentals.description,
        };
        return formatToolResult(data, [`https://robinhood.com/stocks/${ticker}`]);
      }

      // Batch lookup
      const fundamentals = await fundamentalsBatchFn(tickers);
      if (fundamentals.length === 0) {
        return formatToolResult(
          { error: `Robinhood has no fundamentals data for any of: ${tickers.join(', ')}.` },
          [],
        );
      }

      const results = fundamentals.map((f) => ({
        symbol: f.symbol,
        marketCap: f.market_cap,
        adjustedMarketCap: f.market_cap ? parseFloat(f.market_cap) : null,
        priceEarningsRatio: f.pe_ratio,
        earningsPerShare: f.earnings_per_share ?? null,
        dividendsYield: f.dividend_yield,
        dividendsPerShare: f.dividend_per_share ?? null,
        sharesOutstanding: f.shares_outstanding,
        high52Week: f.high_52_weeks,
        low52Week: f.low_52_weeks,
        volume: f.volume,
        averageVolume: f.average_volume,
        description: f.description,
      }));

      const foundSymbols = new Set(fundamentals.map((f) => f.symbol));
      const missing = tickers.filter((t) => !foundSymbols.has(t));
      if (missing.length > 0) {
        return formatToolResult(
          { results, missing },
          [`https://robinhood.com`],
        );
      }

      return formatToolResult({ results }, [`https://robinhood.com`]);
    },
  });

  return { getRobinhoodQuote, getRobinhoodFundamentals };
}

const _tools = makeRobinhoodTools();
export const getRobinhoodQuote = _tools.getRobinhoodQuote;
export const getRobinhoodFundamentals = _tools.getRobinhoodFundamentals;
