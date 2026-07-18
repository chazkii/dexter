import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { getEnv } from '../../utils/env.js';

// NOTE: In cramer-short this tool also fed captured whale movements into a
// "replay cache" consumed by its forecasting engine (arbiter-replay). That
// engine is out of scope for dexter's data-source layer, so the replay hook
// defaults to a no-op here. The on-chain data fetch is unaffected.
type RawWhaleReplayRow = Record<string, unknown>;

export const ONCHAIN_CRYPTO_DESCRIPTION = `
Fetches on-chain and market intelligence metrics for cryptocurrencies from CoinGecko (free, no key). Returns market data, community sentiment, developer activity, and global market context. Use when the user asks about crypto fundamentals, on-chain health, developer activity, or market sentiment beyond just price.

## When to Use
- User asks about crypto fundamentals, on-chain health, or developer activity
- Analyzing whale movements, sentiment, community growth, or ecosystem health
- Comparing crypto projects beyond price (developer commits, community size)
- Global crypto market overview (BTC dominance, total market cap)

## Example Queries
- "What's the on-chain health of Ethereum?"
- "Is Bitcoin developer activity increasing?"
- "What is the BTC dominance right now?"
- "Compare ETH and SOL community/developer metrics"
- "Is crypto market sentiment bullish or bearish?"
`.trim();

/** Map of well-known tickers to CoinGecko IDs. */
export const TICKER_TO_COINGECKO_ID: Record<string, string> = {
  BTC: 'bitcoin',
  ETH: 'ethereum',
  SOL: 'solana',
  BNB: 'binancecoin',
  XRP: 'ripple',
  ADA: 'cardano',
  DOGE: 'dogecoin',
  AVAX: 'avalanche-2',
  MATIC: 'matic-network',
  LINK: 'chainlink',
};

/** Resolve a ticker symbol to a CoinGecko ID. */
export function resolveCoinGeckoId(ticker: string): string {
  const upper = ticker.trim().toUpperCase();
  return TICKER_TO_COINGECKO_ID[upper] ?? ticker.trim().toLowerCase();
}

/** Fetch whale movement data for a cryptocurrency. */
async function fetchWhaleData(
  ticker: string,
  currentPriceUsd: number | null,
): Promise<Record<string, unknown>> {
  const upper = ticker.trim().toUpperCase();

  // Prefer Whale Alert API when a key is configured (multi-chain)
  const whaleAlertKey = getEnv('WHALE_ALERT_API_KEY');
  if (whaleAlertKey) {
    try {
      const res = await fetch(
        `https://api.whale-alert.io/v1/transactions?api_key=${whaleAlertKey}&min_value=500000&currency=${upper}&limit=10`,
        { headers: { Accept: 'application/json' } },
      );
      if (res.ok) {
        const data = asRecord(await res.json());
        const transactions = asRecordArray(data.transactions);
        return {
          source: 'whale-alert',
          transactions,
          count: transactions.length,
        };
      }
    } catch {
      // fall through to free sources
    }
  }

  // Free fallback for BTC via blockchain.info mempool
  if (upper === 'BTC') {
    try {
      const res = await fetch(
        'https://blockchain.info/unconfirmed-transactions?format=json',
        { headers: { Accept: 'application/json' } },
      );
      if (!res.ok) {
        return { error: `Blockchain.info returned ${res.status}` };
      }
      const data = asRecord(await res.json());
      const txs = asRecordArray(data.txs);

      const thresholdBtc = 100;
      const whaleTxs = txs
        .map((tx) => {
          const totalOutput = asRecordArray(tx.out).reduce(
            (sum, o) => sum + (numberOrNull(o.value) ?? 0),
            0,
          );
          const btcValue = totalOutput / 1e8;
          return {
            hash: stringOrNull(tx.hash) ?? '',
            btc_amount: btcValue,
            usd_value: currentPriceUsd ? btcValue * currentPriceUsd : null,
            time: tx.time ?? null,
          };
        })
        .filter((tx: { btc_amount: number }) => tx.btc_amount >= thresholdBtc)
        .sort((a: { btc_amount: number }, b: { btc_amount: number }) => b.btc_amount - a.btc_amount)
        .slice(0, 5);

      const totalBtc = whaleTxs.reduce((sum: number, tx: { btc_amount: number }) => sum + tx.btc_amount, 0);

      return {
        source: 'blockchain.info-mempool',
        note: 'Data from unconfirmed (mempool) transactions. Use web_search for confirmed whale movements.',
        threshold_btc: thresholdBtc,
        whale_transactions_detected: whaleTxs.length,
        total_btc_moved: totalBtc,
        total_usd_moved: currentPriceUsd ? totalBtc * currentPriceUsd : null,
        recent_large_transactions: whaleTxs,
      };
    } catch (err) {
      return {
        error: 'Unable to fetch whale data from blockchain.info',
        details: err instanceof Error ? err.message : 'Unknown error',
      };
    }
  }

  return {
    note: `Whale tracking is currently available for BTC only (or configure WHALE_ALERT_API_KEY for multi-chain). For ${upper}, use web_search for whale alerts.`,
  };
}

const OnchainCryptoInputSchema = z.object({
  ticker: z.string().max(128).describe("Crypto ticker e.g. 'BTC', 'ETH', 'SOL'"),
  metrics: z
    .array(z.enum(['market', 'sentiment', 'developer', 'community', 'global', 'whale']))
    .default(['market', 'sentiment'])
    .describe(
      "Which on-chain/market metrics to fetch. Valid values: 'market', 'sentiment', 'developer', 'community', 'global', 'whale'. " +
      "Use 'whale' for whale movement / large transaction data. Do NOT use 'large_transactions' or 'exchange_flows'.",
    ),
});

type MetricCategory = 'market' | 'sentiment' | 'developer' | 'community' | 'global' | 'whale';

type OnchainCryptoDependencies = {
  recordReplayWhaleCapture?: (row: RawWhaleReplayRow) => void;
};

type JsonRecord = Record<string, unknown>;

function asRecord(value: unknown): JsonRecord {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonRecord
    : {};
}

function asRecordArray(value: unknown): JsonRecord[] {
  return Array.isArray(value) ? value.map(asRecord) : [];
}

function numberOrNull(value: unknown): number | null {
  return typeof value === 'number' ? value : null;
}

function stringOrNull(value: unknown): string | null {
  return typeof value === 'string' ? value : null;
}

function extractMarketMetrics(data: unknown): Record<string, unknown> {
  const record = asRecord(data);
  const md = asRecord(record.market_data);
  const athChangePercentage = asRecord(md.ath_change_percentage);
  const totalVolume = asRecord(md.total_volume);
  const currentPrice = asRecord(md.current_price);
  return {
    price_change_24h_pct: md.price_change_percentage_24h ?? null,
    price_change_7d_pct: md.price_change_percentage_7d ?? null,
    price_change_30d_pct: md.price_change_percentage_30d ?? null,
    market_cap_rank: record.market_cap_rank ?? null,
    ath_change_percentage: athChangePercentage.usd ?? null,
    total_volume_usd: totalVolume.usd ?? null,
    circulating_supply: md.circulating_supply ?? null,
    max_supply: md.max_supply ?? null,
    current_price_usd: currentPrice.usd ?? null,
  };
}

function extractSentimentMetrics(data: unknown): Record<string, unknown> {
  const record = asRecord(data);
  return {
    sentiment_votes_up_percentage: record.sentiment_votes_up_percentage ?? null,
    sentiment_votes_down_percentage: record.sentiment_votes_down_percentage ?? null,
    public_interest_score: record.public_interest_score ?? null,
    coingecko_score: record.coingecko_score ?? null,
  };
}

function extractDeveloperMetrics(data: unknown): Record<string, unknown> {
  const dd = asRecord(asRecord(data).developer_data);
  return {
    forks: dd.forks ?? null,
    stars: dd.stars ?? null,
    total_issues: dd.total_issues ?? null,
    closed_issues: dd.closed_issues ?? null,
    pull_requests_merged: dd.pull_requests_merged ?? null,
    commit_activity_4_weeks: dd.commit_activity_4_weeks ?? null,
  };
}

function extractCommunityMetrics(data: unknown): Record<string, unknown> {
  const cd = asRecord(asRecord(data).community_data);
  return {
    twitter_followers: cd.twitter_followers ?? null,
    reddit_subscribers: cd.reddit_subscribers ?? null,
    reddit_average_posts_48h: cd.reddit_average_posts_48h ?? null,
    telegram_channel_user_count: cd.telegram_channel_user_count ?? null,
  };
}

function extractGlobalMetrics(globalData: unknown): Record<string, unknown> {
  const d = asRecord(asRecord(globalData).data);
  const totalMarketCap = asRecord(d.total_market_cap);
  const totalVolume = asRecord(d.total_volume);
  const marketCapPercentage = asRecord(d.market_cap_percentage);
  return {
    total_market_cap_usd: totalMarketCap.usd ?? null,
    total_volume_24h_usd: totalVolume.usd ?? null,
    btc_dominance: marketCapPercentage.btc ?? null,
    eth_dominance: marketCapPercentage.eth ?? null,
    market_cap_change_24h_pct: d.market_cap_change_percentage_24h_usd ?? null,
    active_cryptocurrencies: d.active_cryptocurrencies ?? null,
  };
}

/**
 * Create the on-chain crypto LangChain tool with an injectable replay-cache
 * write hook. Defaults to a no-op (the forecasting replay cache is not part of
 * dexter's data-source layer).
 */
export function createGetOnchainCryptoTool(dependencies: OnchainCryptoDependencies = {}) {
  const recordReplayWhaleCapture = dependencies.recordReplayWhaleCapture ?? (() => {});

  /** Creates the on-chain crypto analytics tool. */
  return new DynamicStructuredTool({
    name: 'get_onchain_crypto',
    description:
      "Fetches on-chain and market intelligence metrics for cryptocurrencies from CoinGecko (free, no API key needed). " +
      "Supported metrics: 'market' (price, volume, supply), 'sentiment' (sentiment votes, public interest), " +
      "'developer' (commits, forks, issues), 'community' (twitter, reddit, telegram), " +
      "'global' (BTC dominance, total market cap), 'whale' (whale movements / large transactions). " +
      "Do NOT request 'large_transactions' or 'exchange_flows' — use 'whale' instead.",
    schema: OnchainCryptoInputSchema,
    func: async (input) => {
      const coinId = resolveCoinGeckoId(input.ticker);
      const metrics = input.metrics as MetricCategory[];
      const needsCoinData = metrics.some((m) => m !== 'global');
      const needsGlobal = metrics.includes('global');

      const result: Record<string, unknown> = {
        ticker: input.ticker.trim().toUpperCase(),
        coinGeckoId: coinId,
      };
      const sourceUrls: string[] = [];

      try {
        // Fetch coin data if any non-global metric is requested
        if (needsCoinData) {
          const coinUrl =
            `https://api.coingecko.com/api/v3/coins/${coinId}` +
            `?localization=false&tickers=false&market_data=true&community_data=true&developer_data=true`;
          sourceUrls.push(coinUrl);

          const res = await fetch(coinUrl, {
            headers: { Accept: 'application/json' },
          });

          if (res.status === 429) {
            return formatToolResult(
              { error: 'CoinGecko rate limit exceeded. Please retry in a few seconds.' },
              [],
            );
          }

          if (!res.ok) {
            throw new Error(`CoinGecko returned ${res.status} for coin ID "${coinId}"`);
          }

          const data = asRecord(await res.json());
          result.name = stringOrNull(data.name) ?? coinId;
          result.symbol = stringOrNull(data.symbol)?.toUpperCase() ?? input.ticker.toUpperCase();

          for (const metric of metrics) {
            switch (metric) {
              case 'market':
                result.market = extractMarketMetrics(data);
                break;
              case 'sentiment':
                result.sentiment = extractSentimentMetrics(data);
                break;
              case 'developer':
                result.developer = extractDeveloperMetrics(data);
                break;
              case 'community':
                result.community = extractCommunityMetrics(data);
                break;
              case 'whale': {
                const priceUsd = numberOrNull(
                  asRecord(asRecord(data.market_data).current_price).usd,
                );
                const whale = await fetchWhaleData(input.ticker, priceUsd);
                result.whale = whale;
                recordReplayWhaleCapture({ ticker: input.ticker, whale });
                break;
              }
            }
          }
        }

        // Fetch global market data
        if (needsGlobal) {
          const globalUrl = 'https://api.coingecko.com/api/v3/global';
          sourceUrls.push(globalUrl);

          const globalRes = await fetch(globalUrl, {
            headers: { Accept: 'application/json' },
          });

          if (globalRes.status === 429) {
            result.global = { error: 'CoinGecko rate limit exceeded for global endpoint.' };
          } else if (!globalRes.ok) {
            result.global = { error: `CoinGecko global endpoint returned ${globalRes.status}` };
          } else {
            const globalData: unknown = await globalRes.json();
            result.global = extractGlobalMetrics(globalData);
          }
        }

        return formatToolResult(result, sourceUrls);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        return formatToolResult(
          {
            error: `On-chain crypto data unavailable for ${input.ticker}: ${errorMessage}. Try web_search for "${input.ticker} crypto on-chain metrics".`,
            ticker: input.ticker.toUpperCase(),
            coinGeckoId: coinId,
          },
          [],
        );
      }
    },
  });
}

export const getOnchainCrypto = createGetOnchainCryptoTool();
