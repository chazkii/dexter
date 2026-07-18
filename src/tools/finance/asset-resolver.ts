export type ResolvedAssetClass = 'commodity_gold' | 'commodity_silver' | 'commodity_oil' | 'gold_miner' | 'ticker';

export interface ResolvedAssetIntent {
  rawQuery: string;
  rawTicker: string | null;
  resolvedTicker: string | null;
  assetClass: ResolvedAssetClass | null;
  displayName: string | null;
  proxyLabel: string | null;
  needsClarification: boolean;
}

export interface ResolvedTickerSearchIdentity {
  rawTicker: string;
  canonicalTicker: string;
  searchQuery: string;
  canonicalNames: string[];
  strictQuestionMatch: boolean;
}

const BARRICK_CONTEXT_RE = /\bbarrick(?:\s+gold)?\b|\bgold\s+(?:stock|equity|shares|earnings|revenue)\b|\$gold\b/i;
const GOLD_COMMODITY_RE = /\bgold\b|\bxauusd\b/i;
const SILVER_COMMODITY_RE = /\bsilver\b|\bxagusd\b/i;
const OIL_COMMODITY_RE = /\boil\b|\bcrude\b|\bwti\b|\bwticousd\b/i;
const GOLD_PROXY_TICKERS = new Set(['GLD', 'IAU', 'SGOL', 'XAUUSD']);
const SILVER_PROXY_TICKERS = new Set(['SLV', 'SIVR', 'XAGUSD', 'SILVER']);
const OIL_PROXY_TICKERS = new Set(['USO', 'BNO', 'OIL', 'WTICOUSD', 'CRUDE']);
const EXCLUSIVE_ASSET_ONLY_RE = /\b([A-Z]{2,5}(?:-USD)?|bitcoin|ethereum|solana|gold|silver|oil|crude)\s+only\b/gi;
const CRYPTO_QUOTE_PAIR_RE = /\b([A-Z][A-Z0-9]{1,9}(?:USDT|USDC|USD))\b/g;
const CRYPTO_QUOTE_ROOTS = new Set([
  'ADA', 'AVAX', 'BCH', 'BNB', 'BTC', 'DOGE', 'DOT', 'ETH', 'HYPE', 'LINK',
  'LTC', 'MATIC', 'PEPE', 'SOL', 'SUI', 'TRX', 'XRP',
]);

function normalizeExplicitTicker(explicitTicker?: string | null): string | null {
  const value = explicitTicker?.trim().toUpperCase();
  return value && value.length > 0 ? value : null;
}

function mapExclusiveAssetToken(token: string): string | null {
  const normalized = token.trim().toUpperCase();
  if (normalized === 'BITCOIN' || normalized === 'BTC-USD') return 'BTC';
  if (normalized === 'ETHEREUM' || normalized === 'ETH-USD') return 'ETH';
  if (normalized === 'SOLANA' || normalized === 'SOL-USD') return 'SOL';
  if (normalized === 'GOLD') return 'GOLD';
  if (normalized === 'SILVER') return 'SILVER';
  if (normalized === 'CRUDE') return 'CRUDE';
  if (normalized === 'GLD' || normalized === 'IAU' || normalized === 'SGOL' || normalized === 'XAUUSD') return 'GLD';
  if (normalized === 'SLV' || normalized === 'SIVR' || normalized === 'XAGUSD') return 'SLV';
  if (normalized === 'USO' || normalized === 'BNO' || normalized === 'WTICOUSD') return 'USO';
  if (normalized === 'OIL') return 'OIL';
  return /^[A-Z]{2,5}(?:-USD)?$/.test(normalized) ? normalized : null;
}

export function extractExclusiveAssetOverride(query: string): string | null {
  let lastMatch: { index: number; ticker: string } | null = null;

  for (const match of query.matchAll(EXCLUSIVE_ASSET_ONLY_RE)) {
    const token = match[1];
    if (!token) continue;
    const ticker = mapExclusiveAssetToken(token);
    if (!ticker) continue;
    const index = match.index ?? -1;
    if (!lastMatch || index >= lastMatch.index) {
      lastMatch = { index, ticker };
    }
  }

  return lastMatch?.ticker ?? null;
}

export function extractCryptoQuotePairTickers(query: string): string[] {
  const found = new Set<string>();

  for (const match of query.matchAll(CRYPTO_QUOTE_PAIR_RE)) {
    const token = match[1]?.toUpperCase();
    if (!token) continue;

    const root = token.replace(/(?:USDT|USDC|USD)$/, '');
    if (CRYPTO_QUOTE_ROOTS.has(root)) {
      found.add(token);
    }
  }

  return Array.from(found);
}

export function resolveTickerSearchIdentity(ticker: string): ResolvedTickerSearchIdentity {
  const normalized = normalizeExplicitTicker(ticker) ?? ticker.trim().toUpperCase();
  const bareTicker = normalized.replace(/-USD$/, '');

  if (GOLD_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawTicker: normalized,
      canonicalTicker: 'GLD',
      searchQuery: 'gold',
      canonicalNames: ['gold', 'gld'],
      strictQuestionMatch: false,
    };
  }

  if (SILVER_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawTicker: normalized,
      canonicalTicker: 'SLV',
      searchQuery: 'silver',
      canonicalNames: ['silver', 'slv'],
      strictQuestionMatch: false,
    };
  }

  if (OIL_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawTicker: normalized,
      canonicalTicker: 'USO',
      searchQuery: 'oil',
      canonicalNames: ['oil', 'uso'],
      strictQuestionMatch: false,
    };
  }

  if (bareTicker === 'GOLD') {
    return {
      rawTicker: normalized,
      canonicalTicker: 'GOLD',
      searchQuery: 'Barrick Gold',
      canonicalNames: ['barrick gold', 'barrick'],
      strictQuestionMatch: true,
    };
  }

  return {
    rawTicker: normalized,
    canonicalTicker: bareTicker,
    searchQuery: bareTicker,
    canonicalNames: [bareTicker.toLowerCase()],
    strictQuestionMatch: false,
  };
}

export function resolveAssetIntent(query: string, explicitTicker?: string | null): ResolvedAssetIntent {
  const normalizedTicker = normalizeExplicitTicker(extractExclusiveAssetOverride(query) ?? explicitTicker);
  const bareTicker = normalizedTicker?.replace(/-USD$/, '') ?? null;

  if (bareTicker && GOLD_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'GLD',
      assetClass: 'commodity_gold',
      displayName: 'Gold (GLD proxy)',
      proxyLabel: 'GLD',
      needsClarification: false,
    };
  }

  if (bareTicker && SILVER_PROXY_TICKERS.has(bareTicker)) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'SLV',
      assetClass: 'commodity_silver',
      displayName: 'Silver (SLV proxy)',
      proxyLabel: 'SLV',
      needsClarification: false,
    };
  }

  const hasBarrickContext = BARRICK_CONTEXT_RE.test(query);
  if (hasBarrickContext) {
    if (normalizedTicker && normalizedTicker !== 'GOLD') {
      return {
        rawQuery: query,
        rawTicker: normalizedTicker,
        resolvedTicker: normalizedTicker,
        assetClass: 'ticker',
        displayName: normalizedTicker,
        proxyLabel: null,
        needsClarification: false,
      };
    }

    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'GOLD',
      assetClass: 'gold_miner',
      displayName: 'Barrick Gold',
      proxyLabel: null,
      needsClarification: false,
    };
  }

  if ((bareTicker === 'GOLD' && !BARRICK_CONTEXT_RE.test(query)) || (!normalizedTicker && GOLD_COMMODITY_RE.test(query))) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'GLD',
      assetClass: 'commodity_gold',
      displayName: 'Gold (GLD proxy)',
      proxyLabel: 'GLD',
      needsClarification: false,
    };
  }

  if (bareTicker === 'SILVER' || bareTicker === 'XAGUSD' || SILVER_COMMODITY_RE.test(query)) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'SLV',
      assetClass: 'commodity_silver',
      displayName: 'Silver (SLV proxy)',
      proxyLabel: 'SLV',
      needsClarification: false,
    };
  }

  if (
    bareTicker === 'OIL'
    || bareTicker === 'USO'
    || bareTicker === 'WTICOUSD'
    || bareTicker === 'CRUDE'
    || OIL_COMMODITY_RE.test(query)
  ) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: 'USO',
      assetClass: 'commodity_oil',
      displayName: 'Oil (USO proxy)',
      proxyLabel: 'USO',
      needsClarification: false,
    };
  }

  if (normalizedTicker) {
    return {
      rawQuery: query,
      rawTicker: normalizedTicker,
      resolvedTicker: normalizedTicker,
      assetClass: 'ticker',
      displayName: normalizedTicker,
      proxyLabel: null,
      needsClarification: false,
    };
  }

  return {
    rawQuery: query,
    rawTicker: null,
    resolvedTicker: null,
    assetClass: null,
    displayName: null,
    proxyLabel: null,
    needsClarification: false,
  };
}

export function assertAssetConsistency(
  intent: ResolvedAssetIntent,
  toolName: string,
  ticker: string,
): void {
  const normalized = ticker.trim().toUpperCase();

  if (intent.assetClass === 'commodity_gold' && normalized === 'GOLD' && toolName !== 'get_stock_tickers') {
    throw new Error('Commodity gold intent cannot use Barrick GOLD ticker directly; use GLD proxy instead.');
  }

  if (intent.assetClass === 'gold_miner' && normalized === 'GLD') {
    throw new Error('Barrick gold equity intent cannot use GLD commodity proxy.');
  }

  if (intent.assetClass === 'commodity_silver' && normalized === 'SILVER' && toolName !== 'get_stock_tickers') {
    throw new Error('Commodity silver intent cannot use SILVER pseudo-ticker directly; use SLV proxy instead.');
  }

  if (intent.assetClass === 'commodity_oil' && normalized === 'OIL' && toolName !== 'get_stock_tickers') {
    throw new Error('Commodity oil intent cannot use OIL pseudo-ticker directly; use USO proxy instead.');
  }
}
