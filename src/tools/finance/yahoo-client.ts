/**
 * Direct Yahoo Finance HTTP client — replaces the unmaintained yahoo-finance2 package.
 *
 * Yahoo Finance v10 quoteSummary requires a crumb + session cookie obtained via a
 * preflight request to fc.yahoo.com and /v1/test/getcrumb. The crumb is cached
 * for CRUMB_TTL_MS and refreshed automatically on 401/403 responses.
 */
import { z } from 'zod';

const QUERY_HOST = 'https://query1.finance.yahoo.com';
const CRUMB_URL = `${QUERY_HOST}/v1/test/getcrumb`;
const CONSENT_URL = 'https://fc.yahoo.com';
const CRUMB_TTL_MS = 30 * 60 * 1000; // 30 minutes

const UA =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';

interface CrumbCache {
  crumb: string;
  cookie: string;
  fetchedAt: number;
}

let crumbCache: CrumbCache | null = null;

/** @internal Test-only: resets the in-memory crumb cache. */
export function _clearCrumbCache(): void {
  crumbCache = null;
}

async function acquireCrumb(): Promise<CrumbCache> {
  // Step 1: hit fc.yahoo.com to establish a session cookie (A1/A3 pair)
  const initRes = await fetch(CONSENT_URL, {
    headers: { 'User-Agent': UA, Accept: 'text/html' },
    redirect: 'follow',
  });
  const rawCookie = initRes.headers.get('set-cookie') ?? '';
  // Only keep name=value pairs, strip directives, join multiple cookies
  const cookie = rawCookie
    .split(',')
    .map((c) => c.split(';')[0].trim())
    .filter(Boolean)
    .join('; ');

  // Step 2: exchange the session cookie for a crumb
  const crumbRes = await fetch(CRUMB_URL, {
    headers: { 'User-Agent': UA, Accept: '*/*', Cookie: cookie },
  });
  if (!crumbRes.ok) {
    throw new Error(`Yahoo Finance crumb fetch failed: ${crumbRes.status}`);
  }
  const crumb = (await crumbRes.text()).trim();
  if (!crumb) throw new Error('Yahoo Finance returned empty crumb');

  return { crumb, cookie, fetchedAt: Date.now() };
}

async function getCrumb(): Promise<CrumbCache> {
  if (crumbCache && Date.now() - crumbCache.fetchedAt < CRUMB_TTL_MS) {
    return crumbCache;
  }
  crumbCache = await acquireCrumb();
  return crumbCache;
}

const YahooNullableNumberSchema = z.number().nullable();
const YahooNullableStringSchema = z.string().nullable();

const YahooFinancialDataSchema = z.object({
  targetHighPrice: YahooNullableNumberSchema,
  targetLowPrice: YahooNullableNumberSchema,
  targetMeanPrice: YahooNullableNumberSchema,
  targetMedianPrice: YahooNullableNumberSchema,
  recommendationMean: YahooNullableNumberSchema,
  recommendationKey: YahooNullableStringSchema,
  numberOfAnalystOpinions: YahooNullableNumberSchema,
}).passthrough();

const YahooIncomeStatementRecordSchema = z.object({
  endDate: z.union([z.string(), z.number(), z.date()]).nullable(),
  totalRevenue: YahooNullableNumberSchema,
  grossProfit: YahooNullableNumberSchema,
  operatingIncome: YahooNullableNumberSchema,
  netIncome: YahooNullableNumberSchema,
  ebit: YahooNullableNumberSchema,
}).passthrough();

const YahooRecommendationTrendRecordSchema = z.object({
  period: YahooNullableStringSchema,
  strongBuy: YahooNullableNumberSchema,
  buy: YahooNullableNumberSchema,
  hold: YahooNullableNumberSchema,
  sell: YahooNullableNumberSchema,
  strongSell: YahooNullableNumberSchema,
}).passthrough();

const YahooUpgradeDowngradeRecordSchema = z.object({
  epochGradeDate: z.union([z.string(), z.number()]).nullable(),
  firm: YahooNullableStringSchema,
  toGrade: YahooNullableStringSchema,
  fromGrade: YahooNullableStringSchema,
  action: YahooNullableStringSchema,
}).passthrough();

const YahooRecommendationTrendSchema = z.object({
  trend: z.array(YahooRecommendationTrendRecordSchema).min(1),
}).passthrough();

const YahooUpgradeDowngradeHistorySchema = z.object({
  history: z.array(YahooUpgradeDowngradeRecordSchema).min(1),
}).passthrough();

const YahooIncomeStatementHistorySchema = z.object({
  incomeStatementHistory: z.array(YahooIncomeStatementRecordSchema).min(1),
}).passthrough();

const YahooQuoteSummaryResultSchema = z.object({
  financialData: YahooFinancialDataSchema.optional(),
  recommendationTrend: YahooRecommendationTrendSchema.optional(),
  upgradeDowngradeHistory: YahooUpgradeDowngradeHistorySchema.optional(),
  incomeStatementHistory: YahooIncomeStatementHistorySchema.optional(),
}).passthrough();

const YahooQuoteSummaryEnvelopeSchema = z.object({
  quoteSummary: z.object({
    result: z.array(z.unknown()).nullable().optional(),
    error: z.object({
      message: z.string().optional(),
    }).passthrough().nullable().optional(),
  }).passthrough().optional(),
}).passthrough();

type QuoteSummaryResult = z.infer<typeof YahooQuoteSummaryResultSchema>;

function parseQuoteSummaryResultForModules(raw: unknown, modules: string[]): QuoteSummaryResult {
  const requested = new Set(modules);
  const schema = z.object({
    financialData: requested.has('financialData')
      ? YahooFinancialDataSchema
      : YahooFinancialDataSchema.optional(),
    recommendationTrend: requested.has('recommendationTrend')
      ? YahooRecommendationTrendSchema
      : YahooRecommendationTrendSchema.optional(),
    upgradeDowngradeHistory: requested.has('upgradeDowngradeHistory')
      ? YahooUpgradeDowngradeHistorySchema
      : YahooUpgradeDowngradeHistorySchema.optional(),
    incomeStatementHistory: requested.has('incomeStatementHistory')
      ? YahooIncomeStatementHistorySchema
      : YahooIncomeStatementHistorySchema.optional(),
  }).passthrough();

  return schema.parse(raw);
}

/**
 * Fetches Yahoo Finance quoteSummary for the given ticker and modules.
 * Implements the same signature as yahoo-finance2's `quoteSummary()` so it can
 * be used as a drop-in replacement via makeYahooTools().
 */
export async function quoteSummary(
  ticker: string,
  opts: { modules: string[] },
): Promise<QuoteSummaryResult> {
  const modules = opts.modules.join(',');

  const doFetch = async (session: CrumbCache): Promise<Response> => {
    const url =
      `${QUERY_HOST}/v10/finance/quoteSummary/${encodeURIComponent(ticker)}` +
      `?modules=${encodeURIComponent(modules)}` +
      `&crumb=${encodeURIComponent(session.crumb)}` +
      `&formatted=false&lang=en-US&region=US`;
    return fetch(url, {
      headers: { 'User-Agent': UA, Accept: 'application/json', Cookie: session.cookie },
      signal: AbortSignal.timeout(15_000),
    });
  };

  let session = await getCrumb();
  let res = await doFetch(session);

  // On auth failure, force a fresh crumb and retry once
  if (res.status === 401 || res.status === 403) {
    crumbCache = null;
    session = await getCrumb();
    res = await doFetch(session);
  }

  if (!res.ok) {
    throw new Error(`Yahoo Finance quoteSummary failed for ${ticker}: ${res.status}`);
  }

  const json = YahooQuoteSummaryEnvelopeSchema.parse(await res.json());

  const qs = json.quoteSummary;
  if (qs?.error) throw new Error(qs.error.message ?? 'Yahoo Finance error');
  if (!qs?.result?.[0]) throw new Error(`No data returned for ${ticker}`);

  return parseQuoteSummaryResultForModules(qs.result[0], opts.modules);
}
