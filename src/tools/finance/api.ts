import { z } from 'zod';
import { readCache, writeCache, describeRequest } from '../../utils/cache.js';
import { logger } from '../../utils/logger.js';
import { withRetry, isRateLimitError } from '../../utils/retry.js';

const BASE_URL = 'https://api.financialdatasets.ai';

/**
 * Marker embedded in the thrown error when Financial Datasets returns HTTP 402
 * (the endpoint requires a paid plan). Downstream callers — notably the
 * get_financials router and the FMP/Yahoo fallback tools — check for this
 * string to decide whether to retry against an alternative data source.
 */
export const FINANCIAL_DATASETS_PREMIUM = 'FINANCIAL_DATASETS_PREMIUM_REQUIRED';

/** How long a single Financial Datasets request may run before aborting. */
const REQUEST_TIMEOUT_MS = 30_000;

export interface ApiResponse {
  data: Record<string, unknown>;
  url: string;
}

const FinancialDatasetsPriceSchema = z.object({
  close: z.number(),
}).passthrough();

const FinancialDatasetsPricesPayloadSchema = z.union([
  z.array(FinancialDatasetsPriceSchema),
  z.object({
    prices: z.array(FinancialDatasetsPriceSchema),
  }).passthrough(),
]);

export type FinancialDatasetsPrice = z.infer<typeof FinancialDatasetsPriceSchema>;

export class FinancialDatasetsPayloadValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'FinancialDatasetsPayloadValidationError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class FinancialDatasetsHttpError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly body?: string;

  constructor(status: number, statusText: string, body?: string) {
    const detail = `${status} ${statusText}`;
    const bodyDetail = body?.trim()
      ? ` — ${body.trim().replace(/\s+/g, ' ').slice(0, 500)}`
      : '';
    super(`[Financial Datasets API] request failed: ${detail}${bodyDetail}`);
    this.name = 'FinancialDatasetsHttpError';
    this.status = status;
    this.statusText = statusText;
    this.body = body;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export function parseFinancialDatasetsPricesPayload(data: unknown): FinancialDatasetsPrice[] {
  const parsed = FinancialDatasetsPricesPayloadSchema.safeParse(data);
  if (!parsed.success) {
    throw new FinancialDatasetsPayloadValidationError(
      `Malformed Financial Datasets prices payload: ${parsed.error.issues.map((issue) => issue.message).join('; ')}`,
    );
  }
  return Array.isArray(parsed.data) ? parsed.data : parsed.data.prices;
}

/**
 * Remove redundant fields from API payloads before they are returned to the LLM.
 * This reduces token usage while preserving the financial metrics needed for analysis.
 */
export function stripFieldsDeep(value: unknown, fields: readonly string[]): unknown {
  const fieldsToStrip = new Set(fields);

  function walk(node: unknown): unknown {
    if (Array.isArray(node)) {
      return node.map(walk);
    }

    if (!node || typeof node !== 'object') {
      return node;
    }

    const record = node as Record<string, unknown>;
    const cleaned: Record<string, unknown> = {};

    for (const [key, child] of Object.entries(record)) {
      if (fieldsToStrip.has(key)) {
        continue;
      }
      cleaned[key] = walk(child);
    }

    return cleaned;
  }

  return walk(value);
}

function getApiKey(): string {
  return process.env.FINANCIAL_DATASETS_API_KEY || '';
}

/**
 * Shared request execution: handles API key, retry/backoff on 429, a 30s
 * timeout, error handling, logging, and response parsing.
 *
 * On HTTP 402 the request throws an error tagged with FINANCIAL_DATASETS_PREMIUM
 * so callers can fall back to a free data source (FMP / Yahoo) instead.
 */
async function executeRequest(
  url: string,
  label: string,
  init: RequestInit,
): Promise<Record<string, unknown>> {
  const apiKey = getApiKey();

  if (!apiKey) {
    logger.warn(`[Financial Datasets API] call without key: ${label}`);
  }

  let response: Response;
  try {
    response = await withRetry(
      async () => {
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
        try {
          const res = await fetch(url, {
            ...init,
            headers: {
              'x-api-key': apiKey,
              ...init.headers,
            },
            signal: controller.signal,
          });
          // Throw so withRetry can catch and back off on 429.
          if (res.status === 429) throw new Error('429 rate limit');
          return res;
        } finally {
          clearTimeout(timeout);
        }
      },
      { maxAttempts: 4, shouldRetry: isRateLimitError },
    );
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      logger.error(`[Financial Datasets API] timeout: ${label}`);
      throw new Error(`[Financial Datasets API] request timed out after 30s: ${label}`);
    }
    const message = error instanceof Error ? error.message : String(error);
    logger.error(`[Financial Datasets API] network error: ${label} — ${message}`);
    throw new Error(`[Financial Datasets API] request failed for ${label}: ${message}`);
  }

  if (!response.ok) {
    const detail = `${response.status} ${response.statusText}`;
    const body = typeof response.text === 'function'
      ? await response.text().catch(() => '')
      : '';
    logger.error(`[Financial Datasets API] error: ${label} — ${detail}${body ? ` — ${body}` : ''}`);
    if (response.status === 402) {
      throw new Error(
        `${FINANCIAL_DATASETS_PREMIUM}: This endpoint requires a paid Financial Datasets plan. ` +
          'Upgrade at https://financialdatasets.ai or use a free data source (FMP / Yahoo) as a fallback.',
      );
    }
    throw new FinancialDatasetsHttpError(response.status, response.statusText, body);
  }

  const data = await response.json().catch(() => {
    const detail = `invalid JSON (${response.status} ${response.statusText})`;
    logger.error(`[Financial Datasets API] parse error: ${label} — ${detail}`);
    throw new Error(`[Financial Datasets API] request failed: ${detail}`);
  });

  return data as Record<string, unknown>;
}

export const api = {
  async get(
    endpoint: string,
    params: Record<string, string | number | string[] | undefined>,
    options?: { cacheable?: boolean; ttlMs?: number },
  ): Promise<ApiResponse> {
    const label = describeRequest(endpoint, params);

    // Check local cache first — avoids redundant network calls for immutable data
    if (options?.cacheable) {
      const cached = readCache(endpoint, params, options.ttlMs);
      if (cached) {
        return cached;
      }
    }

    const url = new URL(`${BASE_URL}${endpoint}`);

    // Add params to URL, handling arrays
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null) {
        if (Array.isArray(value)) {
          value.forEach((v) => url.searchParams.append(key, v));
        } else {
          url.searchParams.append(key, String(value));
        }
      }
    }

    const data = await executeRequest(url.toString(), label, {});

    // Persist for future requests when the caller marked the response as cacheable
    if (options?.cacheable) {
      writeCache(endpoint, params, data, url.toString());
    }

    return { data, url: url.toString() };
  },

  async post(
    endpoint: string,
    body: Record<string, unknown>,
  ): Promise<ApiResponse> {
    const label = `POST ${endpoint}`;
    const url = `${BASE_URL}${endpoint}`;

    const data = await executeRequest(url, label, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });

    return { data, url };
  },
};

/** @deprecated Use `api.get` instead */
export const callApi = api.get;
