import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { cramerShortPath } from '../../utils/paths.js';

export interface PolymarketSnapshotRecord {
  marketId: string;
  question: string;
  probability: number;
  volume24h: number;
  endDate: string;
  capturedAt: string;
  /** P2c — optional best bid (YES side), [0, 1]. */
  bid?: number;
  /** P2c — optional best ask (YES side), [0, 1]. */
  ask?: number;
}

export const DEFAULT_POLYMARKET_SNAPSHOTS_PATH = cramerShortPath('polymarket-snapshots.jsonl');

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function normalizeTimestamp(value: number | string | Date): number {
  if (typeof value === 'number') return value;
  if (value instanceof Date) return value.getTime();
  return Date.parse(value);
}

export function parseSnapshotLine(rawLine: string): PolymarketSnapshotRecord | null {
  const trimmed = rawLine.trim();
  if (!trimmed) return null;

  try {
    const parsed = JSON.parse(trimmed) as Record<string, unknown>;
    const marketId = parsed['marketId'];
    const question = parsed['question'];
    const probability = parsed['probability'];
    const volume24h = parsed['volume24h'];
    const endDate = parsed['endDate'];
    const capturedAt = parsed['capturedAt'];

    if (typeof marketId !== 'string' || marketId.trim().length === 0) return null;
    if (typeof question !== 'string') return null;
    if (!isFiniteNumber(probability) || probability < 0 || probability > 1) return null;
    if (!isFiniteNumber(volume24h) || volume24h < 0) return null;
    if (typeof endDate !== 'string') return null;
    if (endDate !== '' && !Number.isFinite(Date.parse(endDate))) return null;
    if (typeof capturedAt !== 'string' || !Number.isFinite(Date.parse(capturedAt))) return null;

    // P2c — optional bid/ask fields. Out-of-range values are silently dropped
    // (we keep the record but omit the field) so legacy snapshots stay valid.
    const bidRaw = parsed['bid'];
    const askRaw = parsed['ask'];
    const bid = isFiniteNumber(bidRaw) && bidRaw >= 0 && bidRaw <= 1 ? bidRaw : undefined;
    const ask = isFiniteNumber(askRaw) && askRaw >= 0 && askRaw <= 1 ? askRaw : undefined;

    return {
      marketId,
      question,
      probability,
      volume24h,
      endDate,
      capturedAt,
      ...(bid !== undefined ? { bid } : {}),
      ...(ask !== undefined ? { ask } : {}),
    };
  } catch {
    return null;
  }
}

export function findSnapshotInWindow(
  records: PolymarketSnapshotRecord[],
  marketId: string,
  windowStart: number | string | Date,
  windowEnd: number | string | Date,
): PolymarketSnapshotRecord | undefined {
  const startMs = normalizeTimestamp(windowStart);
  const endMs = normalizeTimestamp(windowEnd);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs) || marketId.trim().length === 0) {
    return undefined;
  }

  let latest: PolymarketSnapshotRecord | undefined;
  let latestMs = Number.NEGATIVE_INFINITY;

  for (const record of records) {
    if (record.marketId !== marketId) continue;
    const capturedMs = Date.parse(record.capturedAt);
    if (!Number.isFinite(capturedMs)) continue;
    if (capturedMs < startMs || capturedMs > endMs) continue;
    if (capturedMs > latestMs) {
      latest = record;
      latestMs = capturedMs;
    }
  }

  return latest;
}

export function readSnapshotRecords(
  filePath: string = DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
  marketId?: string,
): PolymarketSnapshotRecord[] {
  if (!existsSync(filePath)) return [];

  const content = readFileSync(filePath, 'utf-8');
  if (!content.trim()) return [];

  const records: PolymarketSnapshotRecord[] = [];
  const lines = content.split(/\r?\n/);

  for (let index = 0; index < lines.length; index++) {
    const line = lines[index];
    if (!line || !line.trim()) continue;

    const parsed = parseSnapshotLine(line);
    if (!parsed) {
      console.warn(
        `[polymarket-snapshots] Skipping malformed snapshot line ${index + 1} in ${filePath}`,
      );
      continue;
    }

    if (marketId && parsed.marketId !== marketId) continue;
    records.push(parsed);
  }

  return records;
}

export function appendSnapshotRecord(
  filePath: string = DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
  record: PolymarketSnapshotRecord,
): void {
  mkdirSync(dirname(filePath), { recursive: true });
  appendFileSync(filePath, `${JSON.stringify(record)}\n`, 'utf-8');
}

export function appendSnapshotRecords(
  filePath: string = DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
  records: PolymarketSnapshotRecord[],
): void {
  if (records.length === 0) return;
  mkdirSync(dirname(filePath), { recursive: true });
  const payload = `${records.map((record) => JSON.stringify(record)).join('\n')}\n`;
  appendFileSync(filePath, payload, 'utf-8');
}

export function createSnapshotRecord(
  market: {
    marketId?: string;
    question: string;
    probability: number;
    volume24h: number;
    endDate?: string | null;
    bid?: number;
    ask?: number;
  },
  capturedAt: string = new Date().toISOString(),
): PolymarketSnapshotRecord | null {
  if (!market.marketId) return null;

  const bid =
    typeof market.bid === 'number' && Number.isFinite(market.bid) && market.bid >= 0 && market.bid <= 1
      ? market.bid
      : undefined;
  const ask =
    typeof market.ask === 'number' && Number.isFinite(market.ask) && market.ask >= 0 && market.ask <= 1
      ? market.ask
      : undefined;

  return {
    marketId: market.marketId,
    question: market.question,
    probability: market.probability,
    volume24h: market.volume24h,
    endDate: market.endDate ?? '',
    capturedAt,
    ...(bid !== undefined ? { bid } : {}),
    ...(ask !== undefined ? { ask } : {}),
  };
}

/**
 * P2c — Build an in-memory index `marketId → sorted records (oldest→newest)`.
 *
 * Replaces the previous O(n·markets) linear scan in `findSnapshotInWindow`
 * with O(log n) lookups when callers materialise the index once.
 *
 * Records with unparseable `capturedAt` are skipped.
 */
export function buildSnapshotIndex(
  records: PolymarketSnapshotRecord[],
): Map<string, PolymarketSnapshotRecord[]> {
  const index = new Map<string, PolymarketSnapshotRecord[]>();
  for (const r of records) {
    const ms = Date.parse(r.capturedAt);
    if (!Number.isFinite(ms)) continue;
    const list = index.get(r.marketId);
    if (list) list.push(r);
    else index.set(r.marketId, [r]);
  }
  for (const list of index.values()) {
    list.sort((a, b) => Date.parse(a.capturedAt) - Date.parse(b.capturedAt));
  }
  return index;
}

/**
 * P2c — Keep only the `n` most-recent snapshots per `marketId`.
 *
 * Used to compact `polymarket-snapshots.jsonl` so the file does not grow
 * unboundedly. Default `n = 3` (sufficient for 6h + 24h + 48h window probes).
 *
 * Records with unparseable timestamps are dropped. Output is unsorted across
 * marketIds (callers can sort if they need a stable on-disk format).
 */
export function pruneSnapshots(
  records: PolymarketSnapshotRecord[],
  n: number = 3,
): PolymarketSnapshotRecord[] {
  if (records.length === 0 || n <= 0) return [];
  const index = buildSnapshotIndex(records);
  const out: PolymarketSnapshotRecord[] = [];
  for (const list of index.values()) {
    // list is already sorted oldest→newest; take the last `n`.
    const start = Math.max(0, list.length - n);
    for (let i = start; i < list.length; i++) out.push(list[i]);
  }
  return out;
}
