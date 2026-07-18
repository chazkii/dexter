import { join } from 'node:path';

const DEXTER_DIR = '.dexter';

export function getDexterDir(): string {
  return DEXTER_DIR;
}

export function dexterPath(...segments: string[]): string {
  return join(getDexterDir(), ...segments);
}

/**
 * Path helper retained under its original name for tools ported from the
 * cramer-short fork (e.g. arbiter-replay). Resolves under the dexter dir.
 */
export function cramerShortPath(...segments: string[]): string {
  return dexterPath(...segments);
}

/** Cache directory used by the arbiter-replay tool, under the dexter dir. */
export function arbiterReplayCachePath(...segments: string[]): string {
  return dexterPath('cache', 'arbiter-replay', ...segments);
}
