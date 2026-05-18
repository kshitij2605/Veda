import { clsx, type ClassValue } from 'clsx'

/**
 * `cn` — class-name merger. Thin wrapper over `clsx` so every component
 * can take a `className` override that wins over its own defaults.
 */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs)
}
