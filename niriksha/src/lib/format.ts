/** Compact human number: 950 → "950", 3_200_000 → "3.2M". */
export function compactNumber(n: number): string {
  if (n < 1000) return String(n)
  const u = ['', 'K', 'M', 'B', 'T']
  let i = 0
  let v = n
  while (v >= 1000 && i < u.length - 1) {
    v /= 1000
    i++
  }
  return `${(Math.round(v * 10) / 10).toString()}${u[i]}`
}

/** Thousands-separated integer: 12345 → "12,345". */
export function formatInt(n: number): string {
  return n.toLocaleString('en-US')
}
