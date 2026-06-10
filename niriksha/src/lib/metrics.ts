/**
 * Sum every sample of a Prometheus metric family from the raw
 * `/v1/admin/metrics` text exposition. Ignores `# TYPE`/`# HELP`
 * comment lines and only matches the exact metric name (optionally
 * followed by a label set `{...}` or whitespace).
 */
export function sumPromMetric(text: string, metric: string): number {
  let sum = 0
  for (const line of text.split('\n')) {
    if (line.startsWith('#') || !line.startsWith(metric)) continue
    const head = line.slice(metric.length)
    if (head && !head.startsWith(' ') && !head.startsWith('{')) continue
    const v = parseFloat(line.trim().split(/\s+/).pop() || '')
    if (!Number.isNaN(v)) sum += v
  }
  return sum
}
