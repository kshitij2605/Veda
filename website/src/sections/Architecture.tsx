import { Check } from 'lucide-react'
import { Section } from '../components/Section'

const SERVICES: [string, string][] = [
  ['Veda', 'API, indexing, search — one Rust binary'],
  ['MinIO', 'S3-compatible blobs + metrics storage'],
  ['NATS', 'Sub-millisecond message queue, ~20MB'],
]

const INDEXES: [string, string, string][] = [
  ['HNSW', 'General-purpose, low-latency', 'Tunable M / ef'],
  ['DiskANN', 'Billion-scale beyond RAM', 'Vamana graph on SSD'],
  ['IVF-PQ', 'Memory-constrained throughput', 'PQ compresses 8–32×'],
  ['Flat', 'Small sets, 100% recall', 'Brute-force baseline'],
  ['RaBitQ', 'Ultra-fast screening', 'Binary + Hamming'],
  ['Sq4', '4× compression, strong recall', '4-bit scalar quant'],
  ['Sq8', 'Highest-recall quantized', '8-bit scalar quant'],
  ['Suffix array', 'Exact text patterns', 'Hybrid text + vector'],
  ['Scalar', 'Metadata filtering', 'Range / inverted / bitmap'],
]

export function Architecture() {
  return (
    <Section
      id="architecture"
      number="06"
      overline="Architecture"
      title="Lightweight by default, deep when you need it"
      lede="Three mandatory services get you running in seconds. Ten vector index types let you tune recall, latency, and memory per collection."
    >
      <div className="mt-12 grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-surface)] p-7">
          <h3 className="text-[17px] [font-family:var(--font-display)] font-bold text-[var(--color-text)]">
            Three services. That's it.
          </h3>
          <p className="mt-2 text-[14px] leading-relaxed text-[var(--color-text-muted)]">
            No external coordination service, message broker, or metadata store to provision —
            the whole engine runs on three:
          </p>
          <ul className="mt-5 space-y-3.5">
            {SERVICES.map(([name, desc]) => (
              <li key={name} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-[var(--color-accent-dim)] text-[var(--color-accent)]">
                  <Check size={13} strokeWidth={2.5} />
                </span>
                <span className="text-[14px]">
                  <span className="[font-family:var(--font-mono)] font-medium text-[var(--color-text)]">{name}</span>{' '}
                  <span className="text-[var(--color-text-muted)]">— {desc}</span>
                </span>
              </li>
            ))}
          </ul>
          {/* The install command lives once, in the hero. Point here instead of
              repeating a snippet that is incomplete without the clone step. */}
          <p className="mt-6 text-[14px] text-[var(--color-text-muted)]">
            One compose file brings all three up together —{' '}
            <a
              href="/docs/quickstart"
              className="text-[var(--color-accent)] underline decoration-[rgba(255,178,36,0.4)] underline-offset-4 transition-colors hover:decoration-[var(--color-accent)]"
            >
              see the quick start
            </a>
            .
          </p>
        </div>

        <div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)]">
          <table className="w-full text-[13px]">
            <thead className="bg-[var(--color-surface)]">
              <tr className="mono-label text-left text-[var(--color-text-subtle)]">
                <th className="px-5 py-4 font-medium">Index</th>
                <th className="px-5 py-4 font-medium">Best for</th>
                <th className="px-5 py-4 font-medium">Advantage</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)]">
              {INDEXES.map(([name, best, adv]) => (
                <tr key={name} className="transition-colors hover:bg-[var(--color-surface)]">
                  <td className="px-5 py-3 [font-family:var(--font-mono)] font-medium text-[var(--color-text)]">{name}</td>
                  <td className="px-5 py-3 text-[var(--color-text-muted)]">{best}</td>
                  <td className="px-5 py-3 text-[var(--color-text-subtle)]">{adv}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Section>
  )
}
