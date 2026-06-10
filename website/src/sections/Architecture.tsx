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
  ['Suffix array', 'Exact text patterns', 'Hybrid text + vector'],
  ['Scalar', 'Metadata filtering', 'Range / inverted / bitmap'],
]

export function Architecture() {
  return (
    <Section
      id="architecture"
      overline="Architecture"
      title="Lightweight by default, deep when you need it"
      lede="Three mandatory services get you running in seconds. Seven index algorithms let you tune recall, latency, and memory per collection."
    >
      <div className="mt-10 grid gap-5 lg:grid-cols-[0.9fr_1.1fr]">
        <div className="rounded-[var(--radius-lg)] border border-[var(--color-border)] bg-[var(--color-bg-elevated)] p-6 shadow-[var(--shadow-sm)]">
          <h3 className="text-[16px] font-semibold text-[var(--color-text)]">Three services. That's it.</h3>
          <p className="mt-2 text-[14px] leading-relaxed text-[var(--color-text-muted)]">
            Milvus needs etcd + Kafka/Pulsar + MinIO before it stores a single vector. Veda needs three:
          </p>
          <ul className="mt-4 space-y-3">
            {SERVICES.map(([name, desc]) => (
              <li key={name} className="flex items-start gap-3">
                <span className="mt-0.5 flex h-5 w-5 flex-none items-center justify-center rounded-full bg-[var(--color-accent-subtle)] text-[var(--color-accent)]">
                  <Check size={13} strokeWidth={2.5} />
                </span>
                <span className="text-[14px]">
                  <span className="font-medium text-[var(--color-text)]">{name}</span>{' '}
                  <span className="text-[var(--color-text-muted)]">— {desc}</span>
                </span>
              </li>
            ))}
          </ul>
          <div className="mt-5 rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-base-1)] px-4 py-3 [font-family:var(--font-mono)] text-[13px] text-[var(--color-text)]">
            <span className="text-[var(--color-text-subtle)]">$</span> docker compose up
          </div>
        </div>

        <div className="overflow-hidden rounded-[var(--radius-lg)] border border-[var(--color-border)] shadow-[var(--shadow-sm)]">
          <table className="w-full text-[13px]">
            <thead className="bg-[var(--color-bg-elevated)] text-[10.5px] uppercase tracking-[0.06em] text-[var(--color-text-subtle)]">
              <tr>
                <th className="px-4 py-3 text-left font-semibold">Index</th>
                <th className="px-4 py-3 text-left font-semibold">Best for</th>
                <th className="px-4 py-3 text-left font-semibold">Advantage</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)] bg-[var(--color-bg-elevated)]">
              {INDEXES.map(([name, best, adv]) => (
                <tr key={name}>
                  <td className="px-4 py-2.5 font-medium [font-family:var(--font-mono)] text-[var(--color-text)]">{name}</td>
                  <td className="px-4 py-2.5 text-[var(--color-text-muted)]">{best}</td>
                  <td className="px-4 py-2.5 text-[var(--color-text-subtle)]">{adv}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </Section>
  )
}
