/** Veda mark — three offset shards in the brand accent. */
export function BrandMark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 18 18" aria-hidden>
      <rect x="0" y="0" width="9" height="9" rx="1.5" fill="var(--color-accent)" />
      <rect x="5" y="5" width="9" height="9" rx="1.5" fill="var(--color-accent)" opacity="0.7" />
      <rect x="9" y="9" width="9" height="9" rx="1.5" fill="var(--color-accent)" opacity="0.45" />
    </svg>
  )
}
