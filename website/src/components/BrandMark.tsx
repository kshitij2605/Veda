/** Veda mark — constellation V: a V traced through a nearest-neighbor graph. */
export function BrandMark({ size = 20 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" aria-hidden>
      <g stroke="#FFB224" strokeOpacity=".5" strokeWidth="2.8" fill="none" strokeLinecap="round">
        <path d="M13 12 L22.5 30 L32 48 L41.5 30 L51 12" />
      </g>
      <circle cx="13" cy="12" r="4.6" fill="#FFB224" />
      <circle cx="22.5" cy="30" r="4" fill="#FFB224" />
      <circle cx="32" cy="48" r="6" fill="#FFB224" />
      <circle cx="41.5" cy="30" r="4" fill="#FFB224" />
      <circle cx="51" cy="12" r="4.6" fill="#FFB224" />
    </svg>
  )
}
