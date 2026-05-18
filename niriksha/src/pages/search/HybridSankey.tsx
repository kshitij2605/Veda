import { useMemo } from 'react'
import type { HybridExplain } from '../../api/collections'

interface HybridSankeyProps {
  data: HybridExplain
}

const COL_DENSE = 'oklch(0.72 0.18 230)'
const COL_BM25 = 'oklch(0.78 0.18 80)'
const COL_FINAL = 'oklch(0.92 0.05 282)'

/**
 * Phase 5.6 — hybrid retrieval Sankey. Renders three rank-ordered
 * lists of doc-ids (dense leg, BM25 leg, final fused result) and
 * draws SVG splines from each leg entry to the final entry that
 * shares the same doc-id. Lines are tinted by leg, with stroke
 * width inversely proportional to source rank so the most-influential
 * candidates pop visually.
 *
 * Not a "real" Sankey — we don't conserve flow magnitude because the
 * fusion math (RRF or weighted-sum) does not surface a single per-id
 * contribution number. What we do show is which leg(s) contributed to
 * each final hit, which is the question operators are actually
 * asking when they open this panel.
 */
export function HybridSankey({ data }: HybridSankeyProps) {
  const rowH = 22
  const colW = 220
  const gapX = 100
  const padY = 14

  const counts = useMemo(
    () => Math.max(data.dense_top.length, data.bm25_top.length, data.final_top.length),
    [data],
  )

  const height = padY * 2 + counts * rowH
  const width = colW * 3 + gapX * 2

  // Build rank maps for each leg.
  const denseRank = useMemo(() => idsToRank(data.dense_top), [data.dense_top])
  const bm25Rank = useMemo(() => idsToRank(data.bm25_top), [data.bm25_top])
  const finalRank = useMemo(() => idsToRank(data.final_top), [data.final_top])

  // For each final hit, identify the source legs.
  const links = useMemo(() => {
    type Link = {
      key: string
      from: 'dense' | 'bm25'
      sourceRank: number
      destRank: number
    }
    const out: Link[] = []
    for (const id of data.final_top) {
      const dRank = denseRank.get(id)
      if (dRank !== undefined) {
        out.push({
          key: `d-${id}`,
          from: 'dense',
          sourceRank: dRank,
          destRank: finalRank.get(id)!,
        })
      }
      const bRank = bm25Rank.get(id)
      if (bRank !== undefined) {
        out.push({
          key: `b-${id}`,
          from: 'bm25',
          sourceRank: bRank,
          destRank: finalRank.get(id)!,
        })
      }
    }
    return out
  }, [data.final_top, denseRank, bm25Rank, finalRank])

  const yFor = (rank: number) => padY + rowH * rank + rowH / 2

  return (
    <div className="overflow-x-auto">
      <svg
        width={width}
        height={height}
        className="block"
        role="img"
        aria-label="Hybrid retrieval Sankey"
      >
        {/* Column headers */}
        <ColHeader x={0} w={colW} label="Dense leg" tint={COL_DENSE} />
        <ColHeader x={colW + gapX} w={colW} label="BM25 leg" tint={COL_BM25} />
        <ColHeader
          x={(colW + gapX) * 2}
          w={colW}
          label={`Final · ${data.fusion}`}
          tint={COL_FINAL}
        />

        {/* Splines */}
        {links.map((l) => {
          const x1 = colW
          const y1 = yFor(l.sourceRank)
          const x2 = (colW + gapX) * 2
          const y2 = yFor(l.destRank)
          // Width inversely scaled to rank — rank 0 is the thickest.
          const sourceTotal =
            l.from === 'dense' ? data.dense_top.length : data.bm25_top.length
          const t = 1 - l.sourceRank / Math.max(1, sourceTotal)
          const sw = 0.8 + t * 1.8
          const baseColor = l.from === 'dense' ? COL_DENSE : COL_BM25
          // When BM25 connects, route through the BM25 column on the
          // way over — single-arc would obscure the BM25 hits.
          const xMid = (colW + gapX + colW + (colW + gapX)) / 2
          const yMid = (y1 + y2) / 2
          const x1c = x1 + gapX / 2
          const x2c = x2 - gapX / 2
          const d =
            l.from === 'dense'
              ? `M ${x1} ${y1} C ${x1c} ${y1}, ${x2c} ${y2}, ${x2} ${y2}`
              : `M ${colW + gapX} ${y1} L ${colW + gapX + colW} ${y1} C ${
                  colW + gapX + colW + gapX * 0.3
                } ${y1}, ${xMid + 40} ${yMid}, ${x2} ${y2}`
          return (
            <path
              key={l.key}
              d={d}
              fill="none"
              stroke={baseColor}
              strokeWidth={sw}
              strokeOpacity={0.55}
            />
          )
        })}

        {/* Rows */}
        {data.dense_top.map((id, i) => (
          <Row
            key={`d-${id}-${i}`}
            x={0}
            y={yFor(i)}
            w={colW}
            rank={i}
            id={id}
            tint={COL_DENSE}
            weight={data.dense_weight}
          />
        ))}
        {data.bm25_top.map((id, i) => (
          <Row
            key={`b-${id}-${i}`}
            x={colW + gapX}
            y={yFor(i)}
            w={colW}
            rank={i}
            id={id}
            tint={COL_BM25}
            weight={data.bm25_weight}
          />
        ))}
        {data.final_top.map((id, i) => (
          <Row
            key={`f-${id}-${i}`}
            x={(colW + gapX) * 2}
            y={yFor(i)}
            w={colW}
            rank={i}
            id={id}
            tint={COL_FINAL}
          />
        ))}
      </svg>
      <div className="mt-2 flex items-center gap-4 text-[10.5px] text-[var(--color-text-subtle)]">
        <LegendDot color={COL_DENSE} label={`Dense w=${data.dense_weight.toFixed(2)}`} />
        <LegendDot color={COL_BM25} label={`BM25 w=${data.bm25_weight.toFixed(2)}`} />
        <span>
          Line thickness ∝ rank in source leg · arc color = leg of origin
        </span>
      </div>
    </div>
  )
}

function ColHeader({
  x,
  w,
  label,
  tint,
}: {
  x: number
  w: number
  label: string
  tint: string
}) {
  return (
    <g>
      <text
        x={x + w / 2}
        y={10}
        textAnchor="middle"
        fontSize={10}
        fontFamily="Inter, system-ui, sans-serif"
        fontWeight={600}
        style={{ letterSpacing: '0.08em', textTransform: 'uppercase' }}
        fill={tint}
      >
        {label}
      </text>
    </g>
  )
}

function Row({
  x,
  y,
  w,
  rank,
  id,
  tint,
  weight,
}: {
  x: number
  y: number
  w: number
  rank: number
  id: string
  tint: string
  weight?: number
}) {
  const shortId = id.length > 18 ? id.slice(0, 18) + '…' : id
  return (
    <g>
      <rect
        x={x}
        y={y - 9}
        width={w}
        height={18}
        rx={3}
        fill={tint}
        fillOpacity={0.08}
        stroke={tint}
        strokeOpacity={0.35}
      />
      <text
        x={x + 8}
        y={y + 4}
        fontSize={10.5}
        fontFamily="JetBrains Mono, monospace"
        fill="oklch(0.85 0.005 252)"
      >
        #{rank} {shortId}
      </text>
      {weight !== undefined && (
        <text
          x={x + w - 8}
          y={y + 4}
          textAnchor="end"
          fontSize={9.5}
          fontFamily="JetBrains Mono, monospace"
          fill="oklch(0.62 0.005 252)"
        >
          {weight.toFixed(2)}
        </text>
      )}
    </g>
  )
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="w-2 h-2 rounded-full"
        style={{ background: color }}
        aria-hidden
      />
      {label}
    </span>
  )
}

function idsToRank(ids: string[]): Map<string, number> {
  const m = new Map<string, number>()
  ids.forEach((id, i) => m.set(id, i))
  return m
}
