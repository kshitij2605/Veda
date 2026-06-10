import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

export interface DonutDatum {
  label: string
  value: number
  /** Any CSS color (we pass token vars like `var(--color-modality-text)`). */
  color: string
}

/** Token-themed donut. Colors come from the caller so the chart stays on-brand. */
export function DonutChart({ data, size = 168 }: { data: DonutDatum[]; size?: number }) {
  return (
    <div style={{ width: '100%', height: size }}>
      <ResponsiveContainer>
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="label"
            innerRadius="62%"
            outerRadius="100%"
            paddingAngle={2}
            stroke="var(--color-bg-elevated)"
            strokeWidth={2}
          >
            {data.map((d) => (
              <Cell key={d.label} fill={d.color} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: 'var(--color-bg-overlay)',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-sm)',
              fontSize: 12,
              color: 'var(--color-text)',
            }}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  )
}
