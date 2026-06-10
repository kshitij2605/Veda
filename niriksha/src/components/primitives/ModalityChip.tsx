import { type Modality, MODALITY_LABEL, modalityColorVar } from '../../lib/modality'

/** Small colored badge identifying a modality (text/image/audio/…). */
export function ModalityChip({ modality }: { modality: Modality }) {
  const c = modalityColorVar(modality)
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-[var(--radius-sm)] px-1.5 py-0.5 text-[10px] font-medium border"
      style={{
        color: c,
        borderColor: `color-mix(in oklch, ${c} 40%, transparent)`,
        background: `color-mix(in oklch, ${c} 12%, transparent)`,
      }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
      {MODALITY_LABEL[modality]}
    </span>
  )
}
