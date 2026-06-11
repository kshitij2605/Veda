import { Section } from '../components/Section'
import { CodeShowcase } from '../components/CodeShowcase'
import { ShinyText } from '../components/reactbits/ShinyText'

export function CodeSection() {
  return (
    <Section
      number="03"
      overline="Developer experience"
      title={
        <>
          The whole pipeline, <ShinyText text="one API call" />
        </>
      }
      lede="No glue code, no external embedding or OCR services. Create a multimodal collection, fuse across embedding spaces, or turn a PDF into searchable vectors — each is a single request."
      className="bg-[var(--color-bg-raised)] border-y border-[var(--color-border)]"
    >
      <div className="mt-12">
        <CodeShowcase />
      </div>
    </Section>
  )
}
