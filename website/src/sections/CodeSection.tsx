import { Section } from '../components/Section'
import { CodeShowcase } from '../components/CodeShowcase'

export function CodeSection() {
  return (
    <Section
      overline="Developer experience"
      title="The whole pipeline, one API call"
      lede="No glue code, no external embedding or OCR services. Create a multimodal collection, fuse across embedding spaces, or turn a PDF into searchable vectors — each is a single request."
    >
      <div className="mt-10">
        <CodeShowcase />
      </div>
    </Section>
  )
}
