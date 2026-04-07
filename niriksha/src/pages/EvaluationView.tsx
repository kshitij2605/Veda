import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { useCollectionStore } from '../stores/collectionStore'
import { PageHeader } from '../components/PageHeader'

const FEATURES = [
  { title: 'RAG Usability Scoring', description: 'Evaluate how well your data supports retrieval-augmented generation pipelines.' },
  { title: 'FAQ Usability Scoring', description: 'Assess data suitability for FAQ-style question answering.' },
  { title: 'Cross-modal Alignment Detection', description: 'Detect misalignment between text, image, and audio embeddings for the same content.' },
  { title: 'Duplicate Detection', description: 'Identify near-duplicate records using vector similarity thresholds.' },
  { title: 'Data Quality Metrics', description: 'Measure blur, signal-to-noise ratio, text length, and other modality-specific quality signals.' },
]

export function EvaluationView() {
  const { collections, fetchCollections } = useCollectionStore()
  const [selectedCollection, setSelectedCollection] = useState('')

  useEffect(() => {
    fetchCollections()
  }, [fetchCollections])

  return (
    <motion.div
      className="p-8"
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
    >
      <PageHeader
        title="Data Quality Evaluation"
        actions={
          <select
            value={selectedCollection}
            onChange={(e) => setSelectedCollection(e.target.value)}
            className="px-3 py-2 bg-bg-secondary border border-border rounded-lg text-sm text-text-primary focus:outline-none focus:border-accent"
          >
            <option value="">Select collection...</option>
            {collections.map((c) => (
              <option key={c.name} value={c.name}>{c.name}</option>
            ))}
          </select>
        }
      />

      <div className="max-w-2xl">
        <div className="rounded-lg border border-border bg-bg-secondary p-6 mb-6">
          <h2 className="text-lg font-semibold text-text-primary mb-2">Evaluation Engine</h2>
          <p className="text-sm text-text-secondary mb-6">
            Analyze your multimodal data quality across collections.
          </p>

          <div className="space-y-4">
            {FEATURES.map((f) => (
              <div key={f.title} className="flex gap-3">
                <div className="mt-1 w-1.5 h-1.5 rounded-full bg-accent shrink-0" />
                <div>
                  <div className="text-sm font-medium text-text-primary">{f.title}</div>
                  <div className="text-xs text-text-muted mt-0.5">{f.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-border border-dashed bg-bg-tertiary/50 p-6 text-center">
          <p className="text-sm text-text-muted">
            Coming Soon -- Requires LLM endpoint configuration
          </p>
        </div>
      </div>
    </motion.div>
  )
}
