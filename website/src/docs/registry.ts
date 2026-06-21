import introduction from './content/introduction.md?raw'
import quickstart from './content/quickstart.md?raw'
import installation from './content/installation.md?raw'
import collections from './content/collections.md?raw'
import indexes from './content/indexes.md?raw'
import search from './content/search.md?raw'
import multimodal from './content/multimodal.md?raw'
import documentPipeline from './content/document-pipeline.md?raw'
import agentsMcp from './content/agents-mcp.md?raw'
import webConnectors from './content/web-connectors.md?raw'
import restApi from './content/rest-api.md?raw'
import pythonSdk from './content/python-sdk.md?raw'
import observability from './content/observability.md?raw'
import apmObservability from './content/apm-observability.md?raw'
import llmObservability from './content/llm-observability.md?raw'
import aiops from './content/aiops.md?raw'
import opsAdvisor from './content/ops-advisor.md?raw'
import logInsights from './content/log-insights.md?raw'
import resourceSignals from './content/resource-signals.md?raw'
import serviceReliability from './content/service-reliability.md?raw'
import ragAiops from './content/rag-aiops.md?raw'
import contributing from './content/contributing.md?raw'

export interface DocPageEntry {
  slug: string
  title: string
  md: string
}

export interface DocGroup {
  label: string
  items: DocPageEntry[]
}

export const DOCS: DocGroup[] = [
  {
    label: 'Getting started',
    items: [
      { slug: 'introduction', title: 'Introduction', md: introduction },
      { slug: 'quickstart', title: 'Quick start', md: quickstart },
      { slug: 'installation', title: 'Installation & config', md: installation },
    ],
  },
  {
    label: 'Core concepts',
    items: [
      { slug: 'collections', title: 'Collections & fields', md: collections },
      { slug: 'indexes', title: 'Indexes', md: indexes },
      { slug: 'search', title: 'Search & fusion', md: search },
      { slug: 'multimodal', title: 'Multimodal & cross-modal', md: multimodal },
      { slug: 'document-pipeline', title: 'Document pipeline', md: documentPipeline },
      { slug: 'agents-mcp', title: 'Agents & MCP', md: agentsMcp },
      { slug: 'web-connectors', title: 'Web Connectors', md: webConnectors },
    ],
  },
  {
    label: 'Reference',
    items: [
      { slug: 'rest-api', title: 'REST API', md: restApi },
      { slug: 'python-sdk', title: 'Python SDK', md: pythonSdk },
      { slug: 'observability', title: 'Observability', md: observability },
      { slug: 'apm-observability', title: 'APM & Performance', md: apmObservability },
      { slug: 'llm-observability', title: 'LLM Observability', md: llmObservability },
      { slug: 'aiops', title: 'AIOps', md: aiops },
      { slug: 'ops-advisor', title: 'Ops Advisor', md: opsAdvisor },
      { slug: 'log-insights', title: 'Log Insights', md: logInsights },
      { slug: 'resource-signals', title: 'Node Resource Signals', md: resourceSignals },
      { slug: 'service-reliability', title: 'Service Reliability', md: serviceReliability },
      { slug: 'rag-aiops', title: 'RAG & AIOps Intelligence', md: ragAiops },
    ],
  },
  {
    label: 'Community',
    items: [
      { slug: 'contributing', title: 'Contributing', md: contributing },
    ],
  },
]

export const FLAT: DocPageEntry[] = DOCS.flatMap((g) => g.items)

export const bySlug = (slug: string | undefined): DocPageEntry | undefined =>
  FLAT.find((p) => p.slug === slug)
