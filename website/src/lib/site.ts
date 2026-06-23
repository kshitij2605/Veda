// Single source of truth for the site's canonical origin and default
// metadata. The domain is a PLACEHOLDER until production is decided —
// swap `veda.example` here (and in public/robots.txt, public/sitemap.xml,
// public/llms.txt) for the real domain at launch.
export const SITE_URL = 'https://veda.example'

export const SITE_NAME = 'Veda'
export const DEFAULT_TITLE = 'Veda — Multimodal Vector Database & RAG Platform (Rust)'
export const DEFAULT_DESCRIPTION =
  'Veda is an open-source, Rust-native multimodal vector database and RAG platform: one unified vector space for text, images, audio, video and documents, with hybrid search, reranking, and a built-in MCP server.'

/** Absolute URL for a route path (leading-slash form, e.g. "/docs/quickstart"). */
export function canonical(path: string): string {
  return SITE_URL + (path.startsWith('/') ? path : '/' + path)
}
