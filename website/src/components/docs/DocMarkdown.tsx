import { isValidElement, type ReactElement, type ReactNode } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { slugify } from '../../docs/slugify'
import { CodeBlock } from './CodeBlock'

/** Extract plain text from heading children (handles nested elements). */
function textOf(n: ReactNode): string {
  if (Array.isArray(n)) return n.map(textOf).join('')
  if (typeof n === 'string' || typeof n === 'number') return String(n)
  if (isValidElement(n)) return textOf((n.props as { children?: ReactNode }).children)
  return ''
}

/** The docs prose system — every markdown element mapped to obsidian styling. */
export function DocMarkdown({ md }: { md: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => (
          <h1 className="text-[clamp(30px,4vw,38px)] [font-family:var(--font-display)] font-bold leading-[1.1] tracking-[-0.02em] text-[var(--color-text)]">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2
            id={slugify(textOf(children))}
            className="mt-12 scroll-mt-24 border-t border-[var(--color-border)] pt-8 text-[24px] [font-family:var(--font-display)] font-bold tracking-[-0.01em] text-[var(--color-text)]"
          >
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3
            id={slugify(textOf(children))}
            className="mt-8 scroll-mt-24 text-[17px] font-semibold text-[var(--color-text)]"
          >
            {children}
          </h3>
        ),
        // Markdown wraps a standalone image in a paragraph. A video embed
        // renders <figure>/<figcaption>, which is invalid inside <p> and breaks
        // hydration — so unwrap the paragraph for a lone media embed.
        p: ({ children, node }) => {
          const kids = (node as { children?: unknown[] } | undefined)?.children ?? []
          if (kids.length === 1) {
            const only = kids[0] as { tagName?: string; properties?: { src?: unknown } }
            if (only?.tagName === 'img' && /\.(mp4|webm)$/i.test(String(only.properties?.src ?? ''))) {
              return <>{children}</>
            }
          }
          return (
            <p className="mt-4 text-[15.5px] leading-[1.75] text-[var(--color-text-muted)]">{children}</p>
          )
        },
        a: ({ href, children }) => (
          <a
            href={href}
            className="text-[var(--color-accent)] underline decoration-[rgba(255,178,36,0.4)] underline-offset-4 transition-colors hover:decoration-[var(--color-accent)]"
          >
            {children}
          </a>
        ),
        strong: ({ children }) => <strong className="font-semibold text-[var(--color-text)]">{children}</strong>,
        ul: ({ children }) => (
          <ul className="mt-4 list-disc space-y-2 pl-6 text-[15.5px] leading-[1.7] text-[var(--color-text-muted)] marker:text-[var(--color-accent)]">
            {children}
          </ul>
        ),
        ol: ({ children }) => (
          <ol className="mt-4 list-decimal space-y-2 pl-6 text-[15.5px] leading-[1.7] text-[var(--color-text-muted)] marker:text-[var(--color-text-subtle)]">
            {children}
          </ol>
        ),
        blockquote: ({ children }) => (
          <blockquote className="my-6 rounded-r-[var(--radius-md)] border-l-2 border-[var(--color-accent)] bg-[var(--color-surface)] px-5 py-3 [&_p]:mt-0 [&_p]:text-[14.5px]">
            {children}
          </blockquote>
        ),
        hr: () => <hr className="my-10 border-[var(--color-border)]" />,
        table: ({ children }) => (
          <div className="my-6 overflow-x-auto rounded-[var(--radius-md)] border border-[var(--color-border)]">
            <table className="w-full text-[14px]">{children}</table>
          </div>
        ),
        thead: ({ children }) => (
          <thead className="bg-[var(--color-surface)] text-left">{children}</thead>
        ),
        th: ({ children }) => (
          <th className="mono-label px-4 py-3 font-medium text-[var(--color-text-subtle)]">{children}</th>
        ),
        td: ({ children }) => (
          <td className="border-t border-[var(--color-border)] px-4 py-2.5 align-top leading-relaxed text-[var(--color-text-muted)]">
            {children}
          </td>
        ),
        // Markdown has no video syntax and raw HTML is disabled, so an image
        // whose src is a video file renders as a player: ![caption](/videos/x.mp4)
        img: ({ src, alt }) => {
          const s = String(src ?? '')
          if (!/\.(mp4|webm)$/i.test(s)) {
            return (
              <img
                src={s}
                alt={alt ?? ''}
                className="my-6 w-full rounded-[var(--radius-md)] border border-[var(--color-border)]"
              />
            )
          }
          return (
            <figure className="my-8">
              <video
                src={s}
                poster={s.replace(/\.(mp4|webm)$/i, '.jpg')}
                controls
                muted
                playsInline
                preload="none"
                className="w-full rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-surface)]"
              />
              {alt && (
                <figcaption className="mt-3 text-[13.5px] leading-[1.6] text-[var(--color-text-subtle)]">
                  {alt}
                </figcaption>
              )}
            </figure>
          )
        },
        pre: ({ children }) => {
          const el = isValidElement(children)
            ? (children as ReactElement<{ className?: string; children?: ReactNode }>)
            : null
          const lang = /language-([\w-]+)/.exec(el?.props.className ?? '')?.[1]
          const raw = String(el?.props.children ?? '').replace(/\n$/, '')
          return <CodeBlock code={raw} language={lang} />
        },
        code: ({ children }) => (
          <code className="rounded-[var(--radius-xs)] bg-[var(--color-surface-2)] px-1.5 py-0.5 text-[13px] [font-family:var(--font-mono)] text-[var(--color-text)]">
            {children}
          </code>
        ),
      }}
    >
      {md}
    </ReactMarkdown>
  )
}
