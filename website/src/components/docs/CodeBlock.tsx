import { useState } from 'react'
import { Check, Copy } from 'lucide-react'

export function CodeBlock({ code, language }: { code: string; language?: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async () => {
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(code)
      } else {
        // http (non-secure-context) fallback
        const ta = document.createElement('textarea')
        ta.value = code
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        ta.remove()
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable — non-fatal */
    }
  }
  return (
    <div className="my-6 overflow-hidden rounded-[var(--radius-md)] border border-[var(--color-border)] bg-[var(--color-code)]">
      <div className="flex items-center justify-between border-b border-[var(--color-border)] px-4 py-1.5">
        <span className="mono-label text-[var(--color-text-subtle)]">{language ?? 'code'}</span>
        <button
          type="button"
          onClick={copy}
          className="inline-flex cursor-pointer items-center gap-1.5 rounded-[var(--radius-sm)] px-2 py-1 text-[12px] [font-family:var(--font-mono)] text-[var(--color-text-subtle)] transition-colors hover:text-[var(--color-text)]"
        >
          {copied ? <Check size={13} className="text-[var(--color-positive)]" /> : <Copy size={13} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="overflow-x-auto px-5 py-4 text-[13px] leading-relaxed [font-family:var(--font-mono)] text-[var(--color-text)]">
        <code>{code}</code>
      </pre>
    </div>
  )
}
