import { type FormEvent, Suspense, lazy, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Input } from '../components/primitives'
import { GradientText } from '../components/reactbits/GradientText'
import { login } from '../lib/auth'

const Aurora = lazy(() => import('../components/reactbits/Aurora'))

export function Login() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  // The login is a full-screen standalone moment — force the dramatic dark
  // theme here regardless of the app's (now light-first) default; restore on leave.
  useEffect(() => {
    const prev = document.documentElement.dataset.theme
    document.documentElement.dataset.theme = 'dark'
    return () => {
      document.documentElement.dataset.theme = prev || 'light'
    }
  }, [])

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(username, password)
      navigate('/home')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign-in failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-[var(--color-bg)] flex items-center justify-center">
      <div aria-hidden className="absolute inset-0 opacity-70">
        <Suspense fallback={null}>
          <Aurora colorStops={['#6d4dff', '#b06bff', '#6d4dff']} amplitude={1.0} blend={0.6} />
        </Suspense>
      </div>
      <div
        aria-hidden
        className="absolute inset-0"
        style={{ background: 'radial-gradient(58% 50% at 50% 42%, transparent, var(--color-bg) 82%)' }}
      />

      <div className="relative w-[min(400px,calc(100vw-32px))] rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[color-mix(in_oklch,var(--color-bg-elevated)_82%,transparent)] backdrop-blur-md p-8 shadow-[var(--shadow-lg)]">
        <div className="flex flex-col items-center text-center">
          <BrandMark />
          <div className="mt-4">
            <GradientText
              className="text-[34px] font-semibold [font-family:var(--font-display)] tracking-[-0.01em]"
              colors={['#9a7bff', '#c66be6', '#9a7bff']}
            >
              Veda
            </GradientText>
          </div>
          <p className="mt-1.5 text-[14px] text-[var(--color-text-muted)]">Sign in to Niriksha</p>
        </div>

        <form onSubmit={onSubmit} className="mt-6 space-y-3">
          <Input
            placeholder="Username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
          />
          <Input
            placeholder="Password"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {error && <div className="text-[12px] text-[var(--color-error)]">{error}</div>}
          <Button type="submit" variant="primary" size="lg" block disabled={busy}>
            {busy ? 'Signing in…' : 'Sign in'}
          </Button>
        </form>

        <p className="mt-4 text-center text-[11px] text-[var(--color-text-subtle)]">
          Veda · multimodal vector database
        </p>
      </div>
    </div>
  )
}

function BrandMark() {
  // Veda mark — constellation V (the login is the doorway to Veda itself).
  return (
    <svg width={34} height={34} viewBox="0 0 64 64" aria-hidden>
      <g stroke="#FFB224" strokeOpacity=".5" strokeWidth="2.8" fill="none" strokeLinecap="round">
        <path d="M13 12 L22.5 30 L32 48 L41.5 30 L51 12" />
        <path d="M41.5 30 L56 30" strokeOpacity=".25" strokeWidth="2" />
      </g>
      <circle cx="13" cy="12" r="4.6" fill="#FFB224" />
      <circle cx="22.5" cy="30" r="4" fill="#FFB224" />
      <circle cx="32" cy="48" r="6" fill="#FFB224" />
      <circle cx="41.5" cy="30" r="4" fill="#FFB224" />
      <circle cx="51" cy="12" r="4.6" fill="#FFB224" />
      <circle cx="56" cy="30" r="2.6" fill="#F2F4F8" fillOpacity=".7" />
    </svg>
  )
}
