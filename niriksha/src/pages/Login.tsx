import { type FormEvent, Suspense, lazy, useState } from 'react'
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

      <div className="relative w-[min(380px,calc(100vw-32px))] rounded-[var(--radius-xl)] border border-[var(--color-border)] bg-[color-mix(in_oklch,var(--color-bg-elevated)_82%,transparent)] backdrop-blur-md p-7 shadow-[var(--shadow-lg)]">
        <div className="flex flex-col items-center text-center">
          <BrandMark />
          <div className="mt-3">
            <GradientText className="text-[26px] font-semibold" colors={['#9a7bff', '#c66be6', '#9a7bff']}>
              Veda
            </GradientText>
          </div>
          <p className="mt-1 text-[12.5px] text-[var(--color-text-muted)]">Sign in to Niriksha</p>
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
  return (
    <svg width={32} height={32} viewBox="0 0 18 18" aria-hidden>
      <rect x="0" y="0" width="9" height="9" rx="1.5" fill="var(--color-accent-7)" />
      <rect x="5" y="5" width="9" height="9" rx="1.5" fill="var(--color-accent-9)" opacity="0.8" />
      <rect x="9" y="9" width="9" height="9" rx="1.5" fill="var(--color-accent-11)" opacity="0.55" />
    </svg>
  )
}
