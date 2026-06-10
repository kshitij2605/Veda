import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'

export function Landing() {
  return (
    <div id="top" className="min-h-screen bg-[var(--color-bg)]">
      <Nav />
      <main className="pt-16">
        <div className="mx-auto max-w-[1200px] px-8 py-32 text-[var(--color-text-muted)]">
          Sections coming next…
        </div>
      </main>
      <Footer />
    </div>
  )
}
