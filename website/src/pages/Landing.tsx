import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'
import { Hero } from '../sections/Hero'
import { Stats } from '../sections/Stats'
import { Differentiators } from '../sections/Differentiators'
import { CodeSection } from '../sections/CodeSection'

export function Landing() {
  return (
    <div id="top" className="min-h-screen bg-[var(--color-bg)]">
      <Nav />
      <main>
        <Hero />
        <Stats />
        <Differentiators />
        <CodeSection />
        <div className="mx-auto max-w-[1200px] px-8 py-24 text-[var(--color-text-muted)]">
          More sections coming…
        </div>
      </main>
      <Footer />
    </div>
  )
}
