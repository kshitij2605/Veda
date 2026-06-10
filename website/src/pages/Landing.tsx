import { Nav } from '../components/Nav'
import { Footer } from '../components/Footer'
import { Hero } from '../sections/Hero'
import { Stats } from '../sections/Stats'
import { Differentiators } from '../sections/Differentiators'
import { CodeSection } from '../sections/CodeSection'
import { HowItWorks } from '../sections/HowItWorks'
import { Architecture } from '../sections/Architecture'
import { CTA } from '../sections/CTA'

export function Landing() {
  return (
    <div id="top" className="min-h-screen bg-[var(--color-bg)]">
      <Nav />
      <main>
        <Hero />
        <Stats />
        <Differentiators />
        <CodeSection />
        <HowItWorks />
        <Architecture />
        <CTA />
      </main>
      <Footer />
    </div>
  )
}
