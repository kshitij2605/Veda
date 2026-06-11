import { useEffect, useRef, useState } from 'react'

/**
 * Vendored from ReactBits (TextAnimations/DecryptedText), simplified:
 * - plain <span> (no motion/react dependency)
 * - reveals sequentially once when scrolled into view
 * - prefers-reduced-motion renders the plain text immediately
 */
export function DecryptedText({
  text,
  speed = 45,
  characters = '01<>{}[]#$%&*+=/\\',
  className = '',
  encryptedClassName = '',
}: {
  text: string
  speed?: number
  characters?: string
  className?: string
  encryptedClassName?: string
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const [revealed, setRevealed] = useState(0)
  const [started, setStarted] = useState(false)
  const [scramble, setScramble] = useState(text)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      setRevealed(text.length)
      return
    }
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setStarted(true)
          obs.disconnect()
        }
      },
      { threshold: 0.1 },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [text])

  useEffect(() => {
    if (!started || revealed >= text.length) return
    const id = setInterval(() => {
      setRevealed((r) => Math.min(r + 1, text.length))
      setScramble(
        text
          .split('')
          .map((ch) => (ch === ' ' ? ' ' : characters[Math.floor(Math.random() * characters.length)]))
          .join(''),
      )
    }, speed)
    return () => clearInterval(id)
  }, [started, revealed, text, speed, characters])

  return (
    <span ref={ref} className="inline-block whitespace-pre">
      <span className="sr-only">{text}</span>
      <span aria-hidden>
        {text.split('').map((ch, i) => (
          <span key={i} className={i < revealed ? className : encryptedClassName}>
            {i < revealed ? ch : ch === ' ' ? ' ' : scramble[i]}
          </span>
        ))}
      </span>
    </span>
  )
}
