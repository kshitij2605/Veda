import { useEffect, useRef } from 'react'
import { motion, useAnimationFrame, useMotionValue, useTransform } from 'framer-motion'

/**
 * Vendored from ReactBits (TextAnimations/ShinyText), adapted:
 * - imports from `framer-motion` (this repo's package) instead of motion/react
 * - saffron-on-ink defaults, named export, trimmed options
 */
export function ShinyText({
  text,
  speed = 3,
  className = '',
  color = '#F2F4F8',
  shineColor = '#FFB224',
  spread = 120,
}: {
  text: string
  speed?: number
  className?: string
  color?: string
  shineColor?: string
  spread?: number
}) {
  const progress = useMotionValue(0)
  const elapsedRef = useRef(0)
  const lastTimeRef = useRef<number | null>(null)
  const duration = speed * 1000

  useAnimationFrame((time) => {
    if (lastTimeRef.current === null) {
      lastTimeRef.current = time
      return
    }
    elapsedRef.current += time - lastTimeRef.current
    lastTimeRef.current = time
    progress.set(((elapsedRef.current % duration) / duration) * 100)
  })

  useEffect(() => {
    elapsedRef.current = 0
    progress.set(0)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [speed])

  const backgroundPosition = useTransform(progress, (p) => `${150 - p * 2}% center`)

  return (
    <motion.span
      className={`inline-block ${className}`}
      style={{
        backgroundImage: `linear-gradient(${spread}deg, ${color} 0%, ${color} 38%, ${shineColor} 50%, ${color} 62%, ${color} 100%)`,
        backgroundSize: '200% auto',
        WebkitBackgroundClip: 'text',
        backgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
        backgroundPosition,
      }}
    >
      {text}
    </motion.span>
  )
}
