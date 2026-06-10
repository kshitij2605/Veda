/** Time-aware greeting for the Home hero. Boundaries: <12 morning, <18 afternoon, else evening. */
export function greetingFor(now: Date): string {
  const h = now.getHours()
  if (h < 12) return 'Good morning'
  if (h < 18) return 'Good afternoon'
  return 'Good evening'
}
