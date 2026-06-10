import { deriveModality, modalityColorVar } from '../modality'

test('image field name → image', () => {
  expect(deriveModality([{ name: 'image_embedding', type: 'vector' }])).toBe('image')
})

test('text default', () => {
  expect(deriveModality([{ name: 'text_embedding', type: 'vector' }])).toBe('text')
})

test('audio voice field → audio', () => {
  expect(deriveModality([{ name: 'audio_embedding', type: 'vector' }])).toBe('audio')
})

test('two distinct modalities → mixed', () => {
  expect(
    deriveModality([
      { name: 'text_embedding', type: 'vector' },
      { name: 'image_embedding', type: 'vector' },
    ]),
  ).toBe('mixed')
})

test('non-vector fields ignored', () => {
  expect(deriveModality([{ name: 'title', type: 'string' }])).toBe('text')
})

test('color var', () => {
  expect(modalityColorVar('audio')).toBe('var(--color-modality-audio)')
})
