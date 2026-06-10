import { deriveModalityMix } from '../modalityMix'

test('counts by modality in stable order', () => {
  const mix = deriveModalityMix([
    { name: 'a', fields: [{ name: 'text_embedding', type: 'vector' }] },
    { name: 'b', fields: [{ name: 'image_embedding', type: 'vector' }] },
    {
      name: 'c',
      fields: [
        { name: 'text_embedding', type: 'vector' },
        { name: 'image_embedding', type: 'vector' },
      ],
    },
  ])
  expect(mix).toEqual([
    { modality: 'text', count: 1 },
    { modality: 'image', count: 1 },
    { modality: 'mixed', count: 1 },
  ])
})
