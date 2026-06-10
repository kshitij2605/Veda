import { sumPromMetric } from '../metrics'

const TXT = `# TYPE veda_records_total gauge
veda_records_total{collection="a"} 42
veda_records_total{collection="b"} 8
# TYPE veda_other gauge
veda_records_total_extra{collection="c"} 999`

test('sums only the exact metric family', () => {
  expect(sumPromMetric(TXT, 'veda_records_total')).toBe(50)
})
