import * as RadixTabs from '@radix-ui/react-tabs'
import { type ReactNode } from 'react'

export interface TabItem {
  value: string
  label: string
  content: ReactNode
}

/** Token-styled tabs over Radix. Active trigger gets an accent underline. */
export function Tabs({ items, defaultValue }: { items: TabItem[]; defaultValue?: string }) {
  return (
    <RadixTabs.Root defaultValue={defaultValue ?? items[0]?.value} className="w-full">
      <RadixTabs.List className="flex items-center gap-1 border-b border-[var(--color-border)]">
        {items.map((it) => (
          <RadixTabs.Trigger
            key={it.value}
            value={it.value}
            className="group relative px-3 h-8 text-[12px] text-[var(--color-text-muted)] hover:text-[var(--color-text)] data-[state=active]:text-[var(--color-text)] transition-colors"
          >
            {it.label}
            <span className="absolute left-0 right-0 -bottom-px h-[2px] bg-[var(--color-accent)] opacity-0 group-data-[state=active]:opacity-100" />
          </RadixTabs.Trigger>
        ))}
      </RadixTabs.List>
      {items.map((it) => (
        <RadixTabs.Content key={it.value} value={it.value} className="pt-4 focus-visible:outline-none">
          {it.content}
        </RadixTabs.Content>
      ))}
    </RadixTabs.Root>
  )
}
