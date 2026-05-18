import { NavLink, useLocation } from 'react-router-dom'
import {
  Activity,
  Atom,
  BarChart3,
  Beaker,
  Database,
  FlaskConical,
  GitBranch,
  Layers,
  Search,
  Settings,
  TerminalSquare,
} from 'lucide-react'
import { type ComponentType } from 'react'
import { cn } from '../../lib/cn'
import { KBD, modKey } from '../primitives/KBD'

interface NavItem {
  to: string
  label: string
  icon: ComponentType<{ size?: number; strokeWidth?: number }>
  hint?: string
}

interface NavGroup {
  label: string
  items: NavItem[]
}

const GROUPS: NavGroup[] = [
  {
    label: 'Data',
    items: [
      { to: '/collections', label: 'Collections', icon: Database, hint: 'C' },
    ],
  },
  {
    label: 'Explore',
    items: [
      { to: '/search', label: 'Search', icon: Search, hint: 'S' },
      { to: '/atlas', label: 'Atlas', icon: Atom, hint: 'A' },
      { to: '/quality', label: 'Quality Lab', icon: FlaskConical },
      { to: '/evaluation', label: 'Evaluation', icon: Beaker },
    ],
  },
  {
    label: 'Operate',
    items: [
      { to: '/pipelines', label: 'Pipelines', icon: GitBranch },
      { to: '/health', label: 'Health', icon: Activity },
      { to: '/indexes', label: 'Indexes', icon: Layers },
      { to: '/observability', label: 'Observability', icon: BarChart3 },
    ],
  },
  {
    label: 'Tools',
    items: [
      { to: '/console', label: 'Dev Console', icon: TerminalSquare },
      { to: '/settings', label: 'Settings', icon: Settings },
    ],
  },
]

export function Sidebar({ onOpenPalette }: { onOpenPalette: () => void }) {
  return (
    <aside className="flex flex-col w-[220px] flex-none border-r border-[var(--color-border)] bg-[var(--color-bg)]">
      <BrandRow />
      <PaletteButton onClick={onOpenPalette} />
      <nav className="flex-1 overflow-y-auto py-1 no-scrollbar">
        {GROUPS.map((g) => (
          <Group key={g.label} group={g} />
        ))}
      </nav>
      <Footer />
    </aside>
  )
}

function BrandRow() {
  return (
    <div className="h-12 flex items-center px-3 border-b border-[var(--color-border)] gap-2">
      <BrandMark />
      <span className="text-[13px] font-medium tracking-tight text-[var(--color-text)]">
        Niriksha
      </span>
      <span className="ml-auto text-[10px] font-mono text-[var(--color-text-subtle)]">
        v0.1.0
      </span>
    </div>
  )
}

function BrandMark() {
  // Geometric mark — three offset squares evoking shards.
  return (
    <svg width={18} height={18} viewBox="0 0 18 18" aria-hidden>
      <rect x="0" y="0" width="9" height="9" rx="1.5" fill="var(--color-accent-7)" />
      <rect x="5" y="5" width="9" height="9" rx="1.5" fill="var(--color-accent-9)" opacity="0.8" />
      <rect x="9" y="9" width="9" height="9" rx="1.5" fill="var(--color-accent-11)" opacity="0.55" />
    </svg>
  )
}

function PaletteButton({ onClick }: { onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'mx-2 my-2 flex items-center gap-2 h-7 px-2 rounded-[var(--radius-sm)]',
        'bg-[var(--color-base-1)] border border-[var(--color-border)]',
        'text-[12px] text-[var(--color-text-subtle)]',
        'hover:text-[var(--color-text)] hover:border-[var(--color-border-strong)] hover:bg-[var(--color-bg-hover)]',
        'transition-colors duration-[var(--duration-fast)]',
      )}
    >
      <Search size={12} strokeWidth={2.25} />
      <span className="flex-1 text-left">Quick jump</span>
      <KBD>{modKey()}</KBD>
      <KBD>K</KBD>
    </button>
  )
}

function Group({ group }: { group: NavGroup }) {
  return (
    <div className="mb-1.5">
      <div className="px-3 pt-2 pb-1 text-[10px] uppercase tracking-[0.08em] text-[var(--color-text-subtle)] font-medium">
        {group.label}
      </div>
      <ul className="flex flex-col">
        {group.items.map((item) => (
          <Item key={item.to} item={item} />
        ))}
      </ul>
    </div>
  )
}

function Item({ item }: { item: NavItem }) {
  const location = useLocation()
  // Active if path starts with item.to OR if path is `/` and item is /collections
  const active =
    location.pathname === item.to ||
    location.pathname.startsWith(item.to + '/') ||
    (item.to === '/collections' && location.pathname === '/')
  const Icon = item.icon
  return (
    <li>
      <NavLink
        to={item.to}
        className={({ isActive }) =>
          cn(
            'group relative flex items-center gap-2 mx-1.5 h-7 px-2 rounded-[var(--radius-sm)]',
            'text-[12.5px] text-[var(--color-text-muted)]',
            'hover:bg-[var(--color-bg-hover)] hover:text-[var(--color-text)]',
            'transition-colors duration-[var(--duration-fast)]',
            (active || isActive) &&
              'bg-[var(--color-bg-hover)] text-[var(--color-text)]',
          )
        }
      >
        <Icon size={13.5} strokeWidth={2} />
        <span className="flex-1 truncate">{item.label}</span>
        {item.hint && (
          <span className="text-[10px] font-mono text-[var(--color-text-subtle)] opacity-0 group-hover:opacity-100">
            G {item.hint}
          </span>
        )}
        {active && (
          <span
            aria-hidden
            className="absolute left-0 top-1.5 bottom-1.5 w-[2px] rounded-r bg-[var(--color-accent)]"
          />
        )}
      </NavLink>
    </li>
  )
}

function Footer() {
  return (
    <div className="px-3 py-2.5 border-t border-[var(--color-border)] flex items-center justify-between">
      <span className="text-[10px] text-[var(--color-text-subtle)] font-mono">
        veda · 6340
      </span>
      <span className="flex items-center gap-1">
        <span
          className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]"
          aria-hidden
        />
        <span className="text-[10px] text-[var(--color-text-muted)]">
          connected
        </span>
      </span>
    </div>
  )
}

export { GROUPS as NAV_GROUPS }
export type { NavItem }
