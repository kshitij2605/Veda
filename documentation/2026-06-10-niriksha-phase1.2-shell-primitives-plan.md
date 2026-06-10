# Niriksha Phase 1.2 — Full Primitives + Shell Upgrades — Implementation Plan

> **For agentic workers:** Execute task-by-task (TDD). Steps use `- [ ]`. Continues `feat/niriksha-phase1-foundation`. Dev-env already up (veda-server :6340 seeded, niriksha dev :5174). Edit local `niriksha/src` → scp-up → `npm test` / `npm run build` over ssh.

**Goal:** Round out the primitive library and upgrade the app shell (collapsible sidebar, top-bar cluster + theme toggle, command-palette actions/nav/recent, standardized PageHeader) so the system is complete enough for Phase-2 pages.

**Architecture:** New token-driven primitives + a TanStack DataTable shell; theme toggling via `data-theme` on `<html>` (tokens already define the light override); sidebar collapse persisted in localStorage; cmdk palette fed from `NAV_GROUPS` + the collection store.

**Tech Stack:** React 19, Radix (Tabs/Switch deps present where noted), TanStack table + virtual (installed), cmdk (installed), framer-motion. Vitest + RTL.

**Scope note (YAGNI):** Stepper, ProgressBar, Toolbar, CodeBlock are deferred to Phase 2 (built with the document-pipeline wizard / Console that consume them). This plan ships what the shell + Home/Collections/Search need now.

---

## Task 1: Tabs primitive (wrap Radix)

**Files:** Create `src/components/primitives/Tabs.tsx`, test `src/components/primitives/__tests__/tabs.test.tsx`; export from `index.ts`.

- [ ] Test: render Tabs with two items, assert both triggers render and the active panel content shows.
- [ ] Implement a thin wrapper over `@radix-ui/react-tabs` (installed): `Tabs({items: {value,label,content}[], defaultValue?})`, token-styled triggers (underline/active accent), `text-[12px]`, focus rings inherited.
- [ ] Export `Tabs` + `TabItem` type from `index.ts`. Run `npm test` + `npx tsc -b`.
- [ ] Commit: `feat(niriksha): Tabs primitive (radix)`.

## Task 2: SegmentedControl

**Files:** `src/components/primitives/SegmentedControl.tsx` + test; export.

- [ ] Test: render with 3 options, click second, assert `onChange` called with its value; active option has aria-pressed.
- [ ] Implement: `SegmentedControl({options:{value,label}[], value, onChange})` — a token-styled button group (rounded container `bg-[var(--color-bg-overlay)]`, active pill `bg-[var(--color-bg-active)]`), `role=group`, buttons `aria-pressed`.
- [ ] Export + tests green + commit `feat(niriksha): SegmentedControl primitive`.

## Task 3: Switch

**Files:** `src/components/primitives/Switch.tsx` + test; export.

- [ ] Test: render unchecked, click, assert `onChange(true)`; has `role="switch"` + `aria-checked`.
- [ ] Implement a custom accessible switch (no radix-switch dep): `Switch({checked, onChange, label?})` — track `bg-[var(--color-bg-active)]`→`bg-[var(--color-accent)]` when on, knob translate, keyboard (Space/Enter), `role="switch"`.
- [ ] Export + green + commit `feat(niriksha): Switch primitive`.

## Task 4: PageHeader primitive

**Files:** `src/components/primitives/PageHeader.tsx` + test; export.

- [ ] Test: render with title + description + actions, assert all present.
- [ ] Implement `PageHeader({title, description?, actions?, tabs?})` — `h1 text-[20px] font-semibold`, description `text-[13px] muted`, actions right-aligned, optional Tabs row beneath. This is the standard page header for every page (supersedes the ad-hoc `src/components/PageHeader.tsx`; old one stays until pages migrate).
- [ ] Export + green + commit `feat(niriksha): PageHeader primitive`.

## Task 5: DataTable shell

**Files:** `src/components/table/DataTable.tsx` + test `src/components/table/__tests__/dataTable.test.tsx`.

- [ ] Test: render with 2 columns + 3 rows (no virtualization needed for the test), assert a cell value + header render; row `onRowClick` fires.
- [ ] Implement a generic `DataTable<T>({columns, data, onRowClick?})` over `@tanstack/react-table` (installed) with token styling (header `text-[var(--color-text-subtle)]`, rows hover `bg-[var(--color-bg-hover)]`, borders `--color-border`). Virtualize with `@tanstack/react-virtual` only when `data.length > 50` (keep the test path non-virtual).
- [ ] Export from a `src/components/table/index.ts`. Green + commit `feat(niriksha): DataTable shell (tanstack)`.

## Task 6: Theme toggle

**Files:** `src/lib/theme.ts` (+ test), `src/components/primitives/ThemeToggle.tsx`; export ThemeToggle.

- [ ] Test (`src/lib/__tests__/theme.test.ts`): `applyTheme('light')` sets `document.documentElement.dataset.theme='light'` and persists; `applyTheme('dark')` removes it. `getStoredTheme()` returns persisted.
- [ ] Implement `lib/theme.ts`: `type Theme='dark'|'light'`; `getStoredTheme()` (localStorage `veda_theme`, default 'dark'); `applyTheme(t)` (set/remove `data-theme`, persist). `ThemeToggle.tsx`: a Sun/Moon icon button (lucide) toggling theme, initialized from storage.
- [ ] Export + green + commit `feat(niriksha): theme toggle (dark/light)`.

## Task 7: Collapsible sidebar

**Files:** Modify `src/components/shell/Sidebar.tsx`; test update.

- [ ] Test: render Sidebar with a `collapsed` prop true; assert nav labels are not visible (only icons) and the brand still renders.
- [ ] Implement: `Sidebar({onOpenPalette, collapsed, onToggle})` — width `w-[220px]`↔`w-[56px]`; when collapsed hide labels/hints/group headers, center icons, wrap items in `Tooltip` (existing primitive). Add a collapse chevron button in the BrandRow. Persist collapsed in localStorage via the AppShell (Task 8).
- [ ] Green + commit `feat(niriksha): collapsible sidebar with icon rail`.

## Task 8: Top-bar cluster + shell wiring

**Files:** Modify `src/components/shell/AppShell.tsx`.

- [ ] Manage `collapsed` state in AppShell (localStorage `veda_sidebar_collapsed`), pass to Sidebar. Initialize theme on mount (`applyTheme(getStoredTheme())`).
- [ ] Header right cluster: `ConnectionPill` driven by `useCollectionStore().health` (fetch health on mount), `ThemeToggle`, and keep `TrustBadges`. Left keeps `Breadcrumbs`.
- [ ] `npm run build` green (shell compiles). Commit `feat(niriksha): top-bar cluster (health + theme) + sidebar collapse wiring`.

## Task 9: Command palette upgrade

**Files:** Modify `src/components/palette/CommandPalette.tsx`; test `src/components/palette/__tests__/palette.test.tsx`.

- [ ] Read the current CommandPalette first to match its cmdk usage.
- [ ] Test: render palette `open`, assert a nav item (e.g. "Collections") and an action (e.g. "Toggle theme") appear.
- [ ] Implement: groups — **Navigation** (from `NAV_GROUPS` in Sidebar), **Actions** (Toggle theme, Collapse sidebar, New collection), **Recent collections** (from `useCollectionStore().collections`, first ~6). Selecting navigates / runs the action and closes.
- [ ] Green + commit `feat(niriksha): command palette — nav + actions + recent collections`.

## Task 10: Verify on the live app

- [ ] Playwright: toggle sidebar collapse (screenshot icon rail), open `⌘K` palette (screenshot), toggle theme (screenshot light mode), confirm 0 console errors.
- [ ] Final gate: `npm test` + `npm run build` green. Screenshot `shell-upgrades.png`.

---

## Self-review
- Spec coverage: shell IA upgrades (collapsible sidebar, top-bar cluster, palette upgrade, page-header) → T4,T7,T8,T9; primitive library (Tabs/SegmentedControl/Switch/DataTable) → T1,T2,T3,T5; theme → T6. Deferred primitives (Stepper/ProgressBar/Toolbar/CodeBlock) explicitly noted for Phase 2.
- Types consistent: `Theme`, `TabItem`, `SegmentedOption`, `Column<T>` defined once and reused.
- No fabricated data: health pill uses real `/admin/health`; recent collections from real store.
