# Niriksha Phase 1.3 — Login + Animated Background + Auth — Implementation Plan

> Execute task-by-task (TDD). Continues `feat/niriksha-phase1-foundation`. Dev-env up. Edit local → scp-up → `npm test`/`npm run build`.

**Goal:** Ship the login page — the one fully-expressive surface (reactbits **Aurora** WebGL background + animated **GradientText** headline + a real credential form) — wired to JWT auth, finishing Phase 1.

**Architecture:** `/login` is a standalone route (NOT inside `AppShell`); the rest of the app moves under a layout route (`AppShell` + `<Outlet/>`). Aurora (ogl) is **lazy-loaded only on the login route** so WebGL/ogl never enters the main bundle. Auth via `lib/auth.ts` (`POST /v1/auth/login` → store `veda_token`, which the axios client already sends).

**Tech Stack:** React 19, react-router 7 (layout route + Outlet), framer-motion, ogl (new, login-only), reactbits (Aurora, GradientText), Vitest.

**Notes:** Auth isn't enforced on the dev instance (reads return 200 without a token), so a successful login can't be exercised there — but the page, the POST flow, error handling, and token storage are all built and unit-tested. Route-gating (redirect-on-401) is deferred (no enforcement to gate against yet); noted for when auth is turned on.

---

## Task 1: Add ogl + vendor Aurora background

**Files:** `src/components/reactbits/Aurora.tsx` (default export).
- [ ] `npm i ogl` (remote).
- [ ] Vendor the reactbits Aurora source verbatim (it's self-contained WebGL via ogl); brand it by defaulting `colorStops` to indigo→violet hexes near our accent. Keep it a default export (for `React.lazy`).
- [ ] No unit test (WebGL can't run in jsdom; it's mocked wherever consumed). `npx tsc -b` green.
- [ ] Commit `feat(niriksha): vendor reactbits Aurora background (ogl, login-only)`.

## Task 2: Vendor GradientText

**Files:** `src/components/reactbits/GradientText.tsx` (named export).
- [ ] Vendor reactbits GradientText, adapted: import from `framer-motion` (not `motion/react`), `import { type ReactNode }`, export `function GradientText`.
- [ ] Test `src/components/reactbits/__tests__/gradientText.test.tsx`: render `<GradientText>Veda</GradientText>`, assert "Veda" present.
- [ ] Green + commit `feat(niriksha): vendor reactbits GradientText (framer-motion)`.

## Task 3: Auth lib

**Files:** `src/lib/auth.ts` + test `src/lib/__tests__/auth.test.ts`.
- [ ] Test (mock `../api/client` default `post`): `login('u','p')` stores token from `{data:{token}}`; `getToken()` returns it; `logout()` clears it.
- [ ] Implement `login(username,password)` (`api.post('/auth/login',{username,password})` → store `veda_token`; on axios error rethrow `response.data.error ?? message`), `logout()`, `getToken()`, `isAuthed()`.
- [ ] Green + commit `feat(niriksha): auth lib (login/logout/token)`.

## Task 4: Login page

**Files:** `src/pages/Login.tsx` + test `src/pages/__tests__/login.test.tsx`.
- [ ] Test (mock `../components/reactbits/Aurora` to a stub + mock `../lib/auth`): render within MemoryRouter, assert Username/Password inputs + "Sign in" button; type creds, submit, assert `login` called with them.
- [ ] Implement: full-screen, lazy `Aurora` background (opacity + radial vignette to `--color-bg`), centered glass card with the shard brand mark, `GradientText` "Veda" headline, `Input` username + password, error line, primary `Button` "Sign in" (busy state). On submit → `login()` → `navigate('/home')`; on failure show the message.
- [ ] Green + commit `feat(niriksha): login page (Aurora hero + auth form)`.

## Task 5: App routing restructure

**Files:** Modify `src/App.tsx`.
- [ ] Wrap all routes in one top-level `<Suspense>`. Add `<Route path="/login" element={<Login/>} />` (lazy). Move every existing app route under a layout route `<Route element={<ShellLayout/>}>` where `ShellLayout = <AppShell><Outlet/></AppShell>`. Keep `/`→`/home` and `*`→`/home` redirects inside the layout.
- [ ] `npm run build` green (routing compiles). Commit `feat(niriksha): standalone /login route + AppShell layout route`.

## Task 6: Verify
- [ ] Playwright: navigate `/niriksha/login` → screenshot (`login.png`) the Aurora hero + form; confirm 0 console errors; confirm `/niriksha/home` still renders under the shell.
- [ ] Final gate: `npm test` + `npm run build` green.

---
## Self-review
- Spec coverage: login showcase (Aurora + GradientText + form) → T1,T2,T4; auth → T3; standalone route → T5; verify → T6. Global EmptyState already shipped in 1.1.
- No fabricated data: login hits the real `/v1/auth/login`; honest error display.
- Types: `Theme` unaffected; auth returns `{token}`; Aurora default-export for lazy.
