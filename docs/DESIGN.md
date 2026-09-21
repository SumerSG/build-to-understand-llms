# Design system

The lab's UI should feel calm, oriented and in control: one accent, hairlines instead of boxes, generous
space, and motion that explains rather than decorates. Everything below lives in `app/styles.css` as CSS
custom properties on `:root`; dark mode redefines the colour tokens under both
`@media (prefers-color-scheme: dark) :root:not([data-theme="light"])` and `:root[data-theme="dark"]`.
Use the tokens, never raw values, so a screen stays consistent in both themes.

## Tokens

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#f5f5f7` | `#000` | page background |
| `--surface` | `#fff` | `#1d1d1f` | cards, panels, editor |
| `--surface-2` | `#f5f5f7` | `#131314` | inset areas inside a card: editor bar, console, code blocks |
| `--fill` / `--fill-hover` | `#e8e8ed` / `#dcdce1` | `#2c2c2e` / `#3a3a3c` | quiet gray: secondary buttons, chips, assistant bubble |
| `--ink` | `#1d1d1f` | `#f5f5f7` | primary text |
| `--ink-2` | `#515154` | `#c7c7cc` | secondary prose (blurbs, instructions) |
| `--muted` | `#6e6e73` | `#a1a1a6` | captions, meta, breadcrumbs, labels |
| `--border` | `rgba(0,0,0,.08)` | `rgba(255,255,255,.12)` | hairline separators and card borders |
| `--border-strong` | `rgba(0,0,0,.18)` | `rgba(255,255,255,.28)` | input borders, empty dots |
| `--accent` / `--accent-hover` | `#0071e3` / `#0077ed` | `#2997ff` / `#3ea2ff` | the one blue: primary buttons, links, progress, user bubble |
| `--good` / `--good-ink` | `#34c759` / `#1d8a3c` | `#30d158` | dots and fills / text (text needs the darker one on white) |
| `--warn` / `--warn-ink` | `#ff9f0a` / `#c93400` | `#ff9f0a` / `#ffb340` | stale marks, predict label, storage warning |
| `--bad` / `--bad-ink` | `#ff3b30` / `#d70015` | `#ff453a` / `#ff6961` | failed dots / failure text |
| `--*-tint` | `rgba(colour, .10–.14)` | same | soft backgrounds behind a semantic state |
| `--focus` | `0 0 0 4px rgba(0,113,227,.25)` | `rgba(41,151,255,.35)` | focus ring on every input and button |
| `--shadow-float` | `0 8px 24px rgba(0,0,0,.08)` | `rgba(0,0,0,.55)` | only for floating things: tooltips, the mobile sidebar |
| `--glass` | `rgba(255,255,255,.72)` | `rgba(29,29,31,.72)` | the frosted sidebar (`backdrop-filter: saturate(180%) blur(20px)`); `.94`/`.96` on phones |
| `--seg-bg` / `--seg-ind` | `rgba(0,0,0,.06)` / `#fff` | `rgba(255,255,255,.08)` / `#3a3a3c` | segmented control track / sliding indicator |
| `--grid` / `--axis` | `rgba(0,0,0,.08)` / `.25` | `rgba(255,255,255,.10)` / `.28` | chart chrome |
| `--chart-1…8` | validated palette (unchanged) | dark variants | chart series only |

## Type scale

System stack: `-apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Inter,
Helvetica, Arial, sans-serif`. Mono: `"SF Mono", ui-monospace, Menlo, Consolas`. Two weights on a screen
(400 and 600); 700 is reserved for the page title. `b`/`strong` render at 600. Numbers that line up
(durations, test times, ticks, tables, module numbers) use `font-variant-numeric: tabular-nums`.

| Role | Size / weight | Notes |
|---|---|---|
| Display (`.hero h1`) | 44px / 700, −0.025em | home hero only; 34px on phones |
| Page title (`h1`) | 34px / 700, −0.02em | one per screen; 30px on phones |
| Section (`h2`) | 24px / 600, −0.012em | "Run the goal", "Explain it in your own words" |
| Title 3 (`.prose h3`) | 20px / 600 | sub-headings inside reading text |
| Headline (`h3`) | 17px / 600 | card titles, step chips' panel heading is 22px |
| Body | 16px / 400, lh 1.5 | UI text |
| Prose (`.prose`, `.concept`) | 17px / 400, lh 1.55 | reading text, max-width `--measure` (720px) |
| Secondary (`.small`) | 14px | sidebar, tabs, chips, meta |
| Caption (`.caption`) | 12px | timestamps, "Step 1 of 5", tick labels are 11px |
| Eyebrow (`.eyebrow`, `.goal-banner .label`) | 12px / 600, +0.08em, uppercase | small-caps tracking labels |
| Code | 13px mono | inline code is 0.875em of its context |

## Spacing and shape

8-point grid: `--s1` 4 · `--s2` 8 · `--s3` 12 · `--s4` 16 · `--s6` 24 · `--s8` 32 · `--s12` 48 · `--s16` 64.
Page gutter `--gutter` is 48px (16px on phones); sidebar 280px; reading measure 720px.

| Shape | Radius | Border | Shadow |
|---|---|---|---|
| Card (`.card`, goal banner, chart, editor) | 16px (14px on phones) | hairline | none at rest |
| Inset (`.callout`, console, code, hints, quiz options) | 12px | hairline or none | none |
| Input / textarea | 10px | `--border-strong`, accent when focused | focus ring |
| Button, chip, badge | 980px (pill) | none | none |
| Segmented control | 10px track, 8px indicator | none | `--seg-shadow` on the indicator |
| Chat bubble | 18px, 6px on the tail corner | none | none |
| Tooltip, mobile sidebar | 10px / 16px | hairline | `--shadow-float` |

## Motion

One curve, `cubic-bezier(0.25, 0.1, 0.25, 1)`, at 220ms (`--dur`) for hover, colour and chevrons and 260ms
(`--dur-slow`) for things that move: the segmented-control indicator (translateX + width), hint rows
(`grid-template-rows: 0fr → 1fr`), the sidebar sheet, progress bars. `prefers-reduced-motion: reduce`
disables every transition and animation.

## Components

- **Buttons** — `.btn` (quiet gray pill), `.btn-primary` (filled accent, white text), `.btn-ghost` (text
  link with a trailing ›), `.btn-small` (32px). One primary per group. 40px tall by default, 44px on phones.
- **Segmented control** — `.phases` with a `.phases-ind` that `main.js` positions under the active `.phase`;
  it slides when the phase changes within a module and scrolls horizontally on narrow screens.
- **Step chips** — `.step-chip` pills: gray, green tint when passed, ink-filled when active.
- **Cards** — `.card`; a card's first `h2`/`h3` has no top margin. Feature card: `.goal-banner` with an
  accent eyebrow label, 19px goal sentence and the threshold concept under a hairline.
- **Lists with status** — `.test` rows (8px dot, name, mono failure message, tabular ms), `.checklist`,
  the track card module list (number, title, minutes, green check when complete), all separated by hairlines.
- **Hints** — `.hint` rows with a chevron; `.hint.open` expands `.hint-body` in place and unlocks the next rung.
- **Quiz options** — `.quiz-opt` full-width rows; `.correct` green tint, `.wrong` red tint.
- **Predict card** — `.predict` with an orange eyebrow, textarea and a Reveal button; revealed answers sit
  under a hairline.
- **Charts** — `.chart` card: hairline grid, 11px muted ticks, 2px lines, markers stroked in the surface
  colour, a floating `.chart-tip` card. Data logic and the series palette are untouched.
- **Chat** — `.chat-user` right/accent, `.chat-assistant` left/gray, pill composer with a primary Send.
- **Sidebar** — frosted, sticky; on phones a fixed sheet (`.sidebar.open`) with a scrim (`.shell:has(.sidebar.open)::before`)
  that closes on tap outside.

## Do / don't

- Do use `--ink-2`/`--muted` for hierarchy; don't lower opacity to make text "secondary".
- Do put a state in the dot or tint and keep the text in `--*-ink`; don't set body text in `--good`/`--bad`
  (the raw greens and reds fail contrast on white).
- Do use hairlines and spacing to separate; don't add borders on borders, left accent bars, or shadows at rest.
- Do use one accent per screen for the primary action; don't colour secondary buttons.
- Do keep prose inside `.measure`; don't stretch reading text across the full main column.
- Do keep the ids and classes `main.js` and `tools/e2e.mjs` rely on (`#btn-check`, `#btn-check-all`,
  `#btn-solution`, `#sol-copy`, `#btn-run`, `#btn-complete`, `.step-chip`, `.results .status-line`,
  `.goal-out .done-banner`, `.chart`, `.quiz-opt`, `.predict`, `.reflect textarea`, `.sidebar-toggle`);
  add new classes rather than renaming these.
- Do design for the textarea fallback editor as well as CodeMirror; both inherit the editor tokens.
