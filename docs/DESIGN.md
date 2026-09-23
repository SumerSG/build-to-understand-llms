# Design

## Identity: the atelier

The lab's visual identity is monochrome and typographic, after Yohji Yamamoto: paper (#f3f1ea) and ink (#101010) in the light mood, black (#0b0b0b) and bone (#ecebe5) in the dark one; one serif with drama (Cormorant Garamond, falling back to Iowan Old Style, Palatino, Georgia) for anything that speaks, and wide-tracked 11px uppercase Helvetica for anything that labels; rules instead of boxes; no rounded corners; no shadows; one red (#b3121b) reserved for failure; a single indulgence, the faint 260px module numeral behind each module header. Buttons are rectangles that invert on hover. Charts use an editorial palette validated with the dataviz palette validator on both surfaces (all checks pass, no warnings): indigo, vermilion, teal, ochre, plum, olive, sky, rust, in that fixed order, with darker/lighter steps per mood. Indigo leads so a single-series chart never reads as the failure red. Heatmaps use a single paper-to-ink ramp (`--heat-lo` → `--heat-hi`). Legend swatches are short rules, not dots. The section that implements this is the last one in `app/styles.css` ("16. ATELIER"); the sections before it are the structural base it re-tokens.


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

The atelier section re-tokens every radius below to 0 and removes the shadows; the table records the
structural base only.

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
  `#btn-solution`, `#sol-copy` (loads the whole reference), `#btn-run`, `#btn-run-ref`, `#btn-complete`, `.step-chip`, `.results .status-line`,
  `.goal-out .done-banner`, `.goal-out .status-line.bad`, `.test.fail`, `.chart`, `.quiz-opt`, `.predict`,
  `.reflect textarea`, `.sidebar-toggle`); add new classes rather than renaming these. The first
  `.status-line` inside `.results` must stay the "N/M tests passed" summary; the pass count beside the buttons
  is `#check-count`, outside `.results`. A demo failure on the learner's code must keep a `.status-line.bad`
  inside `.goal-out` (inside `.demo-fail`) so e2e sees it.
- Don't show module ids. The learner sees one number, the position on the path (`READY` order, from 00);
  the id appears only in tooltips. Never hard-code module counts: the home lede, "Module N of 00–NN" and the
  About page derive them from the registry.
- Don't let anything scroll the page sideways at 390px: wide tables (`.table-wrap`, `.chart-body`), code
  blocks and horizontal rows (`.steps-nav`, `.phases`) scroll inside their own box; inline code wraps. Scroll a
  row to its active item with `scrollIntoRow()` in `main.js`, never `scrollIntoView()` (which moves the page).

## Beginner on-ramp

Section 17 of `app/styles.css`. Same identity: rules, not boxes; tracked 11px caps for labels; the serif
for anything that speaks.

- **`:::plain`** (markdown) renders `.callout.callout-plain`: a hairline-ruled box with the tracked label "In
  plain words" and the text in the display serif at 21px. It opens a concept.
- **`:::deeper <title>`** renders `details.deeper`, collapsed by default, between two strong rules; the
  summary is a tracked-caps label with a serif +/− at the right. Containers nest (a deeper block may hold a
  note), and a `:::` inside a fenced code block does not close anything.
- **Module references** (`a.modref`): markdown.js rewrites "module NN", "modules NN and MM", "modules NN–MM"
  and "module-NN" in prose (never in code) to the path number, linked, with the title as tooltip; unknown ids
  are left alone. Inside buttons (quiz options) they are `span.modref`. `configureModuleRefs`, `moduleLink`
  and `linkModuleRefs` are exported for `main.js` (goal, threshold, prereqs, chat intro).
- **Home hero**: headline and lede, then `.ways` with two `.way` columns ("Build it" primary, "Just read"
  secondary, serif italic headings) directly under the lede so both ways in are on the first phone screen;
  then `.hero-facts` (what it is / who it is for / what you need, three ruled columns, one on phones; "what you
  need" says reading and demos work on a phone and building is easier on a laptop), `.js-note` (guarded: only
  when `35-javascript` is registered) and `.time-note`.
- **Reader's shortlist** (`#/read`, `renderReader()` and `READER_LIST` in `main.js`): "Just read" leads here,
  not to the build order. `.shortlist` of `.short-item`s, ruled like the track lists: a serif italic topic,
  a tracked-caps honest time ("About 10–15 min": plain box, concept skim, reference demo), a serif one-line
  promise, the module(s) by path number, and a "Read it" button to the Concept (which sets `btu:path = reader`).
  Entries whose modules are not on the path drop out. Linked from the sidebar, About and Where next.
- **Build screen guide** (`.screen-help`): a ruled note with a numbered list and a "Got it" button, shown on
  the first Build visit and remembered in `localStorage` (`btu:buildHelpDismissed`); `#show-help` brings it
  back. After a check the check row scrolls into view and `#check-count` says "N/M passed". A stale step
  (`.step-chip.stale`, `.stale-mark` ↻) carries a tooltip explaining it, and `.stale-legend` appears under
  the chips when any step is stale.
- **Reference solution** (`.ref-panel`): always behind a plain, non-scolding confirm; it opens scrolled into
  view and wraps long lines on phones. By default it shows only the current step's code: `stepReference()`
  in `markdown.js` takes the names the step's instructions put in code spans that are top-level declarations
  of `solution.js`, narrowed to the ones the starter marks "TODO: step N" (plus helpers the starter lacks).
  `#sol-whole` toggles the whole file; `#sol-copy` always loads the WHOLE reference into the editor.
- **Module numbers in messages**: test names, failure messages, error lines and console lines go through
  `moduleRefsText()` (plain text, no links), so "module 15" in a message shows the path number.
- **Hints**: the padding sits on `.hint-inner > div` (inside the clipping element) and a closed hint body is
  `visibility: hidden`, so a locked or folded hint shows nothing.
- **Concept end** (`.concept-end`): "I'm ready to build" (`#btn-to-build`) and "Just reading? See it working"
  (`#btn-see-demo`, to the Goal tab); in reader mode the demo link comes first and is primary.
- **Reader path on Goal**: `.reader-offer` with `#btn-run-ref` runs the demo on `solution.js` without
  recording the goal. For a learner with no passing step it comes first (`.reader-offer.first`, primary) and
  the own-code run (`.own-run`, `#btn-run`) follows under "Built the steps?"; otherwise the own-code run leads.
  Each button has its own status (`#goal-state`, `#ref-state`), and `#btn-stop` moves beside the run in
  progress, so a reference run never reports under the "your code" button. A reference run opens with
  `.ref-note` ("This run uses the reference code, so 'your' in the output below means the reference", a
  ruled serif line) and ends in `.ref-banner`, not `.done-banner`. A run on your own code removes the
  "(last run)" banner (`.done-banner.last-run`) so the summary never shows twice. A failure on the learner's
  code shows `.demo-fail` (a plain sentence, the reference offer, and the stack folded in `details.tech`).
  Choosing "Just read" stores `btu:path = reader`, which makes the reference button primary and opens
  unstarted modules on Concept.
- **Recall intro**: the first module on the path says its questions are everyday ideas, no JavaScript needed.
- **Phones**: `pre.code` wraps (`pre-wrap`, `overflow-wrap: anywhere`) under 800px so right-hand comments stay
  on screen. A wide table (`.table-wrap`, `.chart-body`, `.chart-table`) that overflows gets `.is-scrollable`
  from `markScrollables()` in `main.js`: a `.scroll-cue` ("Scroll sideways for more →") above it, a mask fade at
  the right edge until it is scrolled to the end (`.at-end`), and a sticky first column on the page colour with
  a hairline (capped at 40vw on phones).
- **Where next** (`.where-next`, Reflect of the first two modules): "Next module" plus jumps to tokens, the
  KV cache and serving cost, each labelled with its path number.

- Do design for the textarea fallback editor as well as CodeMirror; both inherit the editor tokens.
