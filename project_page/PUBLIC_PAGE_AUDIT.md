# Public page audit

## Scope

This audit covers every file recursively contained in this directory. The review checks identity exposure, contact details, filesystem and machine identifiers, repository and review-system links, submission metadata, internal notes, source-code leakage, and image metadata.

## Files that would become public

- `index.html` — standalone project page
- `style.css` — page presentation only
- `assets/overview.svg` — planning-interface diagram
- `assets/overview_mobile.svg` — mobile planning-interface diagram
- `assets/diagnosis_repair.svg` — diagnosis-to-intervention diagram
- `assets/diagnosis_repair_mobile.svg` — mobile diagnosis-to-intervention diagram
- `assets/cross_regime.svg` — chart reconstructed from frozen aggregate results
- `assets/cross_regime_mobile.svg` — vertically stacked mobile version of the same chart
- `assets/og_cover.png` — social preview image
- `README.md` — preview and later deployment instructions
- `PUBLIC_PAGE_AUDIT.md` — this audit

No hidden files or build artifacts are included.

## Identity and private-information checks

- Personal-name or institutional identifiers: none found.
- Email addresses, usernames, and machine names: none found.
- Local, network-mounted, or absolute internal filesystem paths: none found.
- Review-system URLs, tracking identifiers, reviewer material, and exact venue/status metadata: none found.
- Private repository URLs, public repository URLs, and commit identifiers: none found.
- Hidden or private destination URLs: none found. The page contains only local section links and relative asset references.

## Code and internal-material checks

- Experimental source code included: no.
- Research configs, caches, checkpoints, datasets, logs, videos, raw artifacts, manuscript sources, or internal bundles included: no.
- The HTML comments contain only post-review reminders and no private values or destinations.
- The chart and diagrams were rebuilt as page-only SVG assets. They contain labels and frozen aggregate values, not research implementation code or artifact paths.

## Asset checks

- All SVG assets are valid standalone XML and contain no editor metadata or external references. Each dense figure has separate desktop and mobile layouts with the same scientific semantics; the cross-regime versions preserve the same values and uncertainty.
- The PNG is 1200 × 630 pixels in RGB mode and has no embedded metadata fields.
- No asset requires a remote font, script, image, or stylesheet.

## Scientific and rendering checks

- Displayed quantitative results match the frozen aggregate outputs used for the project.
- Statistical boundaries are shown for same-state Local evidence, single-checkpoint Wall evidence, and inconclusive H6 intervention evidence.
- The cross-regime statement is descriptive, and the page does not treat gaps as additive or directly normalized across protocols.
- Desktop and true 390-pixel mobile renders were inspected. The mobile breakpoint selects readable vertical versions of the overview and diagnosis workflow plus the vertically stacked cross-regime chart. The measured mobile document width equals the viewport width, with no page-level horizontal overflow.

## Recommendation

**PASS.** In its current state, the entire directory can be copied into a new page-only static-site repository without exposing research source code, private destinations, or anonymous-review metadata. Any post-review edits must trigger a fresh recursive audit before publication.
