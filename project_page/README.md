# PhysPlan project page

This directory is a standalone, static academic project page. It contains only plain HTML, CSS, SVG, and PNG assets; it has no build step, framework, JavaScript, analytics, or external runtime dependency.

## Preview locally

Open `index.html` directly in a browser. For an HTTP preview from this directory, any simple static server also works, for example:

```text
python -m http.server 8000
```

Then visit the local address printed by the server. All page links and assets are relative, so direct-file preview and static hosting both work.

## Anonymous-review state

The current private build intentionally withholds author identity, paper and code links, venue and status information, repository locations, and review-system metadata. Search-engine indexing is also disabled with a `noindex, nofollow` meta tag. The page should remain private until disclosure is appropriate.

## Later deployment with GitHub Pages

Create a new, empty page-only repository and copy **only the contents of this directory** into it. Do not copy the parent research repository, its history, experiments, paper workspace, or internal documentation. After reviewing the files again, enable static hosting from the repository root. No compilation step is required.

The source manuscript and research implementation are deliberately not part of this directory. If code is made public later, link to the separately reviewed public repository instead of copying implementation files into this page repository.

## Post-review checklist

- [ ] add author
- [ ] add paper link
- [ ] add code link if repo is public
- [ ] add venue/acceptance information only after appropriate
- [ ] remove the `noindex, nofollow` directive when public indexing is desired
- [ ] confirm any newly added destination is public and non-identifying as intended
- [ ] rerun public-page audit

The marked comments in `index.html` show where post-review author and project links can be added. They contain no placeholder URLs.
