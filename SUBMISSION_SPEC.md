# NeurIPS 2026 PhysUnderstand Submission Specification

Verified on **2026-08-25** against the current official workshop site and the live OpenReview venue configuration.

## Target

- Workshop: **Physical Understanding for Decision-Making: Bridging Foundation Models and Reliable Agents**.
- Venue: NeurIPS 2026 Workshop, Sydney.
- Authoritative workshop page: <https://sites.google.com/view/neurips-2026-workshop-pudm>
- OpenReview venue: <https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/PhysUnderstand>

## Verified requirements

| Item | Verified requirement | Source |
|---|---|---|
| Submission deadline | **August 26, 2026, 08:00 UTC** | Workshop page and OpenReview submission invitation |
| Track selected for this paper | **Full Paper** | OpenReview paper-type field |
| Main-text page limit | Full Paper: **up to 9 main pages**. Tiny Paper: up to 4 main pages. All submissions must have at least 2 main pages. | OpenReview paper-type field and submission email |
| Excluded from main-page limit | References and appendix are unlimited and excluded from the main-page count. | OpenReview paper-type field and submission email |
| Template | PDF must use the official **NeurIPS 2026** template. | OpenReview PDF field; official template linked by the NeurIPS 2026 CFP |
| Anonymity | **Fully anonymized** at submission. | OpenReview PDF field and submission email |
| Submission system | OpenReview venue `NeurIPS.cc/2026/Workshop/PhysUnderstand`. | Live OpenReview group/invitation |
| PDF | One PDF upload, PDF extension, maximum 100 MB. | OpenReview PDF field |
| Optional supplementary material | One fully anonymized ZIP, maximum 100 MB; examples include code, videos, or demos. | OpenReview supplementary-material field |
| OpenReview profiles | Every author must have an OpenReview profile before submission. | OpenReview authors field |
| Required metadata | Title, authors, paper type, keywords, abstract; TL;DR is optional. | OpenReview form |
| Administrative confirmations | Author-email sharing with program chairs; public release of accepted paper and author names after the conference; non-archival/dual-submission confirmation. | OpenReview form |
| Reviewer nomination | Nominate one eligible author reviewer (up to three assignments), or enter `None`; the field does not affect the decision. | OpenReview form |
| License | Submission form currently exposes **CC BY 4.0**. | OpenReview form |
| Archival status | **Non-archival.** Work under review elsewhere is welcome; already published or accepted work is ineligible during review. | OpenReview dual-submission confirmation |

## Manuscript configuration

- Build with `\usepackage[dblblindworkshop]{neurips_2026}` and no `preprint` or `final` option; set the required `\workshoptitle`.
- Keep author names, affiliations, acknowledgments, identifying URLs, and identifying repository references out of the submission PDF.
- Target at most 9 pages through the conclusion/limitations; references and appendix follow.
- Keep reproducibility metadata anonymous: commit hashes and artifact counts are acceptable, but no identifying repository URL is included.

## Unresolved details

The current workshop website exposes the deadline and scope but does not publish a separate prose CFP with all formatting details. The live OpenReview invitation supplies the requirements above. The following points are not explicitly resolved by either source:

- whether an appendix must be concatenated into the paper PDF or may be supplied only in the optional ZIP;
- whether a NeurIPS paper checklist is required for this workshop submission;
- notification and camera-ready deadlines;
- whether accepted papers are posted before the conference (the form only confirms public release after the conference);
- exact workshop day (the workshop page says December 11 or 12).

The current source tree includes an appendix in the same PDF and omits the main-conference checklist from the submission build unless the organizers clarify that it is required.

## Official template provenance

Downloaded from the NeurIPS 2026 Call for Papers link:
<https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip>
