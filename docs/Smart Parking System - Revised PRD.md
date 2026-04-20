# Smart Parking System - Revised PRD

Use [docs/prd.md](/Users/thebkht/Projects/smart-parking-system/docs/prd.md) as the canonical PRD.

This mirror exists only as a friendly filename for sharing. It should not diverge from the canonical v3 definition:

- fixed ROIs are the default Stage 1 path
- Stage 2 patch classification is the main ML accuracy target
- the runtime contract is `spots` + `confidence` + UTC `timestamp`
- backend persistence stores the edge payload as-is
