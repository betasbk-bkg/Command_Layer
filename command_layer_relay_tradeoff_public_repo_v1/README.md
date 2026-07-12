# Command-Layer Resilience and Relay Over-Allocation Trade-Offs

This repository package supports the associated manuscript:

**Command-Layer Resilience and Relay Over-Allocation Trade-Offs in Heterogeneous Multi-Robot Teams Under Delayed Hazard Information**

The package contains simulation code, Webots validation files, saved seed-level results, analysis tables, six core figures, citation metadata, a validation summary, and a claim-level reproducibility checker. It does not include the manuscript file or pre-release reproducibility logs.

Archived release: Zenodo v1.1.0, https://doi.org/10.5281/zenodo.21311023.

## Main Claim Boundary

The package supports a conservative claim:

- Command-layer resilience is environment-dependent.
- Relay over-allocation reduces effective delay but can harm safe delivery and attrition.

It does not claim physical robot deployment, human-subject validation, unconditional command-layer superiority, or relay-rich superiority.

## Funding and AI Disclosure

- Funding: This research received no specific grant funding from any funding agency in the public, commercial, or not-for-profit sectors.
- Use of generative AI: Generative AI tools were used for language refinement, editorial drafting, and code organization. They were not used to generate data, select models, derive results, or make scientific claims.

## Quick Check

```powershell
python scripts/reproducibility_check.py --root .
```

The pre-release gate used before packaging required at least three independent checker runs with a reproducibility rate of 0.95 or higher. Those logs are not included in the public package; users can rerun the checker with the command above.
