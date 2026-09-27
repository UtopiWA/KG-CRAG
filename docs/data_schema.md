# Data schema

The canonical Python contracts are in `src/kg_crag/models/domain.py`. Raw corpus files are immutable inputs. Interim and processed records must be reproducible from raw inputs plus versioned configuration.

Stable identifiers use the following precedence: DOI, arXiv ID, Semantic Scholar ID, then a normalized-title hash. Graph claims must include their source chunk, extractor version, confidence, and creation time when graph extraction is implemented.

