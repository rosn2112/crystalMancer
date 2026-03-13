# Crystal Mancer

**AI-Driven Catalyst Discovery via Diffusion-Guided GNN + Literature Intelligence**

> Crystal Mancer is an AI catalyst discovery engine that learns synthesis–structure–performance relationships from literature and crystallographic data, then generates new crystal structures using diffusion-guided GNN models, validated via DFT feedback — targeted specifically at the electro- and photocatalysis design space.

---

## 🏗 Architecture Overview

```
INPUT: Performance Target + Application Tag + Lab Constraints
         ↓
[CONDITIONING ENCODER]         ← Embed targets as context vectors
         ↓
[DIFFUSION-GUIDED GNN]        ← ⚠️ Core architecture (redacted)
         ↓
[CRYSTAL GRAPH OUTPUT]        ← CIF via pymatgen
         ↓
[VALIDATION LAYER]            ← MatterSim + CGCNN + DFT flags
         ↓
OUTPUT: Ranked candidate structures + synthesis route suggestions
```

## 📦 Phase 1 — Knowledge Extraction Pipeline

This release implements Phase 1: building the **CIF ↔ synthesis ↔ performance triplet database**.

### What it does

1. **Downloads** perovskite-family oxide CIFs from the Crystallography Open Database (COD)
2. **Filters** by space group (Pm-3m, R-3c, Pbnm, etc.) and ABO₃ stoichiometry
3. **Retrieves** related papers from Semantic Scholar + CrossRef (free APIs)
4. **Extracts** synthesis method, application type, and performance metrics using rule-based NLP
5. **Stores** structured JSON per CIF and generates a summary report

### Output Schema

```json
{
  "cif_id": "COD-1234567",
  "composition": "LaCoO3",
  "spacegroup": "R-3c",
  "spacegroup_number": 167,
  "papers": [{
    "doi": "10.1039/xxxxx",
    "synthesis_method": "sol-gel",
    "application": "OER",
    "performance": {
      "overpotential_mV": 320,
      "tafel_slope_mV_dec": 58,
      "current_density_mA_cm2": 10
    }
  }]
}
```

## 🚀 Quick Start

```bash
# Clone & install
git clone https://github.com/yourusername/crystalMancer.git
cd crystalMancer
pip install -e ".[dev]"

# Dry run (no API calls — validates pipeline with sample data)
crystalmancer --dry-run

# Process 10 CIFs with 3 papers each
crystalmancer --max-cifs 10 --max-papers 3 -v

# Use existing CIFs on disk
crystalmancer --skip-download --cif-dir ./my_cifs -v
```

## 🧪 Testing

```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
crystalmancer/
├── cli.py              # CLI entry point
├── pipeline.py         # End-to-end orchestrator
├── config.py           # Constants & configuration
├── cif/
│   ├── downloader.py   # COD bulk CIF downloader
│   └── filter.py       # Perovskite space group + composition filter
├── literature/
│   ├── semantic_scholar.py
│   ├── crossref.py
│   └── retriever.py    # Per-CIF paper retrieval + caching
├── extraction/
│   ├── synthesis.py    # Synthesis method classifier
│   ├── application.py  # Application type classifier
│   ├── performance.py  # Performance metric extractor
│   └── extractor.py    # Unified extraction pipeline
├── storage/
│   └── json_store.py   # Structured JSON per-CIF writer
└── reporting/
    └── summary.py      # Pandas-based summary report
```

## ⚠️ Redaction Notice

This is a functional but intentionally simplified version of the codebase. The full architecture — including the performance conditioning mechanism, symmetry-aware graph encoding, and score network internals — is available for discussion.

## 📊 Free Data Sources

| Dataset | Size | Access |
|---------|------|--------|
| COD | 500k+ CIFs | Bulk download |
| Materials Project | 154k structures | `mp_api` |
| OMat24 | 110M DFT calcs | HuggingFace |
| OC22 | 62k oxide relaxations | fair-chem |
| Semantic Scholar | 200M+ papers | Free API |
| CrossRef | All DOIs | Free, no key |

## License

MIT
