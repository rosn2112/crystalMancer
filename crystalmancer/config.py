"""Global configuration and constants for Crystal Mancer."""

from pathlib import Path

# ── Project Paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CIF_DIR = PROJECT_ROOT / "data" / "cifs"
DEFAULT_PAPER_CACHE_DIR = PROJECT_ROOT / "data" / "papers"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# ── Perovskite Space Groups ───────────────────────────────────────────────────
# Space group numbers characteristic of perovskite and perovskite-derived
# structures.  Covers ideal cubic, common tilted/distorted variants,
# layered Ruddlesden-Popper and brownmillerite-type phases.

PEROVSKITE_SPACE_GROUPS: set[int] = {
    # Ideal cubic perovskite
    221,  # Pm-3m  (SrTiO3, BaTiO3-HT)
    # Tetragonal distortions
    99,   # P4mm
    123,  # P4/mmm
    127,  # P4/mbm
    140,  # I4/mcm
    # Orthorhombic tilted
    62,   # Pbnm / Pnma  (GdFeO3 type — most common)
    63,   # Cmcm
    36,   # Cmc21
    # Rhombohedral
    167,  # R-3c   (LaAlO3, LaCoO3)
    148,  # R-3
    161,  # R3c
    # Monoclinic
    14,   # P21/n  (low-T distortions)
    12,   # C2/m
    # Layered perovskites (Ruddlesden–Popper, Dion–Jacobson)
    139,  # I4/mmm (RP phases)
    87,   # I4/m
    129,  # P4/nmm
    # Brownmillerite (oxygen-deficient perovskite)
    46,   # Ima2
    74,   # Imma
    # Double perovskites
    225,  # Fm-3m
    87,   # I4/m   (already listed — intentional)
    15,   # C2/c
}

# ── Perovskite Composition Rules ──────────────────────────────────────────────
# Broad A-site and B-site element sets for ABO₃ filtering.

A_SITE_ELEMENTS: set[str] = {
    # Alkaline earth
    "Ca", "Sr", "Ba",
    # Rare earth / lanthanides
    "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Dy", "Y",
    # Alkali (for some layered variants)
    "Na", "K", "Li",
    # Bismuth / Lead
    "Bi", "Pb",
}

B_SITE_ELEMENTS: set[str] = {
    # 3d transition metals
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    # 4d transition metals
    "Zr", "Nb", "Mo", "Ru", "Rh",
    # 5d transition metals
    "Hf", "Ta", "W", "Ir",
    # p-block
    "Al", "Ga", "Sn", "Sb",
}

# ── API Configuration ─────────────────────────────────────────────────────────

# Semantic Scholar
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
S2_RATE_LIMIT_REQUESTS = 95   # stay under 100/5min
S2_RATE_LIMIT_WINDOW = 300    # seconds (5 min)

# CrossRef
CROSSREF_API_BASE = "https://api.crossref.org/works"
CROSSREF_MAILTO = "crystalmancer@research.dev"  # polite-pool identifier

# COD
COD_API_BASE = "https://www.crystallography.net/cod"
COD_SEARCH_URL = f"{COD_API_BASE}/result"
COD_CIF_URL_TEMPLATE = f"{COD_API_BASE}/{{}}.cif"

# ── Rate Limiting ─────────────────────────────────────────────────────────────

BACKOFF_BASE = 1.0       # seconds
BACKOFF_MAX = 60.0       # seconds
BACKOFF_FACTOR = 2.0
MAX_RETRIES = 5
JITTER_MAX = 0.5         # seconds of random jitter

# ── Entity Extraction Sanity Bounds ───────────────────────────────────────────

METRIC_BOUNDS = {
    "overpotential_mV": (50, 1500),
    "faradaic_efficiency_pct": (0.0, 100.0),
    "tafel_slope_mV_dec": (20, 300),
    "current_density_mA_cm2": (0.01, 2000),
    "stability_h": (0.1, 10000),
}
