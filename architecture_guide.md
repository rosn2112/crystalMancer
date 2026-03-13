# Crystal Mancer — Architecture Deep Dive

> **Audience: You.** If you've watched Karpathy's NN Zero-to-Hero series and trained
> GPT-style models, you already have the foundation. This explains how we go from
> "I know what a transformer is" to "I can generate novel crystal structures."

---

## 🧠 The Core Insight (Why This Matters)

Existing AI crystal generators (MatterGen, CDVAE, GNoME) can generate **stable**
crystal structures. But they're blind: they don't know what a catalyst *does*.

Crystal Mancer is different: **it generates structures conditioned on catalytic
performance targets.** You tell it "give me a perovskite with OER overpotential
< 300 mV" and it generates candidate lattices + atom positions optimized for that.

---

## 🎲 How Diffusion Works (the "Image Gen but for Atoms" Analogy)

You know how Stable Diffusion starts with noise and gradually denoises it into an image?
**Crystal diffusion does the same thing, but in 3D.**

### Image Diffusion vs Crystal Diffusion

| Component | Image Diffusion | Crystal Diffusion |
|-----------|----------------|-------------------|
| What is noised | Pixel RGB values | Atom 3D coordinates + atom types + lattice |
| Noise type | Gaussian noise on pixels | Gaussian noise on atomic positions |
| Score network | U-Net | E(3)-equivariant GNN |
| Output | Denoised image | Denoised crystal structure |
| Conditioning | Text prompt (CLIP embedding) | Performance targets (overpotential, etc.) |

### Step by Step

```
t=T (pure noise)      → Random positions in a random box
   ↓ GNN predicts "which way should each atom move?"
t=T-1                 → Slightly less noisy positions
   ↓ GNN predicts again
...                   → Positions getting more structured
   ↓
t=1                   → Almost a crystal, small corrections
   ↓
t=0 (clean structure) → Valid crystal: SrTi₀.₅Fe₀.₅O₃ in Pm-3m
```

At each step, the GNN takes in:
1. **Current noisy structure** (atom positions + types + lattice)
2. **Performance targets** (the conditioning)
3. **Current noise level** (timestep embedding)

And predicts: **"how much noise was added"** (the score / ε).
Then we subtract that predicted noise → cleaner structure.

### Why Not Just Use a Transformer?

Crystals are **fundamentally different from text/images**:
- They're **3D point clouds** (not sequences or grids)
- They have **periodic boundary conditions** (atoms repeat infinitely)
- Physics is **rotation-invariant**: rotating a crystal doesn't change its energy
- Bond lengths of 0.001Å matter (need sub-Angstrom precision)

That's why we need **E(3)-equivariant GNNs** instead of transformers.

---

## 🔮 What is E(3) Equivariance?

E(3) = the group of all 3D rotations, reflections, and translations.

**E(3)-equivariant** means: if you rotate the input crystal, the output
rotates the same way. The network respects the symmetry of physics.

```
If you rotate the input crystal by 30° around the z-axis:
  → All predicted forces rotate by the same 30°
  → Predicted energy stays the SAME (it's a scalar)
  → This is NOT guaranteed by a normal MLP or transformer!
```

Without equivariance, the network would need to learn separately that
"LaCoO₃ pointing north" and "LaCoO₃ pointing east" are the same material.
With equivariance, it gets this for free.

### GemNet / SchNet / DimeNet++

These are the specific E(3)-equivariant GNN architectures:

| Architecture | Key Innovation | Speed | Accuracy |
|-------------|---------------|-------|----------|
| **SchNet** | Continuous-filter convolution (Gaussian distance expansion) | Fast | Good |
| **DimeNet++** | Adds bond ANGLES (2-hop message passing) | Medium | Better |
| **GemNet** | Adds DIHEDRAL angles (4-body interactions) | Slow | Best |

**Crystal Mancer uses SchNet-style** because:
- It's the best speed/accuracy tradeoff for our scale
- Karpathy's autoresearch can explore DimeNet++/GemNet automatically
- We're not yet at the scale where GemNet's accuracy matters

---

## 🏗️ Our Architecture (What Each Parameter Does)

### CrystalMancerGNN Parameters

```python
CrystalMancerGNN(
    atom_feature_dim=108,     # 94 one-hot element + 14 scalar properties
    edge_feature_dim=41,      # 1 raw distance + 40 Gaussian basis functions
    hidden_dim=128,           # Size of the learned representation
    num_layers=4,             # How many message-passing rounds
    num_targets=5,            # overpotential, FE, Tafel, j, stability
    global_feature_dim=239,   # 231 space group one-hot + lattice params
    use_conditioning=False,   # True for generation, False for property prediction
    dropout=0.1,              # Regularization
)
```

#### Why 108-dim atom features?
- **94**: One-hot element (H through Pu). This gives the network a unique
  "identity" for each element type.
- **14 scalars**: Electronegativity, ionic radius, mass, valence electrons,
  block encoding (s/p/d/f), plus flags (is_transition_metal, is_rare_earth, is_oxygen).
  These encode chemical knowledge so the network doesn't have to learn
  "Fe is more electronegative than Ca" from scratch.

#### Why 40 Gaussian basis functions for edges?
This is SchNet's key trick. Instead of telling the GNN "this bond is 2.1 Å long"
(a single number), we expand it into 40 overlapping Gaussian bumps:

```
Distance 2.1Å → [0.01, 0.05, 0.23, 0.78, 0.95, 0.78, 0.23, ...]
                  ↑ center at 0Å         ↑ center at 2.1Å
```

This gives the network a **smooth, differentiable** way to distinguish
"2.1Å bonds" from "2.2Å bonds" — crucial because bond lengths of 0.1Å
difference can mean completely different materials.

#### Why 4 message-passing layers?
- More layers = each atom "sees" farther into the crystal
- Layer 1: atoms know their immediate neighbors (1st coordination shell)
- Layer 2: atoms know their 2nd-shell neighbors
- Layer 4: atoms have a ~4-hop view (covers most of a perovskite unit cell)
- Research shows diminishing returns past 6 layers (oversmoothing)

#### Why hidden_dim=128?
- 128 is the "sweet spot" for our dataset size (~1000-10000 structures)
- Too small (32): underfits, can't capture complex structure-property relationships
- Too large (512): overfits on our dataset, slower on macOS
- Autoresearch will try 64/96/128/192/256 automatically

---

## 🎯 The ConditioningAdapter (How We Tell the GNN What to Generate)

This is what makes Crystal Mancer different from vanilla crystal generators:

```python
class ConditioningAdapter:
    # Input:  target performance values + masks
    # Output: conditioning vector injected into the GNN

    # Example:
    # targets = [280.0, None, 55.0, None, None]
    #             ↑ overpotential  ↑ Tafel slope
    # masks   = [1, 0, 1, 0, 0]
```

During **property prediction** (pretraining): conditioning is OFF.
The GNN learns: structure → performance.

During **structure generation** (fine-tuning): conditioning is ON.
The GNN learns: performance targets → structure.

The adapter encodes the targets as a vector and adds it to the graph
representation at inference time. Think of it like CLIP embeddings for
Stable Diffusion — but instead of "a photo of a cat", it's
"overpotential = 280 mV, Tafel = 55 mV/dec".

---

## 📐 Is the Viewer Mathematically Correct?

**Yes.** The fractional-to-Cartesian conversion uses the standard
crystallographic lattice matrix:

```
     ┌ a         0                                      0                             ┐
L  = │ b·cos(γ)  b·sin(γ)                              0                             │
     └ c·cos(β)  c·(cos(α)-cos(β)cos(γ))/sin(γ)        c·√(remaining)/sin(γ)         ┘
```

This is the **International Tables for Crystallography** convention.
Every CIF viewer (VESTA, Mercury, Avogadro) uses this exact matrix.

**Bond detection** uses a simple distance cutoff (2.8Å), which is
correct for oxide perovskites (Ti-O ≈ 1.95Å, Sr-O ≈ 2.76Å).
For other crystal families, you'd adjust the cutoff.

---

## 🔄 The Full Pipeline Flow

```
                       ┌─────────────────┐
                       │   COD Database   │
                       │  (free CIF files)│
                       └────────┬────────┘
                                │ Download + Filter (perovskites only)
                                ▼
                       ┌─────────────────┐
                       │ Perovskite CIFs │
                       └────────┬────────┘
            ┌───────────────────┼───────────────────┐
            ▼                   ▼                   ▼
  ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
  │ Semantic Scholar  │ │   Europe PMC     │ │    CrossRef      │
  │    + PubMed       │ │    + CORE        │ │    (DOI-based)   │
  └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
           └────────────────────┼────────────────────┘
                                │ Merge + Deduplicate
                                ▼
                       ┌─────────────────┐
                       │  Paper Abstracts │
                       │  (+ full text)   │
                       └────────┬────────┘
                  ┌─────────────┤
                  ▼             ▼
         ┌──────────────┐ ┌──────────────┐
         │  Rule-Based  │ │  LLM (free)  │
         │  Extraction  │ │  Extraction  │
         │  (fallback)  │ │  (primary)   │
         └──────┬───────┘ └──────┬───────┘
                └────────┬───────┘
                         ▼
              ┌─────────────────────┐
              │  JSON Triplets      │
              │  CIF ↔ Synth ↔ Perf │
              └─────────┬───────────┘
           ┌────────────┼────────────┐
           ▼            ▼            ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │ Knowledge  │ │   FAISS    │ │ CIF→Graph  │
    │   Graph    │ │ Embeddings │ │  (PyG)     │
    └────────────┘ └────────────┘ └─────┬──────┘
                                        ▼
                                 ┌─────────────┐
                                 │  GNN Train  │──→ Autoresearch
                                 └──────┬──────┘    (overnight)
                                        ▼
                                 ┌─────────────┐
                                 │  Diffusion  │
                                 │  Generation │
                                 └──────┬──────┘
                                        ▼
                                 ┌─────────────┐
                                 │  DFT / ML   │
                                 │  Validation │
                                 └──────┬──────┘
                                        ▼
                                 ┌─────────────┐
                                 │  RAG Synth  │
                                 │  Planner    │
                                 └─────────────┘
```

---

## References

| Paper | What We Take From It |
|-------|---------------------|
| **MatterGen** (Microsoft, 2024) | Diffusion over atom types + positions + lattice simultaneously |
| **CDVAE** (Xie et al., 2022) | VAE latent space for crystal composition, diffusion for coordinates |
| **GeoDiff** (Xu et al., 2022) | SE(3)-equivariant score matching for molecular geometry |
| **SchNet** (Schütt et al., 2018) | Continuous-filter convolutional neural network for atomistic systems |
| **GemNet** (Gasteiger et al., 2022) | Geometric message passing with dihedral angles |
| **OCP/Open Catalyst** (Meta, 2021) | Large-scale GNN training for catalysis (we use their element features) |
| **GNoME** (DeepMind, 2023) | Graph network for materials exploration (composition generation) |
