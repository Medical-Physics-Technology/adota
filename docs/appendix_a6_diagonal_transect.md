# Appendix A.6: anti-diagonal transect figure, and what the data actually shows

One figure replaces the previous six (overview map + five full per-case comparisons):

* Figures, in `/scratch/mstryja/DoTA_dataset_v2/robustness_figures_combined/diagonal_thoracic_LUNG1-006_e80/`, two independent files so each can be placed on its own:
  * `diagonal_transect_thoracic_LUNG1-006_e80_angle_map.{pdf,svg,png}` = panel (a)
  * `diagonal_transect_thoracic_LUNG1-006_e80_panels.{pdf,svg,png}` = panels (b)-(f)
* Per-position metrics: `..._metrics.csv` (same directory)
* Generator: `scripts/mc/plot_diagonal_transect.py` + `scripts/mc/config_diagonal_transect.yaml`
* Logic: `src/mc_generation/diagonal_transect.py` (extraction/metrics), `src/figures/diagonal_transect.py` (rendering)

Nothing was re-simulated and nothing was re-inferred; the figure reads arrays that
already exist on disk.

## Source arrays

Experiment dir `/scratch/mstryja/DoTA_dataset_v2/beamlet_angle_robustness_thoracic_LUNG1-006_e80_v2`,
stems `a{ix:02d}_{iy:02d}` with `iy = 17 - ix`, so diagonal position `k` is `a{k:02d}_{17-k:02d}`
(position 00 = `a00_17`, position 17 = `a17_00`):

| Array | Shape | Used for |
|---|---|---|
| `{stem}_ct.npy` (int16) | (60, 60, 320) | panel (b) HU |
| `{stem}_flux.npy` (f64) | (60, 60, 320) | lateral weights for the HU profile |
| `{stem}_ds.npy` (f32) | (60, 60, 320) | panels (c), (d), (b) R80(MC), panel (f) |
| `{stem}_ds_pred.npy` (f32) | (1, 1, 320, 60, 60) → (60, 60, 320) | panels (d), (b) R80(ADoTA), panel (f) |
| `{stem}_sim_res.json` | n/a | angles, spacing, entrance, patient/energy metadata |

Gamma pass rates come from the persisted grid
`robustness_figures_combined/grids/thoracic_LUNG1-006_e80_grids.npz`, key `g1_3_0p1`
(the Γ(1%, 3mm, 0.1%) grid that produced the existing panels), used for both the
inset map (a) and the line panel (e). **All 18 positions have complete data; no
position is missing.**

The ROI is *field*-aligned, not beamlet-aligned: a beamlet at θ ≠ 0 crosses that grid
obliquely. The HU profile in panel (b) is therefore a **flux-weighted lateral mean**
rather than a single central column: the fluence has σ ≈ 4 mm here, so the beamlet samples a
finite tube, and weighting by the (already angled) flux follows the beamlet's own drift
with depth without any ray tracing. Cross-checked against a 2 mm-radius disc around the
flux centroid: the two agree on every feature; the flux-weighted version damps thin
structures somewhat (the rib at position 02 reads +249 HU flux-weighted vs +437 HU in the
narrow core), which is physically correct for a finite-width beamlet.

Panels (b)–(d) are laterally integrated, whereas panel (e) is the full 3-D gamma on the
whole beamlet volume. Panel (e) is therefore *not* derivable from panels (c)/(d); they
are independent views of the same beamlets.

## Definitions

**Sampling.** The angular grid has $N=18$ steps over $[\theta^-,\theta^+]=[-2^\circ,+2^\circ]$,

$$\theta_k=\theta^-+k\,\frac{\theta^+-\theta^-}{N-1},\qquad k=0,\dots,N-1 .$$

Cell $(i,j)$ is the beamlet with $(\theta_x,\theta_y)=(\theta_i,\theta_j)$. The sampled
anti-diagonal is $p\mapsto(i,j)=(p,\,N-1-p)$, and because the range is symmetric,

$$\theta_y^{(p)}=\theta_{N-1-p}=-\theta_p=-\theta_x^{(p)} ,$$

so the transect runs from $(-2^\circ,+2^\circ)$ at $p=0$ to $(+2^\circ,-2^\circ)$ at $p=17$.

**Why the anatomy changes with $p$.** The incidence angle is produced by displacing the
spot on the divergent beam,

$$x_s=d_{\mathrm{SM}x}\tan\theta_x,\qquad y_s=d_{\mathrm{SM}y}\tan\theta_y ,$$

so $p$ changes the entry point as well as the direction. Over this transect the spot moves
$\pm70$ mm in $x$ and $\mp90$ mm in $y$, sweeping the ray across the thorax. This is the
whole point of the appendix: $p$ is not a pure angle scan.

**Volumes.** For each $p$ the experiment dir holds four arrays on the field-aligned ROI
grid $\Omega=\{0,\dots,59\}^2\times\{0,\dots,319\}$ with isotropic $\Delta=1$ mm, indexed
$(u,v,w)$ with $w$ along the field axis and depth $z_w=w\Delta$:

$$H_p(u,v,w)\ [\mathrm{HU}],\quad \Phi_p(u,v,w)\ [\text{fluence}],\quad
D_p^{\mathrm{MC}}(u,v,w),\quad D_p^{\mathrm{AD}}(u,v,w).$$

The ROI is aligned to the *field*, not to the beamlet, so a beamlet with $\theta\neq0$
crosses it obliquely.

**Panel (b), flux-weighted HU.** Rather than the central column, take the fluence-weighted
lateral mean at each depth,

$$h_p(w)=\frac{\sum_{u,v}\Phi_p(u,v,w)\,H_p(u,v,w)}{\sum_{u,v}\Phi_p(u,v,w)} ,$$

falling back to the centre voxel where the denominator vanishes. Because $\Phi_p$ already
carries both the beamlet's lateral drift with depth and its finite spot width
($\sigma\approx4$ mm), $h_p$ is the HU the beamlet actually samples, with no ray tracing.

**Depth-dose curves.** Integrate laterally,

$$d_p^{X}(w)=\sum_{u,v}D_p^{X}(u,v,w),\qquad X\in\{\mathrm{MC},\mathrm{AD}\} .$$

Panel (c) shows $d^{\mathrm{MC}}$ normalised per position, panel (d) the signed difference
against the *same* denominator, so both are read on one per-position scale:

$$c_p(w)=100\,\frac{d_p^{\mathrm{MC}}(w)}{\max_{w'}d_p^{\mathrm{MC}}(w')},\qquad
\delta_p(w)=100\,\frac{d_p^{\mathrm{AD}}(w)-d_p^{\mathrm{MC}}(w)}{\max_{w'}d_p^{\mathrm{MC}}(w')} .$$

**Distal range $R_f$.** For a curve $d$ with peak index $w^\*=\arg\max_w d(w)$, let
$j=\min\{w>w^\*: d(w)<f\,d(w^\*)\}$. Linear interpolation between the bracketing samples
gives

$$R_f[d]=z_{j-1}+\frac{d(j-1)-f\,d(w^\*)}{d(j-1)-d(j)}\,(z_j-z_{j-1}) ,$$

undefined if the curve never drops below the threshold inside the ROI. Panel (b) overlays
$R_{80}^{\mathrm{MC}}=R_{0.8}[d_p^{\mathrm{MC}}]$ and
$R_{80}^{\mathrm{AD}}=R_{0.8}[d_p^{\mathrm{AD}}]$; panel (f) plots
$|\Delta R_{80}|=|R_{80}^{\mathrm{AD}}-R_{80}^{\mathrm{MC}}|$.

**Surface and path composition.** The entry depth is the first crossing of an air/tissue
threshold, $s_p=\min\{z_w: h_p(w)>-500\}$, and the traversed segment is
$S_p=\{w: s_p\le z_w<R_{80,p}^{\mathrm{MC}}\}$ with physical length
$L_p=R_{80,p}^{\mathrm{MC}}-s_p$. On that segment,

$$f_p^{\text{lung}}=\frac{100}{|S_p|}\,\bigl|\{w\in S_p: h_p(w)<-300\}\bigr| ,$$

and likewise $f^{\text{soft}}$ for $-300\le h_p<150$ and $f^{\text{bone}}$ for
$h_p\ge150$. Two shape descriptors complete the set: the mean HU over the last 20 mm
before $R_{80}$,

$$\bar h_p^{\text{distal}}=\operatorname*{mean}_{R_{80,p}-20\le z_w<R_{80,p}}h_p(w) ,$$

and the width of the dose plateau,

$$w^{90}_p=\Delta\,\bigl|\{w: d_p^{\mathrm{MC}}(w)>0.9\max_{w'}d_p^{\mathrm{MC}}(w')\}\bigr| ,$$

which separates a sharp Bragg peak (2 to 4 mm) from a flattened dome (up to 31 mm here).

**Panel (e), gamma.** $\Gamma_p$ is the 3-D *local* gamma pass rate of the prediction
against the MC reference over the whole volume, criterion $(\delta_D,\Delta_d)=(1\%,3\,\mathrm{mm})$:

$$\gamma(\mathbf r)=\min_{\mathbf r'}\sqrt{\frac{\lVert\mathbf r'-\mathbf r\rVert^2}{\Delta_d^2}
+\frac{\bigl(D^{\mathrm{AD}}(\mathbf r')-D^{\mathrm{MC}}(\mathbf r)\bigr)^2}
{\bigl(\delta_D\,D^{\mathrm{MC}}(\mathbf r)\bigr)^2}} ,\qquad
\Gamma=100\,\frac{|\{\mathbf r\in\Omega_c:\gamma(\mathbf r)\le1\}|}{|\Omega_c|} ,$$

evaluated with 5-fold interpolation and $\gamma$ capped at 2. The evaluated set $\Omega_c$
is the voxels above a 0.1 % cutoff of the reference peak, the peak being the 99.5th
percentile of the positive MC voxels; the volume is first cropped to the high-dose
bounding box dilated by more than $\Delta_d$, which leaves $\Gamma$ unchanged.

Note that $\Gamma_p$ is computed on the full 3-D dose, whereas panels (b) to (d) are
laterally integrated. Panel (e) is therefore **not** derivable from panels (c) and (d);
they are independent views of the same beamlets.

**Verification statistics.** Associations between $\Gamma_p$ and the geometric scalars
$(L_p, f_p^{\text{lung}}, \bar h_p^{\text{distal}}, w^{90}_p, |\Delta R_{80,p}|)$ are
reported as Spearman rank correlations $\rho$, over all 18 positions and over the left
half $p=0,\dots,9$ separately.

## Cross-check against the values supplied

Γ(1%, 3mm, 0.1%) read from `g1_3_0p1[k, 17-k]`, floored, reproduces the filename values
exactly for all 18 positions:

```
pos     00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17
expect  88 84 85 82 83 81 80 85 82 85 88 92 94 94 94 94 93 93
found   88 84 85 82 83 81 80 85 82 85 88 92 94 94 94 94 93 93
```

MC Bragg peak (argmax of the integrated depth dose, measured along the field axis from
the ROI front face): **135 mm at position 00** (quoted ≈134), **166 mm at position 06**
(quoted ≈164), **91 mm at position 17** (quoted ≈90). Positions 00 and 17 agree within
one voxel.

> **Caveat on position 06.** Its depth dose is flat over 31 mm at >90% of maximum, so
> `argmax` is not a meaningful range statistic there: a 2 mm shift in the quoted peak
> depth is noise, and any number in a ~30 mm band is defensible. Quote R80 = 185.7 mm
> instead if a range number is needed for that position.

## Does the claimed geometry/accuracy correspondence hold?

**Mostly yes, with one factual correction and one over-claim to drop.**

### Confirmed: the accuracy step at 10–11 coincides with a geometric transition

| pos | Γ [%] | path surface→R80 [mm] | lung fraction of path [%] | mean HU, last 20 mm | \|ΔR80\| [mm] |
|---|---|---|---|---|---|
| 09 | 85.5 | 60 | 23 | −514 | 3.6 |
| **10** | **88.6** | **51** | **6** | **−73** | **4.1** |
| **11** | **92.8** | **49** | **2** | **+12** | **0.2** |
| 12 | 94.2 | 49 | 2 | −41 | 0.1 |

Every geometric variable turns over in the same two columns as the gamma step. Across
all 18 positions the association is strong: Spearman ρ(Γ, path length) = −0.92
(p = 7e−8), ρ(Γ, lung fraction) = −0.88 (p = 2e−6), ρ(Γ, distal HU) = +0.88 (p = 1e−6).
The two halves are cleanly separated: Γ = 84.0 ± 2.4 % over positions 00–09 versus
93.8 ± 0.6 % over positions 11–17, with position 10 intermediate in both accuracy and
geometry. Nothing in the incidence angle itself distinguishes position 09 from position
11: |θ| is 0.12° and 0.59°, both far smaller than |θ| = 2.0° at position 00, which is
the *best* of the left half.

The distal-end description also holds: on the right half the ray stops in soft tissue
(mean HU over the last 20 mm between −41 and +51) with a sharp falloff (2–4 mm above 90%
of maximum, |ΔR80| ≤ 0.5 mm); on the left half it stops in or just past lung
(−117 to −904 HU) with |ΔR80| up to 3.6 mm.

### Correction: there is no rib at position 06

The claim that position 06 crosses a rib near the entrance is **not supported**. Along
that ray the HU never exceeds −11 in the first 40 mm past the surface (narrow-core
sampling; the flux-weighted profile agrees). The ribs on this transect are at **positions
02 and 03**, with 7 mm and 6 mm of bone peaking at +437 and +455 HU in the narrow core,
about 20–25 mm past the surface.

The broad flattened dome at position 06 **is** real and **is** the widest on the
transect (31 mm above 90% of maximum, against 2–4 mm on the right half), but its cause is
different: the Bragg peak lands *inside* lung parenchyma (mean HU −904 over the last
20 mm before R80), and at ρ ≈ 0.1 the peak is stretched by roughly 1/ρ in physical
distance. Suggested rewording: *"position 06 additionally has its peak fall inside the
lung parenchyma rather than at a tissue interface, stretching the Bragg peak into a broad
flattened dome."*

Related nuance worth keeping out of the claim: a long lung path does **not** by itself
produce a dome. Positions 03 and 04 have the *longest* lung paths on the transect (100 mm
each) yet the sharpest peaks of the left half (4 mm above 90%, |ΔR80| = 0.8 mm), because
the ray terminates at a lung/mediastinum interface that truncates the peak. Depth-dose
*shape* follows what the ray stops in; gamma follows the whole path.

### Does not hold: the left-half variation is not explained by path composition

The 80–85 % band over positions 00–09 is a **plateau with scatter, not a gradient**.
Within that half no geometric scalar orders it: ρ(Γ, lung fraction) = −0.55 (p = 0.10),
ρ(Γ, peak width) = −0.35 (p = 0.32), ρ(Γ, |ΔR80|) = −0.05 (p = 0.88). Concretely:

* **Position 08 contradicts the ordering.** It has a short, comparatively dense path
  (64 mm, 30 % lung), geometrically closer to position 00 (63 mm, 29 % lung, Γ = 88.9 %)
  than to position 06, yet Γ = 82.9 %, i.e. among the lowest on the transect.
* **Position 02** carries the largest range error of the left half after 09
  (|ΔR80| = 3.3 mm) and a rib, yet scores Γ = 85.1 %, *above* positions 03–06, which have
  sub-millimetre range errors.
* The per-beamlet MC statistical uncertainty is 0.55 %, so part of the ±2 % scatter is
  simply noise; the figure leaves it unsmoothed, as instructed.

**Recommended claim for the appendix:** the *step* between the two halves is explained by
path composition (length, lung fraction, distal density) and not by incidence angle. The
*residual scatter within the lung-dominated half* is not attributable to any single
geometric scalar we can measure from these 18 beamlets, and we should say so rather than
imply a monotone trend across positions 00–09.

## Figure construction notes

* Two files: panel (a) at 3.8 × 3.1 in (single-column), panels (b)-(f) at 7.1 × 7.8 in
  (full text width). Vector PDF and SVG plus a 300 dpi PNG for each. Only the heatmap
  meshes are rasterised inside the vector files (`rasterized=True`), so the PDFs stay
  small while all text, axes and overlay curves remain vector.
* Both are laid out with `fig.subplot_mosaic`, so panes and their colorbars are placed by
  name rather than by index arithmetic.
* No prose inside the figures; everything descriptive belongs in the LaTeX caption.
* The `(θx, θy)` pairs sit on a secondary x axis below the index ticks of panel (f), one
  line per pair, so a pair is never broken across lines or squeezed into one column's
  width. Which positions are annotated is set by `annotate_positions` in the config.
* Fonts are sized for A4 print through a single `font_scale` knob (`FONT_SCALE = 1.3`,
  overridable per figure in the config). Base sizes are colorbar labels 10 pt, colorbar
  ticks 9 pt, axis labels 9.5 pt, ticks 8.5 pt, legend and bracket labels 9 pt; at 1.3x
  these become 13 / 11.7 / 12.35 / 11.05 / 11.7 pt.
* **Watch the mathtext, not the nominal sizes.** Matplotlib draws sub- and superscripts at
  0.7x, so the "80" in `$R_{80}$` is the smallest glyph in the figure, not any label the
  code sets explicitly. An earlier revision was flagged at 4.9 pt on the page: that was
  the legend's `$R_{80}$` subscript, 8 pt base -> 5.6 pt subscript -> 4.9 pt at the ~0.875
  page scale. The legend size was therefore lifted to 9 pt base so the mathtext-bearing
  text is no longer the outlier. The smallest glyph is now 8.19 pt in a figure authored
  7.85 in wide, so at a printed width of W inches it renders `8.19 * W / 7.85` pt: it
  clears a 6 pt floor for any W >= 5.75 in. The angle map's smallest glyph is 8.64 pt at
  3.94 in authored, clearing 6 pt for any W >= 2.74 in.
* To re-check after any edit, pull the `Tf` operators straight out of the PDF rather than
  trusting the source constants, since the mathtext sizes never appear in the source.
* Colorbar labels are auto-fitted: each is measured against its own bar height and the
  smallest fitting size is applied to all three, so they stay uniform and none overruns
  into the neighbouring panel (`_style_cbars`, floor 9 pt). All three currently fit at the
  full 10 pt. Panel (c)'s colorbar is in % of the per-position maximum rather than a 0-1
  fraction, which both shortens the label and puts it on the same scale as panel (d).
* Both label columns are pinned to a common vertical line by `_align_axis_labels`: the
  five panel y labels on the left, the three colorbar labels on the right. Left to
  matplotlib each label is pushed out by its own tick labels, and "-1000" is far wider
  than "100", so the stacks come out ragged. All labels are also two lines, quantity then
  unit, so the blocks are the same width and the alignment reads cleanly.
* No panel letters are drawn inside either figure; label the sub-panels from LaTeX
  (`subcaption`/`subfigure`) so the lettering matches the manuscript's own scheme.
* Panel (e) as specified would have needed two y scales (Γ in %, |ΔR80| in mm) on one
  axes. It is split into (e) and (f), stacked and sharing the x axis, so neither quantity
  is read off a scale that does not belong to it.
* Colour: grayscale for HU (the CT convention, and achromatic ramps are safe under any
  colour-vision deficiency; window −1000 to +300 HU so lung, soft tissue and rib separate);
  `viridis` for the sequential dose, matching the existing grid panels; `RdBu_r`, a
  ColorBrewer colourblind-safe diverging pair with a neutral midpoint, centred on zero
  and symmetric at ±20 % for panel (d), with `extend="both"` for the handful of voxels
  beyond (min −19.3 %, max +33.6 %, both at distal interfaces). Overlay lines use the
  Okabe-Ito palette.
* Depth is shown over 30–235 mm along the field axis (measured from the ROI front face,
  not from the patient surface, which is drawn as the green dotted line). Every axis
  carries its unit.
