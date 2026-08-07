import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
INK,MUTED,GRID,BASE,SURF="#0b0b0b","#898781","#e1e0d9","#c3c2b7","#fcfcfb"
BLUE,ORANGE,RED="#2a78d6","#eb6834","#e34948"
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
 "axes.edgecolor":BASE,"text.color":INK,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":MUTED,
 "font.size":11,"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})
# real patient-grouped CV Spearman vs #features (Lasso path) + full-linear + GBM ceiling
nf=[1,2,8,9,14,30]; sp=[0.718,0.732,0.775,0.786,0.799,0.856]
fig,ax=plt.subplots(figsize=(7.2,4.6))
ax.axhline(0.947,ls=":",lw=1.6,color=MUTED); ax.text(30,0.951,"GBM (non-linear) ceiling  0.95",ha="right",va="bottom",color=MUTED,fontsize=9.5)
ax.axhline(0.80,ls="--",lw=1.5,color=RED); ax.text(1,0.806,"target 0.80",color=RED,fontsize=9.5,va="bottom")
ax.plot(nf,sp,"-o",color=BLUE,lw=2.2,ms=7,zorder=5)
for n,s in [(14,0.799),(30,0.856)]:
    ax.annotate(f"{n} terms\nSpearman {s:.2f}", xy=(n,s), xytext=(n-1.5 if n==30 else n+1, s-0.055),
                fontsize=9.5, color=INK, ha="right" if n==30 else "left",
                arrowprops=dict(arrowstyle="-", color=BASE, lw=1))
ax.scatter([14],[0.799],s=140,facecolors="none",edgecolors=ORANGE,lw=2.5,zorder=6)
ax.text(14,0.744,"chosen\ninterpretable",color=ORANGE,fontsize=9,ha="center",va="top",fontweight="bold")
ax.set_xlabel("number of metrics kept in the score"); ax.set_ylabel("held-out Spearman (patient-grouped)")
ax.set_title("Sparsity path — how many metrics the score needs",fontsize=12.5,pad=8)
ax.set_ylim(0.68,0.98); ax.set_xlim(0,31); ax.grid(color=GRID,lw=0.7)
fig.tight_layout(); fig.savefig("/home/mstryja/projects/adota/research/figures/acquisition/fig_sparsity_path.png",dpi=150)
print("wrote fig_sparsity_path.png")
