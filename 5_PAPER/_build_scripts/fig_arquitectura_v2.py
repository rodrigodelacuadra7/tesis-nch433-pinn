# -*- coding: utf-8 -*-
"""Figura arquitectura PINN_Modal_v4 — version grafica/detallada."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path
import sys

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- paleta ----
C_IN     = "#EDE7F6"   # entrada
C_ENC    = "#E3F2FD"   # encoders
C_HEAD   = "#FFF3E0"   # heads
C_OUT    = "#FFFDE7"   # outputs
C_PHYS   = "#E8F5E9"   # restricciones fisicas
C_ENC_E  = "#1565C0"
C_HEAD_E = "#E65100"
C_OUT_E  = "#F9A825"
C_PHYS_E = "#2E7D32"
INK      = "#212121"

def box(ax, x, y, w, h, text, fc, ec=INK, fs=9.5, lw=1.3, weight="normal", tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.06",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center",
            fontsize=fs, color=tc, weight=weight, zorder=3)

def arrow(ax, p1, p2, color=INK, lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
        mutation_scale=13, linewidth=lw, color=color, ls=ls, zorder=1,
        shrinkA=0, shrinkB=0))

def dimtag(ax, x, y, txt, color):
    ax.text(x, y, txt, ha="center", va="center", fontsize=8,
            style="italic", color=color, zorder=4,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))

fig, ax = plt.subplots(figsize=(14.5, 9.6))
ax.set_xlim(0, 14.5); ax.set_ylim(0, 9.8); ax.axis("off")

# ================= ENTRADA =================
# edificio (pisos apilados)
bx, by = 0.5, 8.15
for i in range(5):
    ax.add_patch(Rectangle((bx, by+i*0.28), 1.05, 0.24, fc="#B39DDB",
                 ec=C_ENC_E, lw=0.9, zorder=3))
ax.text(bx+0.525, by-0.22, "Edificio\nFamilia B", ha="center", va="top",
        fontsize=8.5, color=INK)
arrow(ax, (bx+1.05, 8.9), (2.30, 8.75))
ax.text((bx+1.05+2.30)/2, 9.02, "extraccion\nde features", ha="center", va="bottom",
        fontsize=6.8, color=INK, style="italic")

# vector de features agrupado
fvx, fvy, cellw = 2.35, 8.55, 0.30
groups = [("zona\n(3)", 3, "#D1C4E9"),
          ("suelo\n(4)", 4, "#C5CAE9"),
          ("geometria + material\n(14)", 14, "#B3E5FC")]
x = fvx
for lbl, n, col in groups:
    for k in range(n):
        ax.add_patch(Rectangle((x, fvy), cellw, 0.42, fc=col, ec="white", lw=0.6, zorder=3))
        x += cellw
    ax.text(x-n*cellw/2, fvy-0.14, lbl, ha="center", va="top", fontsize=7.6, color=INK)
    x += 0.05
ax.text(fvx + (x-fvx)/2, fvy+0.72, r"Vector de entrada  $\mathbf{x}\in\mathbb{R}^{21}$  (one-hot zona/suelo)",
        ha="center", va="bottom", fontsize=10, weight="bold", color=INK)

# ================= ENCODERS =================
# encoder T1
ex1, ey1, ew1 = 1.1, 5.55, 2.6
box(ax, ex1, ey1, ew1, 1.55, "", C_ENC, C_ENC_E, lw=1.5)
ax.text(ex1+ew1/2, ey1+1.35, "Encoder  $T_1$", ha="center", fontsize=10.5,
        weight="bold", color=C_ENC_E)
box(ax, ex1+0.2, ey1+0.72, ew1-0.4, 0.42, "Linear 21$\\to$128 + LN + SiLU", "#FFFFFF", C_ENC_E, fs=8.3)
box(ax, ex1+0.2, ey1+0.18, ew1-0.4, 0.42, "ResBlock $\\times$2  (128)", "#FFFFFF", C_ENC_E, fs=8.6, weight="bold")

# encoder compartido
ex2, ey2, ew2 = 4.6, 5.55, 9.2
box(ax, ex2, ey2, ew2, 1.55, "", C_ENC, C_ENC_E, lw=1.5)
ax.text(ex2+ew2/2, ey2+1.35, "Encoder compartido", ha="center", fontsize=10.5,
        weight="bold", color=C_ENC_E)
box(ax, ex2+0.3, ey2+0.72, ew2-0.6, 0.42, "Linear 21$\\to$256 + LayerNorm + SiLU", "#FFFFFF", C_ENC_E, fs=8.8)
box(ax, ex2+0.3, ey2+0.18, ew2-0.6, 0.42, "ResBlock $\\times$4   (256)", "#FFFFFF", C_ENC_E, fs=9, weight="bold")

# ================= HEADS =================
hy, hh = 3.7, 1.0
heads = [
    (1.1,  2.6, "Head $T_1$\n128$\\to$32$\\to$1"),
    (4.6,  1.75,"Head $T_2..T_{18}$\n256$\\to$128$\\to$17"),
    (6.65, 1.75,"Head $\\Phi$\n256$\\to$512$\\to$972"),
    (8.9,  2.2, "Head respuestas\n256$\\to$256$\\to$72"),
    (11.7, 2.1, "Head $V_b$\n256$\\to$64$\\to$2"),
]
hcx = []
for hx, hw, txt in heads:
    box(ax, hx, hy, hw, hh, txt, C_HEAD, C_HEAD_E, fs=8.8)
    hcx.append(hx+hw/2)

# ================= OUTPUTS =================
oy, oh = 1.85, 0.95
outs = [
    (1.1,  2.6, "$T_1$\nperiodo fundamental", C_OUT),
    (4.6,  1.75,"$T_2\\,..\\,T_{18}$\nperiodos superiores", C_OUT),
    (6.65, 1.75,"$\\Phi_x,\\Phi_y,\\Phi_\\theta$\nformas modales", C_OUT),
    (8.9,  2.2, "$U_x,U_y$, deriva$_x$,\nderiva$_y$  por piso", C_OUT),
    (11.7, 2.1, "$V_{b,x},\\,V_{b,y}$\ncortante basal", C_OUT),
]
ocx = []
for ox, ow, txt, col in outs:
    box(ax, ox, oy, ow, oh, txt, col, C_OUT_E, fs=8.6)
    ocx.append(ox+ow/2)

# ================= FLECHAS =================
# input -> encoders
arrow(ax, (fvx+0.3, 8.5), (ex1+ew1/2, ey1+1.55))
arrow(ax, (fvx+2.2, 8.5), (ex2+2.0, ey2+1.55))
dimtag(ax, ex1+ew1/2-0.9, 7.35, "21", C_ENC_E)
dimtag(ax, ex2+1.4, 7.35, "21", C_ENC_E)

# encoder T1 -> head_T1
arrow(ax, (hcx[0], ey1), (hcx[0], hy+hh))
dimtag(ax, hcx[0]+0.55, ey1-0.28, "128", C_ENC_E)
# shared -> 4 heads (bus)
busy = ey2-0.35
ax.plot([hcx[1], hcx[4]], [busy, busy], color=C_ENC_E, lw=1.4, zorder=1)
arrow(ax, (ex2+ew2/2, ey2), (ex2+ew2/2, busy))
for cx in hcx[1:]:
    arrow(ax, (cx, busy), (cx, hy+hh))
dimtag(ax, ex2+ew2/2+0.5, busy+0.02, "256", C_ENC_E)

# heads -> outputs
for cx in ocx:
    arrow(ax, (cx, hy), (cx, oy+oh))
# dims salida
for cx, d in zip(ocx, ["1","17","18x18x3","18x4","2"]):
    dimtag(ax, cx+0.0, hy-0.32, d, C_HEAD_E)

# ================= RESTRICCIONES FISICAS (PINN) =================
py = 0.30
box(ax, 0.5, py, 13.5, 1.15, "", C_PHYS, C_PHYS_E, lw=1.4)
ax.text(0.85, py+0.92, "Restricciones fisicas embebidas (PINN)",
        ha="left", va="center", fontsize=10, weight="bold", color=C_PHYS_E)
constraints = [
    r"$\bullet$  Periodos ordenados:  $T_n = T_{n-1} - \mathrm{softplus}(\delta_n)\;\Rightarrow\;T_1>T_2>\cdots>T_{18}$",
    r"$\bullet$  Acople de deriva:  deriva$_x = \mathrm{ratio}\cdot$deriva$_y$  con ratio$\,\geq 0$",
    r"$\bullet$  Mascara segun n$^\circ$ de pisos    $\bullet$  Salidas $V_b$ y deriva en espacio logaritmico",
]
for i, c in enumerate(constraints):
    ax.text(0.95, py+0.60 - i*0.27, c, ha="left", va="center", fontsize=8.3, color=INK)

# ================= INSET: detalle ResBlock =================
ix, iy, iw, ih = 9.35, 7.95, 5.0, 1.78
ax.add_patch(FancyBboxPatch((ix, iy), iw, ih,
    boxstyle="round,pad=0.02,rounding_size=0.05",
    linewidth=1.2, edgecolor=C_ENC_E, facecolor="#F5FAFF",
    linestyle=(0, (4, 2)), zorder=2))
ax.text(ix+0.18, iy+ih-0.18, "Detalle del ResBlock", ha="left", va="center",
        fontsize=8.6, weight="bold", color=C_ENC_E)
# cadena horizontal
cy = iy+0.32
elems = ["Linear", "LN", "SiLU", "Linear", "LN"]
bw, gap, bh = 0.5, 0.08, 0.42
cx0 = ix+0.62
xs = []
for j, e in enumerate(elems):
    xx = cx0 + j*(bw+gap)
    box(ax, xx, cy, bw, bh, e, "#FFFFFF", C_ENC_E, fs=6.2)
    xs.append(xx)
midy = cy+bh/2
xin = ix+0.32
xsum = xs[-1]+bw+0.30
xsilu = xsum+0.42
ax.add_patch(plt.Circle((xsum, midy), 0.14, fc="#FFFFFF", ec=C_ENC_E, lw=1.2, zorder=3))
ax.text(xsum, midy, "+", ha="center", va="center", fontsize=10, color=C_ENC_E, zorder=4)
box(ax, xsilu, cy, 0.5, bh, "SiLU", "#FFFFFF", C_ENC_E, fs=6.2)
# flechas cadena
arrow(ax, (xin, midy), (xs[0], midy), color=C_ENC_E, lw=1.0)
for j in range(len(xs)-1):
    arrow(ax, (xs[j]+bw, midy), (xs[j+1], midy), color=C_ENC_E, lw=1.0)
arrow(ax, (xs[-1]+bw, midy), (xsum-0.14, midy), color=C_ENC_E, lw=1.0)
arrow(ax, (xsum+0.14, midy), (xsilu, midy), color=C_ENC_E, lw=1.0)
arrow(ax, (xsilu+0.5, midy), (ix+iw-0.14, midy), color=C_ENC_E, lw=1.0)
ax.text(xin, midy-0.28, "x", ha="center", va="top", fontsize=7.5, style="italic", color=INK)
ax.text(ix+iw-0.14, midy-0.28, "out", ha="right", va="top", fontsize=6.6, style="italic", color=INK)
# conexion residual (arco por arriba, contenido dentro del inset)
ax.add_patch(FancyArrowPatch((xin, midy+0.10), (xsum, midy+0.14),
    connectionstyle="arc3,rad=-0.24", arrowstyle="-|>", mutation_scale=9,
    linewidth=1.1, color=C_PHYS_E, ls="--", zorder=3, shrinkA=1, shrinkB=3))
ax.text((xin+xsum)/2, midy+0.62, "conexion residual", ha="center", va="center",
        fontsize=6.6, style="italic", color=C_PHYS_E)

plt.tight_layout()
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("arch_v2.png")
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print("Guardado:", OUT)
