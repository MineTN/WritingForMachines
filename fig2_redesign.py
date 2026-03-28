"""
Figure 2 yeniden tasarımı — okunabilir 2 panel
Veri kaynağı: Table 2 ve Section 3.4 body text
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─── Veri (Table 2'den) ─────────────────────────────────────────────────────
tiers  = ["Tier 1\nASCII", "Tier 2\nLatin-1", "Tier 3\nMath\nUnicode", "Tier 4\nEmoji"]
n      = [10, 10, 15, 5]
safe   = [9,  5,  1,  0]   # cross-tokenizer safe
benef  = [4,  6,  7,  0]   # individually beneficial (word-level)
colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]

safe_pct  = [s/n_*100 for s, n_ in zip(safe,  n)]
benef_pct = [b/n_*100 for b, n_ in zip(benef, n)]

# ─── Corpus-level strategy outcomes (actual results_claim_a_v2.json) ─────────
strategies  = ["Strategy A\n(all 40 pairs)", "Strategy B\n(safe + beneficial)", "Strategy C\n(Tier 1 ASCII only)"]
gpt2_pct    = [+10.148,  -0.018,  +9.808]
gpt4o_pct   = [+9.619,   -0.018,  +9.355]

# ─── Figure ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle(
    'Figure 2 — Tokenizer-Friendly Symbol Experiment (Claim A)',
    fontsize=13, fontweight='bold', y=1.01
)

# Panel 1: Cross-safe vs Beneficial rates by tier
ax = axes[0]
ax.set_title("Panel A: Cross-tokenizer Safety vs. Word-level Benefit\nby Symbol Tier (n = 40 pairs total)", fontsize=11)

x     = np.arange(len(tiers))
width = 0.35
bars1 = ax.bar(x - width/2, safe_pct,  width, label="Cross-tokenizer safe (%)", color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
bars2 = ax.bar(x + width/2, benef_pct, width, label="Individually beneficial (%)", color=colors, alpha=0.45, edgecolor=colors, linewidth=1.5, hatch="//")

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.0f}%",
            ha='center', va='bottom', fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(tiers, fontsize=11)
ax.set_ylabel("Percentage of pairs in tier (%)", fontsize=11)
ax.set_ylim(0, 105)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.axhline(50, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)

# Custom legend
solid_patch  = mpatches.Patch(facecolor='#888888', alpha=0.85, label='Cross-tokenizer safe (%)')
hatch_patch  = mpatches.Patch(facecolor='#888888', alpha=0.45, hatch='//', edgecolor='#888888', label='Individually beneficial (%)')
ax.legend(handles=[solid_patch, hatch_patch], fontsize=10, loc='upper right')

# n annotation
for i, (tier, ni) in enumerate(zip(tiers, n)):
    ax.text(i, -8, f"n={ni}", ha='center', va='top', fontsize=9, color='gray')

ax.set_xlim(-0.6, 3.6)

# Panel 2: Corpus-level token change by strategy
ax = axes[1]
ax.set_title("Panel B: Corpus-level Token Change by\nSubstitution Strategy (Wikipedia corpus)", fontsize=11)

x2    = np.arange(len(strategies))
w2    = 0.3
b1 = ax.bar(x2 - w2/2, gpt2_pct,  w2, label="GPT-2",  color="#1565C0", alpha=0.85, edgecolor="white")
b2 = ax.bar(x2 + w2/2, gpt4o_pct, w2, label="GPT-4o", color="#2E7D32", alpha=0.85, edgecolor="white")

ax.axhline(0, color='black', linewidth=1.0)
ax.set_xticks(x2)
ax.set_xticklabels(strategies, fontsize=10)
ax.set_ylabel("Token count change (%)", fontsize=11)
ax.legend(fontsize=11)

# Value labels
for bar in list(b1) + list(b2):
    h = bar.get_height()
    sign = "+" if h >= 0 else ""
    va   = 'bottom' if h >= 0 else 'top'
    ypos = h + (0.15 if h >= 0 else -0.15)
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{sign}{h:.1f}%", ha='center', va=va, fontsize=10, fontweight='bold')

# Colour zones
ax.axhspan(0, 15, alpha=0.04, color='red')
ax.axhspan(-10, 0, alpha=0.04, color='green')
ax.text(2.65, 9, "↑ Higher\ncost", ha='center', fontsize=9, color='#C62828', alpha=0.7)
ax.text(1, -3.5, "↓ Lower\ncost", ha='center', fontsize=9, color='#1B5E20', alpha=0.7)
ax.set_ylim(-5, 13)
ax.set_xlim(-0.6, 2.9)

# BPE key insight: Strategy C (Tier 1 ASCII only) still causes +9.8%
ax.annotate(
    'BPE context effect:\nASCII-only still +9.8%',
    xy=(2, 9.808), xytext=(1.1, 12),
    fontsize=9, color='#C62828',
    arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.2),
    ha='center'
)
# Strategy B near-zero annotation
ax.annotate(
    '−0.02%\n(near zero)',
    xy=(1, -0.018), xytext=(1, -3.5),
    fontsize=9, color='#1B5E20',
    arrowprops=dict(arrowstyle='->', color='#1B5E20', lw=1.0),
    ha='center'
)

plt.tight_layout()
out = "C:/natal/token_experiment/fig2_v2.png"
plt.savefig(out, dpi=180, bbox_inches='tight')
print(f"Kaydedildi: {out}")
