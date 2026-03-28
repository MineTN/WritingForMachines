"""
Claim A v2 — Tokenizer-Friendly Symbol Experiment
"Writing for Machines" makalesinin yeniden çerçevelenmiş Claim A testi.

Merkezi soru:
    Matematiksel semboller değil, tokenizer-dostu semboller tercih edilirse
    token ekonomisi sağlanır mı?

Kurulum:
    pip install tiktoken wikipedia-api tqdm pandas matplotlib

Çalıştırma:
    python claim_a_v2.py
"""

import re
import json
import random
import statistics
from collections import defaultdict

import tiktoken
import wikipediaapi
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use("Agg")

# ─── Tokenizer'lar ─────────────────────────────────────────────────────────
ENCODERS = {
    "GPT-2":  tiktoken.get_encoding("gpt2"),
    "GPT-4o": tiktoken.get_encoding("cl100k_base"),
}

def token_ids(text, enc):
    return enc.encode(text)

def n_tokens(text, enc):
    return len(enc.encode(text))

# ─── Sembol Katmanları ─────────────────────────────────────────────────────
# Her sembol için: (verbose_phrase, compact_symbol, tier, tier_label)
PAIRS = [
    # ── Tier 1 — ASCII (base vocab garantili) ─────────────────────────────
    ("and",                     "&",   1, "ASCII"),
    ("at",                      "@",   1, "ASCII"),
    ("number",                  "#",   1, "ASCII"),
    ("percent",                 "%",   1, "ASCII"),
    ("divided by",              "/",   1, "ASCII"),
    ("times",                   "*",   1, "ASCII"),
    ("equals",                  "=",   1, "ASCII"),
    ("less than",               "<",   1, "ASCII"),
    ("greater than",            ">",   1, "ASCII"),
    ("plus or minus",           "+/-", 1, "ASCII"),

    # ── Tier 2 — Latin-1 (genellikle vocabulary'de) ───────────────────────
    ("degrees",                 "°",   2, "Latin-1"),
    ("plus or minus",           "±",   2, "Latin-1"),
    ("multiplied by",           "×",   2, "Latin-1"),
    ("divided by",              "÷",   2, "Latin-1"),
    ("one half",                "½",   2, "Latin-1"),
    ("one quarter",             "¼",   2, "Latin-1"),
    ("three quarters",          "¾",   2, "Latin-1"),
    ("micro",                   "µ",   2, "Latin-1"),
    ("squared",                 "²",   2, "Latin-1"),
    ("cubed",                   "³",   2, "Latin-1"),

    # ── Tier 3 — Math Unicode (değişken — karşılaştırma grubu) ─────────────
    ("less than or equal to",   "≤",   3, "Math Unicode"),
    ("greater than or equal to","≥",   3, "Math Unicode"),
    ("not equal to",            "≠",   3, "Math Unicode"),
    ("approximately",           "≈",   3, "Math Unicode"),
    ("therefore",               "∴",   3, "Math Unicode"),
    ("because",                 "∵",   3, "Math Unicode"),
    ("implies",                 "→",   3, "Math Unicode"),
    ("if and only if",          "↔",   3, "Math Unicode"),
    ("element of",              "∈",   3, "Math Unicode"),
    ("subset of",               "⊂",   3, "Math Unicode"),
    ("infinity",                "∞",   3, "Math Unicode"),
    ("proportional to",         "∝",   3, "Math Unicode"),
    ("sum of",                  "∑",   3, "Math Unicode"),
    ("square root of",          "√",   3, "Math Unicode"),
    ("integral of",             "∫",   3, "Math Unicode"),

    # ── Tier 4 — Emoji (her zaman byte-level) ─────────────────────────────
    ("happy",                   "😊",  4, "Emoji"),
    ("correct",                 "✅",  4, "Emoji"),
    ("warning",                 "⚠️",  4, "Emoji"),
    ("fire",                    "🔥",  4, "Emoji"),
    ("star",                    "⭐",  4, "Emoji"),
]

# ─── Faz 1: Vocabulary Analizi ─────────────────────────────────────────────

def phase1_vocabulary_audit():
    """Her sembolün her tokenizer'daki token sayısını ve ID'lerini çıkarır."""
    print("\n" + "="*60)
    print("FAZ 1: Tokenizer Vocabulary Analizi")
    print("="*60)

    results = []
    for phrase, symbol, tier, tier_label in PAIRS:
        row = {"phrase": phrase, "symbol": symbol, "tier": tier, "tier_label": tier_label}
        for enc_name, enc in ENCODERS.items():
            p_tokens = n_tokens(phrase, enc)
            s_tokens = n_tokens(symbol, enc)
            s_ids    = token_ids(symbol, enc)
            delta    = p_tokens - s_tokens
            row[f"{enc_name}_phrase_tokens"] = p_tokens
            row[f"{enc_name}_symbol_tokens"] = s_tokens
            row[f"{enc_name}_delta"]         = delta
            row[f"{enc_name}_symbol_ids"]    = s_ids
        # Cross-tokenizer safe: her iki encoder'da da 1 token
        cross_safe = all(row[f"{e}_symbol_tokens"] == 1 for e in ENCODERS)
        # Net faydalı: her iki encoder'da da delta > 0 (sembol phrase'den ucuz)
        net_beneficial = all(row[f"{e}_delta"] > 0 for e in ENCODERS)
        row["cross_tokenizer_safe"] = cross_safe
        row["net_beneficial"]       = net_beneficial
        results.append(row)

    df = pd.DataFrame(results)

    print(f"\n{'Phrase':<30} {'Symbol':<6} {'Tier':<14} {'GPT-2 D':>8} {'GPT-4o D':>9} {'Safe':>6} {'Beneficial':>11}")
    print("-" * 90)
    for _, r in df.iterrows():
        safe_str  = "✓" if r["cross_tokenizer_safe"] else "✗"
        bene_str  = "✓" if r["net_beneficial"]       else "✗"
        d2  = r["GPT-2_delta"]
        d4o = r["GPT-4o_delta"]
        d2_str  = f"+{d2}"  if d2  > 0 else str(d2)
        d4o_str = f"+{d4o}" if d4o > 0 else str(d4o)
        print(f"  {r['phrase']:<28} {r['symbol']:<6} {r['tier_label']:<14} {d2_str:>8} {d4o_str:>9} {safe_str:>6} {bene_str:>11}")

    # Özet
    print("\nTier özeti:")
    for tier_id in [1, 2, 3, 4]:
        sub = df[df["tier"] == tier_id]
        safe  = sub["cross_tokenizer_safe"].sum()
        bene  = sub["net_beneficial"].sum()
        total = len(sub)
        print(f"  Tier {tier_id} ({sub.iloc[0]['tier_label']:12}): {total} çift | Cross-safe: {safe}/{total} | Beneficial: {bene}/{total}")

    # Cross-tokenizer safe + beneficial olan listesi
    safe_beneficial = df[df["cross_tokenizer_safe"] & df["net_beneficial"]]
    print(f"\nCross-tokenizer safe AND beneficial: {len(safe_beneficial)} çift")
    for _, r in safe_beneficial.iterrows():
        print(f"  '{r['phrase']}' → '{r['symbol']}'  (Tier {r['tier']})")

    return df

# ─── Wikipedia Corpus ──────────────────────────────────────────────────────

WIKI_TOPICS = [
    "Artificial intelligence", "Climate change", "Quantum mechanics",
    "World War II", "Human genome", "Solar system", "French Revolution",
    "Protein folding", "Internet", "Democracy", "Photosynthesis",
    "Black hole", "Evolutionary biology", "Cryptocurrency", "Vaccine",
    "Philosophy of mind", "Roman Empire", "Superconductivity",
    "Cognitive science", "Game theory", "Plate tectonics",
    "Machine learning", "Ancient Greece", "Thermodynamics",
    "Linguistics", "Capitalism", "Relativity", "Neuroscience",
    "Computer science", "Ecology",
]

def fetch_corpus(n=30, words_each=1000):
    wiki = wikipediaapi.Wikipedia(language="en", user_agent="TokenExperiment/2.0")
    samples = []
    random.shuffle(WIKI_TOPICS)
    print(f"\n[Corpus] {n} Wikipedia parçası çekiliyor...")
    for topic in tqdm(WIKI_TOPICS):
        if len(samples) >= n:
            break
        try:
            page = wiki.page(topic)
            if not page.exists():
                continue
            text = re.sub(r"==+[^=]+=+\n?", "", page.text)
            text = re.sub(r"\n{2,}", "\n", text).strip()
            words = text.split()
            if len(words) < words_each:
                continue
            start = random.randint(0, len(words) - words_each)
            samples.append(" ".join(words[start:start + words_each]))
        except Exception:
            continue
    print(f"  {len(samples)} parça toplandı ({sum(len(s.split()) for s in samples):,} kelime)")
    return samples

# ─── Faz 2: Frekans Analizi ────────────────────────────────────────────────

def phase2_frequency(audit_df, corpus):
    """Her phrase'in corpus'taki frekansını ölçer."""
    print("\n" + "="*60)
    print("FAZ 2: Corpus Frekans Analizi")
    print("="*60)

    full_text = " ".join(corpus).lower()
    total_words = len(full_text.split())

    freq_rows = []
    for _, row in audit_df.iterrows():
        phrase_lower = row["phrase"].lower()
        count = full_text.count(phrase_lower)
        per_million = count / total_words * 1_000_000
        freq_rows.append({
            "phrase": row["phrase"],
            "symbol": row["symbol"],
            "tier": row["tier"],
            "tier_label": row["tier_label"],
            "cross_tokenizer_safe": row["cross_tokenizer_safe"],
            "net_beneficial": row["net_beneficial"],
            "count": count,
            "per_million": round(per_million, 1),
        })

    freq_df = pd.DataFrame(freq_rows).sort_values("count", ascending=False)

    print(f"\n  Corpus: {total_words:,} kelime\n")
    print(f"  {'Phrase':<30} {'Symbol':<6} {'Tier':<14} {'Count':>7} {'Per Mw':>8} {'Safe':>6}")
    print("  " + "-"*75)
    for _, r in freq_df.iterrows():
        safe = "✓" if r["cross_tokenizer_safe"] else " "
        bene = "✓" if r["net_beneficial"]       else " "
        print(f"  {r['phrase']:<30} {r['symbol']:<6} {r['tier_label']:<14} {r['count']:>7} {r['per_million']:>8.1f}   {safe}")

    return freq_df

# ─── Faz 3: Corpus-Level Uygulama ─────────────────────────────────────────

def apply_substitutions(text, pairs_subset):
    """Verilen çiftleri metne uygular."""
    result = text
    for phrase, symbol in pairs_subset:
        pattern = re.compile(re.escape(phrase), re.IGNORECASE)
        result = pattern.sub(symbol, result)
    return result

def phase3_corpus_application(audit_df, freq_df, corpus):
    """
    4 strateji karşılaştırır:
    A. Tüm çiftler (karma)
    B. Sadece cross-tokenizer-safe + beneficial
    C. Sadece Tier 1 (ASCII)
    D. Sadece Tier 1+2 (ASCII + Latin-1)
    """
    print("\n" + "="*60)
    print("FAZ 3: Corpus-Level Uygulama — 4 Strateji")
    print("="*60)

    # Hangi çiftleri kullanacağız
    all_pairs      = [(r["phrase"], r["symbol"]) for _, r in audit_df.iterrows()]
    safe_bene      = [(r["phrase"], r["symbol"]) for _, r in audit_df[
                        audit_df["cross_tokenizer_safe"] & audit_df["net_beneficial"]].iterrows()]
    tier1_only     = [(r["phrase"], r["symbol"]) for _, r in audit_df[audit_df["tier"]==1].iterrows()]
    tier12_only    = [(r["phrase"], r["symbol"]) for _, r in audit_df[audit_df["tier"].isin([1,2])].iterrows()]

    strategies = {
        "A. All pairs\n(mixed tiers)":        all_pairs,
        "B. Safe+Beneficial\n(cross-tokenizer)": safe_bene,
        "C. Tier 1 only\n(ASCII)":             tier1_only,
        "D. Tier 1+2\n(ASCII + Latin-1)":      tier12_only,
    }

    results = {}
    for strat_name, pairs in strategies.items():
        enc_results = {}
        for enc_name, enc in ENCODERS.items():
            before_tokens = 0
            after_tokens  = 0
            for sample in corpus:
                modified = apply_substitutions(sample, pairs)
                before_tokens += n_tokens(sample,   enc)
                after_tokens  += n_tokens(modified, enc)
            pct_change = (after_tokens - before_tokens) / before_tokens * 100
            enc_results[enc_name] = {
                "before": before_tokens,
                "after":  after_tokens,
                "delta":  after_tokens - before_tokens,
                "pct":    round(pct_change, 3),
            }
        results[strat_name] = enc_results
        n_pairs = len(pairs)
        print(f"\n  {strat_name.replace(chr(10),' ')} ({n_pairs} çift)")
        for enc_name, r in enc_results.items():
            sign = "+" if r["pct"] >= 0 else ""
            print(f"    {enc_name}: {r['before']:,} → {r['after']:,} token  ({sign}{r['pct']}%)")

    # Teknik yoğun metin testi (Tier 1+2 ile)
    print("\n  [Bonus] Matematiksel açıdan yoğun paragraf (Tier 1+2):")
    dense_text = (
        "The value of x times y divided by z equals one half when x equals one quarter and y squared "
        "plus z cubed is greater than one hundred percent of the total. "
        "The temperature is twenty degrees above zero, with a tolerance of plus or minus five degrees. "
        "The ratio is three quarters, approximately equal to zero point seven five. "
        "When the variable is less than the threshold divided by two, the system is in the micro regime. "
        "The frequency is one hundred mega hertz, and the power output is divided by four in each stage. "
        "One half of the samples show a value greater than the mean times two, "
        "while three quarters remain within one standard deviation squared. "
        "The error is plus or minus three percent, and the signal is multiplied by the gain factor. "
        "Each component contributes one quarter of the total energy, divided by the number of stages. "
        "At ninety degrees rotation, the phase shift is one half of the full cycle. "
        "The squared error is less than one hundredth of the original value squared, "
        "and the cubed distortion is divided by the sampling rate times the bandwidth. "
        "Results show three quarters agreement, with micro variations at the sub-percent level. "
        "The gain is multiplied by a factor of two squared, and losses are one quarter of input. "
        "Temperature control is maintained at twenty-two degrees, plus or minus two degrees."
    )
    for enc_name, enc in ENCODERS.items():
        before = n_tokens(dense_text, enc)
        after  = n_tokens(apply_substitutions(dense_text, tier12_only), enc)
        pct    = (after - before) / before * 100
        sign = "+" if pct >= 0 else ""
        print(f"    {enc_name}: {before} → {after} token  ({sign}{pct:.2f}%)")

    return results

# ─── Grafik ────────────────────────────────────────────────────────────────

TIER_COLORS = {1: "#2196F3", 2: "#4CAF50", 3: "#FF9800", 4: "#F44336"}
TIER_LABELS = {1: "Tier 1: ASCII", 2: "Tier 2: Latin-1",
               3: "Tier 3: Math Unicode", 4: "Tier 4: Emoji"}

def plot_results(audit_df, phase3_results):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        '"Writing for Machines" — Tokenizer-Friendly Symbol Experiment (Claim A v2)',
        fontsize=13, fontweight="bold"
    )

    # ── Panel 1: Token Delta per symbol pair (GPT-4o) ──
    ax = axes[0]
    ax.set_title("Symbol Token Δ vs Verbose Phrase\n(GPT-4o, positive = savings)", fontsize=10)
    df_sorted = audit_df.sort_values(["tier", "GPT-4o_delta"], ascending=[True, False])
    bar_colors = [TIER_COLORS[t] for t in df_sorted["tier"]]
    bars = ax.barh(range(len(df_sorted)), df_sorted["GPT-4o_delta"],
                   color=bar_colors, edgecolor="white", alpha=0.85)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{r['symbol']} ← {r['phrase'][:22]}"
                        for _, r in df_sorted.iterrows()], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Token savings (positive = symbol cheaper)")
    legend_patches = [mpatches.Patch(color=TIER_COLORS[t], label=TIER_LABELS[t]) for t in [1,2,3,4]]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right")

    # ── Panel 2: Cross-tokenizer safety heatmap ────────
    ax = axes[1]
    ax.set_title("Symbol Token Cost\nGPT-2 vs GPT-4o", fontsize=10)
    df_s = audit_df.copy()
    scatter_colors = [TIER_COLORS[t] for t in df_s["tier"]]
    ax.scatter(df_s["GPT-2_symbol_tokens"], df_s["GPT-4o_symbol_tokens"],
               c=scatter_colors, s=100, alpha=0.8, edgecolors="white", linewidths=0.5)
    for _, r in df_s.iterrows():
        ax.annotate(r["symbol"], (r["GPT-2_symbol_tokens"], r["GPT-4o_symbol_tokens"]),
                    fontsize=7, ha="center", va="bottom")
    ax.plot([0, 6], [0, 6], "k--", alpha=0.3, label="GPT-2 = GPT-4o")
    ax.axvline(1.5, color="gray", linestyle=":", alpha=0.5)
    ax.axhline(1.5, color="gray", linestyle=":", alpha=0.5)
    ax.fill_between([0, 1.5], [0, 0], [1.5, 1.5], alpha=0.07, color="green")
    ax.text(0.75, 0.75, "SAFE\nZONE\n(1 token\nboth)", ha="center", va="center",
            fontsize=7, color="green", alpha=0.8)
    ax.set_xlabel("GPT-2 tokens")
    ax.set_ylabel("GPT-4o tokens")
    ax.legend(handles=legend_patches, fontsize=8)
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 6.5)

    # ── Panel 3: Corpus savings by strategy ────────────
    ax = axes[2]
    ax.set_title("Corpus-Level Token Change\nby Substitution Strategy", fontsize=10)
    strat_labels = list(phase3_results.keys())
    x = range(len(strat_labels))
    width = 0.35
    gpt2_pcts  = [phase3_results[s]["GPT-2"]["pct"]  for s in strat_labels]
    gpt4o_pcts = [phase3_results[s]["GPT-4o"]["pct"] for s in strat_labels]
    b1 = ax.bar([i - width/2 for i in x], gpt2_pcts,  width, label="GPT-2",  color="#1565C0", alpha=0.8)
    b2 = ax.bar([i + width/2 for i in x], gpt4o_pcts, width, label="GPT-4o", color="#2E7D32", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("\n", "\n") for s in strat_labels], fontsize=8)
    ax.set_ylabel("Token change (%)")
    ax.legend(fontsize=9)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        sign = "+" if h >= 0 else ""
        ax.text(bar.get_x() + bar.get_width()/2,
                h + (0.01 if h >= 0 else -0.02),
                f"{sign}{h:.2f}%", ha="center", va="bottom" if h >= 0 else "top",
                fontsize=7)

    plt.tight_layout()
    out = "results_claim_a_v2.png"
    plt.savefig(out, dpi=150)
    print(f"\nGrafik: {out}")

# ─── Ana Akış ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    audit_df = phase1_vocabulary_audit()

    corpus = fetch_corpus(n=30, words_each=1000)
    if len(corpus) < 10:
        print("HATA: Yeterli corpus çekilemedi.")
        exit(1)

    freq_df = phase2_frequency(audit_df, corpus)

    phase3_results = phase3_corpus_application(audit_df, freq_df, corpus)

    plot_results(audit_df, phase3_results)

    # JSON çıktı
    out = {
        "audit": audit_df[["phrase","symbol","tier","tier_label",
                            "GPT-2_delta","GPT-4o_delta",
                            "cross_tokenizer_safe","net_beneficial"]].to_dict("records"),
        "phase3": {
            k: {enc: {kk: v for kk, v in vv.items() if kk != "delta"}
                for enc, vv in strats.items()}
            for k, strats in phase3_results.items()
        }
    }
    with open("results_claim_a_v2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Veri: results_claim_a_v2.json")
    print("\nTamamlandı.")
