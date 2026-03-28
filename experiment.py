"""
Token Ratio Experiment
"Writing for Machines" makalesindeki 2.1x iddiasını test eder.

Kurulum:
    pip install tiktoken wikipedia-api datasets tqdm pandas matplotlib

Çalıştırma:
    python experiment.py
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
import matplotlib
matplotlib.use("Agg")  # başsız sunucu uyumlu

# ─── Ayarlar ──────────────────────────────────────────────────────────────────

SAMPLE_COUNT   = 30      # her koşul için kaç metin parçası
WORDS_PER_SAMPLE = 1000  # her parça yaklaşık kaç kelime

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

# ─── Tokenizer'lar ──────────────────────────────────────────────────────────

TOKENIZERS = {
    "GPT-2":   tiktoken.get_encoding("gpt2"),
    "GPT-4o":  tiktoken.get_encoding("cl100k_base"),
}

def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))

def count_words(text: str) -> int:
    return len(text.split())

# ─── Gürültü Ekleme Fonksiyonları ───────────────────────────────────────────

def add_symbols(text: str) -> str:
    """Sembolik ifadeler ekle (makale önerisi — iyi yön)"""
    replacements = {
        "greater than or equal to": "≥",
        "less than or equal to": "≤",
        "not equal to": "≠",
        "approximately": "≈",
        "one half": "½",
        "one quarter": "¼",
        "therefore": "∴",
        "implies": "→",
        "and": "&",
        "section": "§",
        "degree": "°",
    }
    for phrase, sym in replacements.items():
        text = text.replace(phrase, sym)
    return text

def add_emojis(text: str) -> str:
    """Emojiler ekle (byte-level fallback tetikleyen karakterler)"""
    emojis = ["🙂", "✅", "🔥", "⭐", "💡", "🚀", "❌", "📊", "🎯", "🌍"]
    sentences = text.split(". ")
    for i in range(0, len(sentences), 5):
        sentences[i] = random.choice(emojis) + " " + sentences[i]
    return ". ".join(sentences)

def add_html_artifacts(text: str) -> str:
    """Ham web metnindeki HTML artıklarını simüle et"""
    artifacts = [
        "<br/>", "&amp;", "&nbsp;", "&#x200B;", "<span>", "</span>",
        "&lt;", "&gt;", "<!-- comment -->", "\u00a0", "\u200c",
    ]
    words = text.split()
    for i in range(0, len(words), 3):
        words[i] = random.choice(artifacts) + " " + words[i]
    return " ".join(words)

def add_mixed_unicode(text: str) -> str:
    """Nadir Unicode karakterler ekle (dead-weight token adayları)"""
    rare = ["ﬁ", "ﬂ", "‼", "⁉", "℃", "№", "℅", "⅓", "⅔", "‰", "‱"]
    words = text.split()
    for i in range(0, len(words), 15):
        words[i] = words[i] + random.choice(rare)
    return " ".join(words)

NOISE_PIPELINE = {
    "clean":         lambda t: t,
    "+symbols":      add_symbols,
    "+emojis":       add_emojis,
    "+html":         add_html_artifacts,
    "+unicode":      add_mixed_unicode,
    "full_noisy":    lambda t: add_mixed_unicode(add_html_artifacts(add_emojis(t))),
}

# ─── Veri Toplama ──────────────────────────────────────────────────────────

def fetch_wikipedia_samples(n: int, words_each: int) -> list[str]:
    """Wikipedia'dan n adet ~words_each kelimelik metin parçası çeker."""
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="TokenExperiment/1.0 (research)"
    )
    samples = []
    topics = WIKI_TOPICS.copy()
    random.shuffle(topics)

    print(f"\n[1/3] Wikipedia'dan {n} metin parçası çekiliyor...")
    for topic in tqdm(topics, desc="Wikipedia"):
        if len(samples) >= n:
            break
        try:
            page = wiki.page(topic)
            if not page.exists():
                continue
            text = page.text
            # Başlık satırlarını çıkar, düz paragraf al
            text = re.sub(r"==+[^=]+=+\n?", "", text)
            text = re.sub(r"\n{2,}", "\n", text).strip()
            words = text.split()
            if len(words) < words_each:
                continue
            # Rastgele bir pencere al
            start = random.randint(0, max(0, len(words) - words_each))
            chunk = " ".join(words[start:start + words_each])
            samples.append(chunk)
        except Exception:
            continue

    print(f"   {len(samples)} temiz metin parçası toplandı.")
    return samples[:n]

# ─── Analiz ────────────────────────────────────────────────────────────────

def analyze(samples: list[str]) -> dict:
    """Her tokenizer ve gürültü koşulu için token/kelime oranını hesaplar."""
    results = defaultdict(list)  # key: (tokenizer, condition), value: list of ratios

    print("\n[2/3] Token sayımı yapılıyor...")
    for sample in tqdm(samples, desc="Metinler"):
        base_words = count_words(sample)
        for cond_name, noise_fn in NOISE_PIPELINE.items():
            noisy = noise_fn(sample)
            for tok_name, enc in TOKENIZERS.items():
                tokens = count_tokens(noisy, enc)
                ratio = tokens / base_words
                results[(tok_name, cond_name)].append(ratio)

    return results

def report(results: dict, samples: list[str]) -> None:
    print("\n[3/3] Sonuçlar hesaplanıyor...\n")

    rows = []
    for (tok, cond), ratios in sorted(results.items()):
        mean  = statistics.mean(ratios)
        stdev = statistics.stdev(ratios)
        rows.append({
            "Tokenizer": tok,
            "Koşul":     cond,
            "Ortalama token/kelime": round(mean, 3),
            "Std sapma": round(stdev, 3),
            "n": len(ratios),
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # ── 2.1x İddiası Testi ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MAKALE İDDİASI TESTİ: clean → full_noisy oranı")
    print("="*60)
    for tok in TOKENIZERS:
        clean_mean  = statistics.mean(results[(tok, "clean")])
        noisy_mean  = statistics.mean(results[(tok, "full_noisy")])
        ratio       = noisy_mean / clean_mean
        verdict     = "✓ DESTEKLER" if ratio >= 2.0 else "✗ DESTEKLEMİYOR"
        print(f"  {tok:8s}  clean={clean_mean:.3f}  noisy={noisy_mean:.3f}  oran={ratio:.2f}x  {verdict}")

    # ── Faktör Ayrıştırması ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("FAKTÖR AYRIŞMASI: her gürültü tipinin katkısı")
    print("="*60)
    cond_order = ["clean", "+symbols", "+emojis", "+html", "+unicode", "full_noisy"]
    for tok in TOKENIZERS:
        print(f"\n  [{tok}]")
        baseline = statistics.mean(results[(tok, "clean")])
        for cond in cond_order:
            m = statistics.mean(results[(tok, cond)])
            delta = m - baseline
            pct   = (delta / baseline) * 100
            bar   = "█" * int(abs(pct) / 2)
            sign  = "+" if delta >= 0 else "-"
            print(f"    {cond:12s}  {m:.3f}  ({sign}{abs(pct):.1f}%)  {bar}")

    # ── Grafik ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('"Writing for Machines" — Token/Kelime Oranı Analizi', fontsize=13)

    for ax, tok in zip(axes, TOKENIZERS):
        means  = [statistics.mean(results[(tok, c)]) for c in cond_order]
        errors = [statistics.stdev(results[(tok, c)]) for c in cond_order]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#795548"]
        bars = ax.bar(cond_order, means, yerr=errors, capsize=4,
                      color=colors, edgecolor="white", alpha=0.85)
        ax.axhline(y=1.2, color="blue",   linestyle="--", alpha=0.5, label="Makale: clean (1.2)")
        ax.axhline(y=2.5, color="red",    linestyle="--", alpha=0.5, label="Makale: noisy (2.5)")
        ax.set_title(tok)
        ax.set_ylabel("Token / Kelime")
        ax.set_ylim(0, max(means) * 1.3)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = "results.png"
    plt.savefig(out, dpi=150)
    print(f"\nGrafik kaydedildi: {out}")

    # ── JSON Çıktı ──────────────────────────────────────────────────────────
    summary = {
        str(k): {
            "mean": round(statistics.mean(v), 4),
            "stdev": round(statistics.stdev(v), 4),
            "n": len(v),
        }
        for k, v in results.items()
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Veriler kaydedildi: results.json")

# ─── Ana Akış ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)
    samples = fetch_wikipedia_samples(SAMPLE_COUNT, WORDS_PER_SAMPLE)
    if len(samples) < 10:
        print("HATA: Yeterli Wikipedia verisi çekilemedi. İnternet bağlantısını kontrol edin.")
        exit(1)
    results = analyze(samples)
    report(results, samples)
