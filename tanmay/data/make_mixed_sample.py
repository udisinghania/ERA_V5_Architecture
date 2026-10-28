"""Build the mixed EN+HI training text. Two tiers, nothing manufactured.

TIER 1 — gate (default, ~20k sentences, 6.7 MB, no login, <1 min):
  XNLI test split (facebook/xnli): 5,010 premise/hypothesis pairs, each row
  carrying HUMAN-TRANSLATED en + hi (verified 25 Oct 2026: dict keys include
  'en','hi'; e.g. HI premise is a real translation, not MT). Premises AND
  hypotheses of both languages, interleaved en/hi line by line.

    python data/make_mixed_sample.py
    # writes data/mixed_en_hi_xnli.txt (regenerate on each instance; never commit)

TIER 2 — GPU scale-up (streaming, no full download):
  EN = HuggingFaceFW/fineweb-edu, HI = ai4bharat/IndicCorpV2 hindi
  (both open-access, verified 25 Oct 2026; flores is gated — do NOT use).

    python data/make_mixed_sample.py --scale --n 5000

HELD-OUT EVAL (XNLI validation, 2490 rows, same en/hi interleaving, never trained on):

    python data/make_mixed_sample.py --split validation
    # writes data/mixed_en_hi_xnli_val.txt (never commit)

Requires: pip install pandas pyarrow (tier 1); + datasets (tier 2).
"""

import argparse

XNLI_URL = (
    "https://huggingface.co/datasets/facebook/xnli/resolve/"
    "refs%2Fconvert%2Fparquet/all_languages/{split}/0000.parquet"
)


def _lang_text(field_val, lang):
    """XNLI parquet uses two layouts: premise is {lang: text}, hypothesis is
    {language: [...], translation: [...]}. Returns text or ''."""
    if isinstance(field_val, dict):
        if lang in field_val and isinstance(field_val[lang], str):
            return field_val[lang]
        langs = field_val.get("language", [])
        trans = field_val.get("translation", [])
        if lang in list(langs):
            return trans[list(langs).index(lang)]
    return ""


def _xnli_lines(d):
    """Yield (lang, text) in the interleaved order: per row, en premise, en hypothesis, hi premise, hi hypothesis."""
    for _, r in d.iterrows():  # interleaved = gradual blend by construction
        for lang in ("en", "hi"):
            for field in ("premise", "hypothesis"):
                t = _lang_text(r[field], lang).strip().replace("\n", " ")
                if len(t) > 20:
                    yield lang, t


def tier1(out, split="test"):
    import pandas as pd

    d = pd.read_parquet(XNLI_URL.format(split=split))
    assert _lang_text(d.premise.iloc[0], "hi"), "XNLI schema changed (premise.hi)"
    assert _lang_text(d.premise.iloc[0], "en"), "XNLI schema changed (premise.en)"
    exclude = set()
    if split != "test":  # held-out set must share no exact line with the train sample (XNLI test)
        exclude = {t for _, t in _xnli_lines(pd.read_parquet(XNLI_URL.format(split="test")))}
    n_hi = n_en = dropped = 0
    with open(out, "w") as f:
        for lang, t in _xnli_lines(d):
            if t in exclude:
                dropped += 1
                continue
            f.write(t + "\n")
            if lang == "hi":
                n_hi += 1
            else:
                n_en += 1
    extra = f" (dropped {dropped} lines also in the train sample)" if split != "test" else ""
    print(f"wrote {out}: EN lines={n_en} HI lines={n_hi}{extra}")


def tier2(out, n):
    from datasets import load_dataset

    def take(stream, k):
        got = []
        for ex in stream:
            t = (ex.get("text") or "").strip().replace("\n", " ")
            if len(t) > 100:
                got.append(t)
            if len(got) >= k:
                break
        return got

    en = take(load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True), n)
    hi = take(
        load_dataset("ai4bharat/IndicCorpV2", "hindi", split="train", streaming=True), n
    )
    print(f"collected EN={len(en)} HI={len(hi)}")
    with open(out, "w") as f:
        for e, h in zip(en, hi):
            f.write(e + "\n" + h + "\n")
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", action="store_true", help="tier 2 instead of tier 1")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--split", default="test", choices=["test", "validation"],
        help="XNLI split for tier 1. 'validation' (2490 rows) is the held-out eval set; "
        "default out data/mixed_en_hi_xnli_val.txt. Never commit the output.",
    )
    a = ap.parse_args()
    if a.scale:
        tier2(a.out or "data/mixed_en_hi_10k.txt", a.n)
    else:
        default = "data/mixed_en_hi_xnli.txt" if a.split == "test" else "data/mixed_en_hi_xnli_val.txt"
        tier1(a.out or default, a.split)


if __name__ == "__main__":
    main()
