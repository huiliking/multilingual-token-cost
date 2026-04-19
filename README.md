# multilingual-token-cost

Why does the same sentence cost 2.3x more tokens in Japanese than English? This repo investigates tokenization cost inequity across languages by building a BPE tokenizer from scratch and testing whether vocabulary extension can close the gap.

## Background

Most major LLMs (GPT, Claude, LLaMA) use [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE) for tokenization, originally introduced for NLP by [Sennrich et al., 2016](https://arxiv.org/abs/1508.07909). BPE learns merge rules from a training corpus — and when that corpus is predominantly English, English text gets better compression while other languages pay more tokens for the same meaning.

## Experiments

### Experiment 1: English-Heavy BPE

Train a BPE tokenizer on a 90% English corpus. Tokenize the same sentence in five languages.

| Language | Tokens | vs English |
|----------|--------|------------|
| English | 34 | baseline |
| Japanese | 77 | 2.3x |
| Korean | 65 | 1.9x |
| French | 48 | 1.4x |
| Chinese | 39 | 1.1x |

### Experiment 2: Balanced Training Data (Japanese)

Inflate Japanese training data to balance the corpus. Japanese improved (77 → 64 tokens), but English got worse (34 → 43). Merge slots are a zero-sum resource.

### Experiment 3: Balanced Training Data (Chinese)

Repeat with Chinese. Chinese didn't improve at all — its byte patterns are too scattered for BPE merges to cascade. But Chinese has a compensating advantage: language density (fewer characters to express the same meaning).

### Experiment 4: Bolt-On Vocabulary

Add language-specific tokens (Chinese characters/bigrams, Japanese hiragana/katakana/words, Korean syllables, French accented characters/words) to the tokenizer after training, without retraining. Inspired by [Chinese LLaMA](https://arxiv.org/abs/2304.08177).

| Language | Before | After | Reduction | vs English |
|----------|--------|-------|-----------|------------|
| Japanese | 77 | 11 | 86% | 2.3x → 0.3x |
| Chinese | 39 | 9 | 77% | 1.1x → 0.3x |
| French | 48 | 22 | 54% | 1.4x → 0.6x |
| Korean | 65 | 43 | 34% | 1.9x → 1.3x |
| English | 34 | 34 | — | baseline |

The tokenizer-side fix works. But making the model understand new tokens requires continued pretraining — expensive, risky (catastrophic forgetting), and architecturally constrained. See the [full writeup](https://www.linkedin.com/in/YOUR_PROFILE) for details.

## Three Root Causes of Tokenization Inequity

1. **UTF-8 encoding.** CJK characters are 3 bytes each; Latin characters are 1 byte. That's a 3x cost disadvantage before any merging begins.
2. **Training data distribution.** BPE is a popularity contest. The language with more text in the corpus wins more merge slots. English dominates most training corpora.
3. **Character set structure.** Some languages (like Chinese) have byte patterns too scattered for BPE merges to cascade effectively, regardless of training data balance.

## Files

- `bpe_from_scratch.py` — Experiments 1–3: BPE tokenizer from scratch, English-heavy vs balanced training
- `bolt_on_vocab_experiment.py` — Experiment 4: Bolt-on vocabulary extension and cost analysis

## Run

```bash
python bpe_from_scratch.py
python bolt_on_vocab_experiment.py
```

No dependencies. Pure Python. No API keys needed.

## Related

- [BPE original NLP paper](https://arxiv.org/abs/1508.07909) — Sennrich et al., 2016
- [Chinese LLaMA](https://arxiv.org/abs/2304.08177) — Vocabulary extension approach
- [ICML 2025 Tokenization Workshop](https://tokenization-workshop.github.io/) — Active research on alternatives to BPE

## License

MIT
