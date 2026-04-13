# multilingual-token-cost

Why does the same sentence cost 2.3x more tokens in Japanese than English? This repo investigates tokenization cost inequity across languages by building a BPE tokenizer from scratch.

## Background

Most major LLMs (GPT, Claude, LLaMA) use [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte-pair_encoding) (BPE) for tokenization. BPE learns merge rules from a training corpus — and when that corpus is predominantly English, English gets better compression while other languages pay more per API call for the same meaning.

Instead of accepting this as a given and working around it by downsizing prompts, this project digs into the root cause.

## What's in the repo

`bpe_from_scratch.py` — A minimal BPE tokenizer implementation with three experiments:

**Experiment 1: English-heavy training.** Trains on a corpus that's ~90% English, then tokenizes the same sentence in five languages.

| Language | Tokens | vs English |
|----------|--------|------------|
| English  | 34     | baseline   |
| Japanese | 77     | 2.3x       |
| French   | 48     | 1.4x       |
| Chinese  | 39     | 1.1x       |
| Korean   | 65     | 1.9x       |

**Experiment 2: Balanced training (English + Japanese).** Inflates Japanese training data to give it equal representation.

| | English-heavy | Balanced |
|---|---|---|
| English | 34 tokens | 43 tokens |
| Japanese | 77 tokens | 64 tokens (1.5x) |

Gap reduced by 34% — but English got worse. Merge slots are a fixed resource.

**Experiment 3: Balanced training (English + Chinese).** Replaces Japanese with Simplified Chinese.

| | English-heavy | Balanced |
|---|---|---|
| English | 34 tokens | 43 tokens |
| Chinese | 39 tokens | 39 tokens (no change) |

Chinese didn't improve at all. Its byte patterns are too scattered for BPE merges to cascade.

## Three factors behind the inequity

**UTF-8 encoding.** CJK characters are 3 bytes each; Latin characters are 1 byte. A 3x cost disadvantage before any merging begins.

**Training data distribution.** BPE is a popularity contest. The language with more representation in the training corpus wins more merge slots and gets better compression.

**Character set structure.** Japanese kana share UTF-8 byte prefixes, so balancing helps. Chinese characters are spread across wider byte ranges, making merges harder even with balanced data. However, Chinese compensates with language density — fewer characters needed to express the same meaning.

## Run it

```bash
python bpe_from_scratch.py
```

No dependencies. No API keys. Just Python 3.

## What's next

Exploring vocabulary extension approaches — bolting language-specific tokens onto existing tokenizers to reduce cost without full retraining. Similar to the approach used by [Chinese LLaMA](https://github.com/ymcui/Chinese-LLaMA-Alpaca).

## License

MIT
