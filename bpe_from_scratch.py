"""
BPE Tokenizer From Scratch
Week 1: See why non-English text costs more tokens

No libraries. Just the algorithm, two corpora, and the numbers.
"""


def get_pair_counts(token_sequences):
    """Count frequency of every adjacent pair across all sequences"""
    counts = {}
    for seq in token_sequences:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(token_sequences, pair, new_token):
    """Replace every occurrence of pair with new_token in all sequences"""
    merged = []
    for seq in token_sequences:
        new_seq = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                new_seq.append(new_token)
                i += 2
            else:
                new_seq.append(seq[i])
                i += 1
        merged.append(new_seq)
    return merged


def train_bpe(corpus_text, num_merges=50, verbose=True):
    """
    Train BPE tokenizer from scratch.
    
    Returns:
        merge_rules: list of (pair, new_token) in order they were learned
        vocab: final vocabulary
    """
    # Step 1: Start with UTF-8 bytes as initial tokens
    # Each character becomes a sequence of bytes
    token_sequences = []
    for word in corpus_text.split():
        byte_seq = list(word.encode('utf-8'))
        token_sequences.append(byte_seq)
    
    merge_rules = []
    next_token_id = 256  # First 256 are raw bytes
    
    if verbose:
        total_tokens_start = sum(len(seq) for seq in token_sequences)
        print(f"  Starting tokens: {total_tokens_start}")
    
    for step in range(num_merges):
        # Count all adjacent pairs
        pair_counts = get_pair_counts(token_sequences)
        
        if not pair_counts:
            break
        
        # Find the most frequent pair
        best_pair = max(pair_counts, key=pair_counts.get)
        best_count = pair_counts[best_pair]
        
        if best_count < 2:
            break  # No pair appears more than once
        
        # Merge it
        token_sequences = merge_pair(token_sequences, best_pair, next_token_id)
        merge_rules.append((best_pair, next_token_id, best_count))
        
        if verbose and step < 10:
            # Show what's being merged (decode bytes to show readable text)
            left = chr(best_pair[0]) if best_pair[0] < 128 else f"[{best_pair[0]}]"
            right = chr(best_pair[1]) if best_pair[1] < 128 else f"[{best_pair[1]}]"
            print(f"  Merge {step + 1}: '{left}' + '{right}' (count: {best_count}) -> token {next_token_id}")
        
        next_token_id += 1
    
    if verbose:
        total_tokens_end = sum(len(seq) for seq in token_sequences)
        print(f"  Final tokens: {total_tokens_end}")
        print(f"  Compression: {total_tokens_start} -> {total_tokens_end} ({total_tokens_end/total_tokens_start:.1%})")
    
    return merge_rules, token_sequences


def tokenize(text, merge_rules):
    """Apply learned merge rules to new text"""
    # Start with bytes
    tokens = list(text.encode('utf-8'))
    
    # Apply each merge rule in order
    for (pair, new_token, _) in merge_rules:
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                tokens[i] = new_token
                tokens.pop(i + 1)
            else:
                i += 1
    
    return tokens


def show_tokenization(text, tokens, label=""):
    """Display how text was tokenized"""
    byte_count = len(text.encode('utf-8'))
    print(f"  {label}")
    print(f"    Text:       {text}")
    print(f"    UTF-8 bytes: {byte_count}")
    print(f"    BPE tokens:  {len(tokens)}")
    print(f"    Ratio:       {len(tokens)/byte_count:.2f} (lower = better compression)")
    print()


# ============================================================
# EXPERIMENT: Train on English-heavy data, test on both
# ============================================================

if __name__ == "__main__":
    
    # --- Corpora ---
    # Simulating real-world: English text dominates training data
    
    english_corpus = """
    The quick brown fox jumps over the lazy dog. The dog was sleeping 
    in the sun while the fox was running through the forest. The forest 
    was green and full of trees and the trees were tall and the leaves 
    were green. The sun was shining and the birds were singing and the 
    wind was blowing through the trees. The fox jumped over the fence 
    and ran into the field. The field was large and open and the grass 
    was tall. The farmer was working in the field and the dog was 
    watching the farmer. The quick brown fox jumps over the lazy dog 
    again and again. The dog barked at the fox and the fox ran away.
    Payment processing is important for global commerce. Users sign up 
    and create accounts to access subscription services. The billing 
    system processes credit cards and digital payments every month.
    The customer enters their email address and password to register.
    Free trial starts immediately after the account is created.
    """
    
    japanese_corpus = """
    素早い茶色の狐が怠惰な犬を飛び越える。犬は太陽の下で寝ていた。
    狐は森を走っていた。森は緑で木がたくさんあった。
    """

    chinese_corpus = """
    敏捷的棕色狐狸跳过了懒惰的狗。狗在阳光下睡觉。
    狐狸在森林里奔跑。森林是绿色的，有很多树木。
    """
    
    # The key: what proportion is English vs Japanese?
    # Real-world tokenizers: ~90% English training data
    
    print("=" * 60)
    print("BPE TOKENIZER FROM SCRATCH")
    print("Week 1: Why non-English costs more")
    print("=" * 60)
    
    # --- Experiment 1: Train on English-heavy corpus ---
    print("\n[EXPERIMENT 1] Train on 90% English corpus")
    print("-" * 60)
    
    # Combine with English dominating
    training_corpus = (english_corpus + " ") * 3 + japanese_corpus
    
    print(f"\nTraining ({len(training_corpus.split())} words, mostly English)...")
    merge_rules, _ = train_bpe(training_corpus, num_merges=80)
    
    # Test sentences with equivalent meaning
    test_en = "The user creates an account and starts a free trial"
    test_ja = "ユーザーがアカウントを作成して無料トライアルを開始する"
    test_fr = "L'utilisateur crée un compte et commence un essai gratuit"
    test_zh = "用户创建账户并开始免费试用"
    test_ko = "사용자가 계정을 만들고 무료 체험을 시작합니다"
    
    print(f"\nTokenizing equivalent sentences:")
    print()
    
    results = {}
    for label, text in [
        ("English", test_en),
        ("Japanese", test_ja),
        ("French", test_fr),
        ("Chinese", test_zh),
        ("Korean", test_ko),
    ]:
        tokens = tokenize(text, merge_rules)
        show_tokenization(text, tokens, label=label)
        results[label] = len(tokens)
    
    # Show the gap
    en_count = results["English"]
    print("=" * 60)
    print("TOKEN COUNT COMPARISON (same meaning)")
    print("-" * 60)
    for lang, count in results.items():
        ratio = count / en_count
        bar = "█" * int(ratio * 20)
        marker = " ← baseline" if lang == "English" else f" ← {ratio:.1f}x English"
        print(f"  {lang:<10} {count:>3} tokens  {bar}{marker}")
    
    print()
    print("=" * 60)
    
    # --- Experiment 2: What if we train on balanced data? ---
    print("\n[EXPERIMENT 2] Train on BALANCED corpus")
    print("-" * 60)
    
    # Repeat Japanese to balance
    balanced_corpus = english_corpus + " " + (japanese_corpus + " ") * 10
    
    print(f"\nTraining ({len(balanced_corpus.split())} words, balanced)...")
    balanced_rules, _ = train_bpe(balanced_corpus, num_merges=80)
    
    print(f"\nTokenizing same sentences with balanced tokenizer:")
    print()
    
    balanced_results = {}
    for label, text in [
        ("English", test_en),
        ("Japanese", test_ja),
    ]:
        tokens = tokenize(text, balanced_rules)
        show_tokenization(text, tokens, label=label)
        balanced_results[label] = len(tokens)
    
    print("=" * 60)
    print("BEFORE vs AFTER BALANCING")
    print("-" * 60)
    print(f"  English-heavy tokenizer:")
    print(f"    English:  {results['English']} tokens")
    print(f"    Japanese: {results['Japanese']} tokens ({results['Japanese']/results['English']:.1f}x)")
    print(f"  Balanced tokenizer:")
    print(f"    English:  {balanced_results['English']} tokens")
    print(f"    Japanese: {balanced_results['Japanese']} tokens ({balanced_results['Japanese']/balanced_results['English']:.1f}x)")
    
    gap_before = results['Japanese'] / results['English']
    gap_after = balanced_results['Japanese'] / balanced_results['English']
    
    if gap_after < gap_before:
        improvement = (1 - gap_after / gap_before) * 100
        print(f"\n  → Gap reduced by {improvement:.0f}% with balanced training data")
    
    print()
    print("=" * 60)

    # --- Experiment 3: Balanced English + Chinese ---
    print("\n[EXPERIMENT 3] Train on BALANCED corpus (English + Chinese)")
    print("-" * 60)

    balanced_zh_corpus = english_corpus + " " + (chinese_corpus + " ") * 10

    print(f"\nTraining ({len(balanced_zh_corpus.split())} words, balanced)...")
    balanced_zh_rules, _ = train_bpe(balanced_zh_corpus, num_merges=80)

    print(f"\nTokenizing same sentences with balanced tokenizer:")
    print()

    balanced_zh_results = {}
    for label, text in [
        ("English", test_en),
        ("Chinese", test_zh),
    ]:
        tokens = tokenize(text, balanced_zh_rules)
        show_tokenization(text, tokens, label=label)
        balanced_zh_results[label] = len(tokens)

    # Compare English-heavy (Exp 1) vs balanced Chinese (Exp 3)
    print("=" * 60)
    print("BEFORE vs AFTER BALANCING (Chinese)")
    print("-" * 60)
    print(f"  English-heavy tokenizer:")
    print(f"    English: {results['English']} tokens")
    print(f"    Chinese: {results['Chinese']} tokens ({results['Chinese']/results['English']:.1f}x)")
    print(f"  Balanced tokenizer:")
    print(f"    English: {balanced_zh_results['English']} tokens")
    print(f"    Chinese: {balanced_zh_results['Chinese']} tokens ({balanced_zh_results['Chinese']/balanced_zh_results['English']:.1f}x)")

    gap_before_zh = results['Chinese'] / results['English']
    gap_after_zh = balanced_zh_results['Chinese'] / balanced_zh_results['English']

    if gap_after_zh < gap_before_zh:
        improvement_zh = (1 - gap_after_zh / gap_before_zh) * 100
        print(f"\n  → Gap reduced by {improvement_zh:.0f}% with balanced training data")

    # Side-by-side: Japanese vs Chinese balancing effect
    print()
    print("=" * 60)
    print("JAPANESE vs CHINESE: BALANCING EFFECT COMPARISON")
    print("-" * 60)
    print(f"  {'':12} {'Eng-heavy':>12} {'Balanced':>10} {'Gap reduction':>14}")
    print(f"  {'Japanese':<12} {results['Japanese']:>6} tokens  {balanced_results['Japanese']:>6} tokens  {(1 - gap_after/gap_before)*100:>10.0f}%")
    print(f"  {'Chinese':<12} {results['Chinese']:>6} tokens  {balanced_zh_results['Chinese']:>6} tokens  {(1 - gap_after_zh/gap_before_zh)*100:>10.0f}%")
    print()
    print("=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
  BPE merges frequent byte pairs first.
  
  English letters are 1 byte each (ASCII).
  CJK characters are 3 bytes each (UTF-8).
  
  In English-heavy training data:
    - 'th' merges early (very common pair)
    - 'the' merges next
    - English words compress to 1-2 tokens
  
  But CJK bytes rarely repeat in the same patterns,
  so they never get merged → each character stays as
  2-3 separate tokens.
  
  The inequity isn't a bug — it's a direct consequence
  of training data distribution.
    """)
