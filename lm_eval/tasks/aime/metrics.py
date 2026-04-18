def is_degenerating_text(
    text: str,
    max_n: int = 3,
    threshold: float = 0.9,
) -> bool:
    words = text.split()
    for n in range(1, max_n + 1):
        if len(words) < n:
            continue
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        repetition = 1 - (len(set(ngrams)) / len(ngrams))
        if repetition > threshold:
            return True
    return False

def degeneration(items):
    """items = [gold, prediction]"""
    gold, pred = items
    return int(is_degenerating_text(pred.lower()))