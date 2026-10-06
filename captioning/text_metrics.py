"""Lexical metrics and repetition shared by current workflows."""
from data.RSCC.rscc import tokenize
from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge

def repetition(tokens):
    """Adjacent lexical repeats exclude standalone punctuation; case is retained."""
    words = [t for t in tokens if any(c.isalnum() for c in t)]
    adjacent = sum(a == b for a, b in zip(words, words[1:]))
    trigrams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
    return {'words': len(words), 'adjacent_repeat_fraction': adjacent / max(1, len(words) - 1),
            'repeated_trigram_fraction': 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0.}

def metrics(predictions):
    refs = [[' '.join(tokenize(p['reference'])[1:-1])] for p in predictions]
    hyps = [[p['prediction']] for p in predictions]
    bleu, _ = Bleu(4).compute_score(refs, hyps)
    rouge, _ = Rouge().compute_score(refs, hyps)
    return {'pairs': len(predictions), 'BLEU_1_to_4': bleu, 'ROUGE_L': rouge,
            'mean_adjacent_repeat_fraction': sum(p['repetition']['adjacent_repeat_fraction'] for p in predictions) / len(predictions),
            'mean_repeated_trigram_fraction': sum(p['repetition']['repeated_trigram_fraction'] for p in predictions) / len(predictions),
            'mean_output_whitespace_tokens': sum(len(p['prediction'].split()) for p in predictions) / len(predictions),
            'output_cap_reached': sum(p['output_cap_reached'] for p in predictions)}
