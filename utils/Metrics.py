import nltk

def get_bleu(hypothesis, reference):
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis)