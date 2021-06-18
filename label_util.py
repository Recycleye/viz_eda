from nltk.corpus import wordnet

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def word_distance(w1: str, w2: str) -> float:
    w1 = w1.replace(' ', '_')
    w2 = w2.replace(' ', '_')
    w1_syns = wordnet.synsets(w1)
    w2_syns = wordnet.synsets(w2)
    score = 0
    try:
        score = w1_syns[0].wup_similarity(w2_syns[0])
    except Exception as e:
        print(f"error in word_distance({w1}, {w2}) {e}")
    return score


if __name__ == '__main__':
    # import nltk
    # nltk.download('wordnet')
    # print(word_distance("motorbike", "motorcycle"))
    print(word_distance("dining table", "table"))
