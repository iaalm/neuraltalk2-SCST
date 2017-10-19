import json
import math
from functools import reduce

# Compute BLEU
# SiMon
# Note there are two main differen with coco-caption
# 1. While compute reflen, we use "shorest" policy rather than "closest"
# 2. only tokenize sentences only by space which is different Stanford NLP

ref_file = 'coco-caption/annotations/captions_val2014.json'
input_file = 'coco-caption/val.json'


def ngram(sent, n):
    l = len(sent) - n + 1
    for i in range(l):
        yield sent[i:i+n]


def count_ngram(w, sent):
    if type(sent[0]) != list:
        sent = [sent]
    res = 0
    l = len(w)
    for s in sent:
        count = 0
        for i in range(len(s) - l + 1):
            if w == s[i:i+l]:
                count = count + 1
        res = max(count, res)
    return res


def max_count(sents, n):
    res = {}
    for sent in sents:
        for w in ngram(sent, n):
            w_key = ' '.join(w)
            res[w_key] = max(res.get(w_key, 0), count_ngram(w, sent))
    return res

with open(ref_file) as fd:
    ref_list = json.load(fd)['annotations']
    ref_dict = {}
    for i in ref_list:
        ref_dict.setdefault(i['image_id'], [])
        ref_dict[i['image_id']]\
            .append(i['caption'].lower().strip().replace('.', '').split())


with open(input_file) as fd:
    input_sen = json.load(fd)['val_predictions']


bleus = [0] * 4
for bleu_n in range(1, 5):
    res_success = 0
    res_guess = 0
    inputLen = 0
    refLen = 0
    for i in input_sen:
        img_id = i['image_id']
        if img_id in ref_dict:
            inputLen += len(i['caption'].split())
            # use "shorest" here
            # different from default "closest" in coco-caption
            refLen += reduce(lambda x, y: min(x, y),
                             [len(i) for i in ref_dict[img_id]])
            ref_count = max_count(ref_dict[img_id], bleu_n)
            guess_count = max_count([i['caption'].split()], bleu_n)
            success_list = [min(ref_count.get(i, 0), guess_count[i])
                            for i in guess_count]
            # print(guess_list, input_count)
            # print(reduce(lambda x, y: x+y, guess_list))
            # print(reduce(lambda x, y: x + y, input_count.values()))
            res_success += reduce(lambda x, y: x+y, success_list)
            res_guess += reduce(lambda x, y: x + y, guess_count.values())
    print('guess', res_guess)
    print('success', res_success)
    bleus[bleu_n - 1] = res_success / res_guess \
        * math.exp(min(0, 1 - refLen/inputLen))
print('testlen:', inputLen)
print('reflen:', refLen)
bleu = 1
for i in range(4):
    bleu *= bleus[i]
    print(bleu_n, bleu**(1/(i+1)))
