import json
from functools import reduce

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
    res_total = 0
    res_acc = 0
    for i in input_sen:
        img_id = i['image_id']
        if img_id in ref_dict:
            ref_count = max_count(ref_dict[img_id], bleu_n)
            input_count = max_count([i['caption'].split()], bleu_n)
            guess_list = [min(ref_count.get(i, 0), input_count[i])
                          for i in input_count]
            res_total += 1
            # print(guess_list, input_count)
            # print(reduce(lambda x, y: x+y, guess_list))
            # print(reduce(lambda x, y: x + y, input_count.values()))
            res_acc += reduce(lambda x, y: x+y, guess_list) \
                / reduce(lambda x, y: x + y, input_count.values())
    bleus[bleu_n - 1] = res_acc / res_total
bleu = 1
for i in range(4):
    bleu *= bleus[i]
    print(bleu_n, bleu**(1/(i+1)))
