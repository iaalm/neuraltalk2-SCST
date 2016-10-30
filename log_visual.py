#!/usr/bin/python3

import matplotlib.pyplot as plt
import json
import sys
import os
import re

rrr = re.compile(r'\.t7$')
for filename in sys.argv[1:]:
    startx = 0
    dirs = [filename]
    while filename:
        data = json.load(open(filename))
        filename = rrr.sub('.json', data['opt']['start_from'])
        if filename and filename not in dirs and os.path.exists(filename):
            dirs.append(filename)
        else:
            break
    dirs.reverse()
    print(dirs)

    for filename in dirs:
        data = json.load(open(filename))
        val_name = ["CIDEr", "Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR",
                    "ROUGE_L"]
        val_max = {}
        val_data = data["val_lang_stats_history"]
        val_data = sorted(val_data.items(), key=lambda t: int(t[0]))

        for name in val_name:
            x = []
            y = []
            max_result = 0
            for k, v in val_data:
                x.append((int(k)+startx)/1000)
                y.append(float(v[name]))
                if max_result < float(v[name]) and name == 'CIDEr':
                    cider_max = startx + int(k)
                max_result = max(max_result, float(v[name]))
            val_max[name] = max_result
            plt.plot(x, y, label=name)[0]
        startx = startx + int(val_data[-1][0])
    print(cider_max/1000, startx/1000)
    for name in val_name:
        plt.text(startx*1.02, val_data[-1][1][name]-0.01, name)
        print(name, val_max[name])
# plt.legend(loc='lower left')
plt.show()
