#!/usr/bin/python3

import matplotlib.pyplot as plt
import json
import sys
import os

startx = 0
dirname = sys.argv[1]
dirs = [dirname]
if not(len(sys.argv) > 2 and sys.argv[2] == '-n'):
    while dirname:
        data = json.load(open(os.path.join(dirname,'model_id.json')))
        dirname = os.path.dirname(data['opt']['start_from'])
        if dirname and dirname not in dirs and os.path.exists(os.path.join(dirname,'model_id.json')):
            dirs.append(dirname)
        else:
            break
    dirs.reverse()
print(dirs)

for dirname in dirs:
    data = json.load(open(os.path.join(dirname,'model_id.json')))
    val_name = ["CIDEr","Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L"]
    val_data = data["val_lang_stats_history"]
    val_data = sorted(val_data.items(),key=lambda t:int(t[0]))

    for name in val_name:
        x = []
        y = []
        for k,v in val_data:
            x.append(int(k)+startx)
            y.append(float(v[name]))
        plt.plot(x,y,label=name)[0]
    startx = startx + int(val_data[-1][0])
print(startx)
for name in val_name:
    plt.text(startx*1.02,val_data[-1][1][name]-0.01,name)
    print(name,val_data[-1][1][name])
#plt.legend(loc='upper left')
plt.show()
