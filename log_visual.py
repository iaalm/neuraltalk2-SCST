#!/usr/bin/python3

import matplotlib.pyplot as plt
import json
import sys
import os

data = json.load(open(os.path.join(sys.argv[1],'model_id.json')))
val_name = ["CIDEr","Bleu_1","Bleu_2","Bleu_3","Bleu_4","METEOR","ROUGE_L"]
val_data = data["val_lang_stats_history"]
val_data = sorted(val_data.items(),key=lambda t:int(t[0]))

print(val_data[-1][0])
for name in val_name:
    x = []
    y = []
    for k,v in val_data:
        x.append(int(k))
        y.append(float(v[name]))
    line = plt.plot(x,y,label=name)[0]
    plt.text(x[-1]*1.01,y[-1]-0.01,name,color=line.get_color())
    print(name,y[-1])
#plt.legend(loc='upper left')
plt.show()
