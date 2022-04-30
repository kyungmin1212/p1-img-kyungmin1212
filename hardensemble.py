import os
import glob
import pandas as pd
import numpy as np
data_path='/opt/ml/realcode/output'
choice_data=['focalefficientb4SGD0.01steplr.csv','label0.2efficientb0SGD0.1cosine.csv']

mt=[]
info_path='/opt/ml/input/data/eval/info.csv'
info = pd.read_csv(info_path)
for data in choice_data:
    csv_path=f'{data_path}/{data}'
    print(csv_path)
    a=pd.read_csv(csv_path)
    mt.append(list(a['ans']))
              
mt=list(np.array(mt).T)
answerlist=[]
from collections import Counter
for data in mt:
    counter=Counter(data)
    answerlist.append(counter.most_common()[0][0])
    
info['ans']=answerlist
info.to_csv('/opt/ml/realcode/output/ensembel.csv',index=False)

