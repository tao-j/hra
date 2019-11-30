import numpy as np
import pandas as pd
import os
import glob

path = './topic_document'
extension = 'txt'
os.chdir(path)
result = [i for i in glob.glob('*.{}'.format(extension))]

files = [path+"\\"+i for i in result]
df_list = []
for file in files:
    df_list.append(pd.read_csv(file, header=None, sep = "\\t"))

df = pd.concat(df_list, ignore_index=True)
df.head()

df.isna().sum(axis=0) #some do not have worker id (drop these)
df = df.dropna()
df = df[df[10]==0] #get only those that were not machine labelled/evaluating Mturk


df.hist()

import xlwings as xw

sht = xw.Book().sheets[0]
sht.range('A1').value = df

###########
df = pd.read_csv("C:\\Users\\ASUS\\Downloads\\judgments.csv", encoding = "ISO-8859-1")
df1 = pd.read_csv("MTURK.qrel", sep="\\t", header=None)

df = df[['Input.query', 'WorkerId','Input.docA', 'Input.docB','Answer.radio']]
(df['Input.docA']==df['Input.docB']).any()
''' how to handle S in Answer.radio?'''

df = df[df["Answer.radio"]!="s"]
t1 = pd.concat([df[['Input.docA', 'Input.docB']].stack(), df[['Input.docA', 'Input.docB']].stack().astype('category').cat.codes+1], axis=1).drop_duplicates()
df[['Input.docA', 'Input.docB']] = (df[['Input.docA', 'Input.docB']].stack().astype('category').cat.codes+1).unstack()
t1 = t1.set_index(t1[0])[[1]].sort_values(1).to_dict()
df1[[4,5]] = df1[[1,2]].replace(t1[1])



d1 = df[df["Answer.radio"]=="b"]
df = df[df["Answer.radio"]=="a"]
d1 = d1.rename(index=str, columns={'Input.docA': 'loser', 'Input.docB': 'winner'})
df = df.rename(index=str, columns={'Input.docB': 'loser', 'Input.docA': 'winner'})
df = pd.concat([d1,df])
df = df[['Input.query','WorkerId','winner','loser']]
df = df.sort_values("WorkerId")

len(np.sort(df[['winner', 'loser']].stack().unique()))

for i in df['Input.query'].unique():    
    df[df['Input.query']==i].drop(columns=["Input.query"]).to_csv('C:\\Users\\ASUS\\Google Drive\\UVa Classes\\Semester 8\\data-ashish\\MIREX\\all_pair_'+i+'.txt', sep=" ", header=False, index=False)

i=df1[1].unique()[0]
for i in df1[1].unique():    
    t2 = df1[df1[1]==i][[4,5,3]] #(topic, document, rank)
    #print(i)
    t3 = pd.Series(np.zeros(max(t2[5])))
    for j in range(len(t2)):
        t3[t2[5].iloc[j]-1] = t2[3].iloc[j]
    t3.to_csv('C:\\Users\\ASUS\\Google Drive\\UVa Classes\\Semester 8\\data-ashish\\MIREX\\doc_info_'+i+'.txt', sep=" ", header=False, index=False)

