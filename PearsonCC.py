import pandas as pd
from tqdm import tqdm
from scipy import stats

###################################
# Pearson correlation coefficient and p-value Of Features
###################################
df = pd.read_csv('Training Dataset.csv')
#data = arff.loadarff('Training Dataset original.arff')
#df = pd.DataFrame(data[0])
df.head()

dictionary = {}
SF = 40
i = 1
entropy_list = []
#print(df[0])
listt = df.keys()

for item in tqdm(df):
    j = 1
    for i in range( listt[j:listt.__len__()-1].__len__()):

        X = df[item].values
        Y = df[listt[i]].values
        a = stats.pearsonr(X,Y)


        if str([item,listt[i],listt[i+1]]) not in dictionary:
            dictionary[str([item,listt[i],listt[i+1]])]=[]
        dictionary[str([item,listt[i],listt[i+1]])] = a[0]
        #print(dictionary[str([item,i])] )
        j = j+1

