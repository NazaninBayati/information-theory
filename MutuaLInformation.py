from scipy import stats
from scipy.io import arff
import pandas as pd
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
###################################
# Mutual Information Of Features
###################################

data = arff.loadarff('Training Dataset original.arff')
df = pd.DataFrame(data[0])
df.head()

dictionary = {}
SF = 40
i = 1
entropy_list = []
#print(df[0])
listt = df.keys()

for item in tqdm(df):
    j = 1
    for i in listt[j:listt.__len__()-1]:

        data1 = df[item]
        data2 = df[i]
        pd_series_H1 = pd.Series(data1)
        pd_series_H2 = pd.Series(data2)
        pd_series = pd.Series(data1,data2)
        counts_H1 = pd_series_H1.value_counts()
        counts_H2 = pd_series_H2.value_counts()
        counts = pd_series.value_counts()
        entropy_H1 = stats.entropy(counts_H1/sum(counts_H1) , base=2)
        entropy_H2 = stats.entropy(counts_H2 / sum(counts_H2), base=2)
        entropy = stats.entropy(counts / sum(counts), base=2)

        if str([item,i]) not in dictionary:
            dictionary[str([item,i])]=[]
        dictionary[str([item,i])] = entropy_H1 + entropy_H2 - entropy
        #print(dictionary[str([item,i])] )
        j = j+1

    #print(stats.entropy([0.5, 0.5]))  # entropy of 0.69, expressed in nats
    #print(mutual_info_classif(a.reshape(-1, 1), b,discrete_features=True))  # mutual information of 0.69, expressed in nats
   # print(mutual_info_score(a, b))
