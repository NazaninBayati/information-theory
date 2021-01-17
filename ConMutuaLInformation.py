from scipy.io import arff
import pandas as pd
from tqdm import tqdm
from pyitlib import discrete_random_variable as drv

###################################
# Conditional Mutual Information Of Features
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
        Z = df[listt[i+1]].values
        b = X.reshape(-1,1)[0]
        a = drv.information_mutual_conditional(X, Y, Z)
        print(a)

        if str([item,listt[i],listt[i+1]]) not in dictionary:
            dictionary[str([item,listt[i],listt[i+1]])]=[]
        dictionary[str([item,listt[i],listt[i+1]])] = a
        #print(dictionary[str([item,i])] )
        j = j+2

    #print(stats.entropy([0.5, 0.5]))  # entropy of 0.69, expressed in nats
    #print(mutual_info_classif(a.reshape(-1, 1), b,discrete_features=True))  # mutual information of 0.69, expressed in nats
   # print(mutual_info_score(a, b))
