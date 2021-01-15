from scipy import stats
from scipy.io import arff
import pandas as pd

###################################
# Entropy Of Each Feature
###################################

data = arff.loadarff('Training Dataset original.arff')
df = pd.DataFrame(data[0])
df.head()

dictionary = {}
SF = 40
i = 1
entropy_list = []
for item in df:
    data = df[item]
    pd_series = pd.Series(data)
    counts = pd_series.value_counts()
    entropy = stats.entropy(counts)
    print(entropy)
    if str(item) not in dictionary:
        dictionary[str(item)] = []
    dictionary[str(item)] = entropy
    i += 1
