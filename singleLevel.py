from scipy import stats
from scipy.io import arff
import pandas as pd
from tqdm import tqdm

class singleLevel:
    def MI(self,df):
        dictionary = {}
        result = 0.00
        listt = df.keys()
        for item in (df):
            j = 1
            for i in listt[j:listt.__len__() - 1]:

                data1 = df[item]
                data2 = df[i]
                pd_series_H1 = pd.Series(data1)
                pd_series_H2 = pd.Series(data2)
                pd_series = pd.Series(data1, data2)
                counts_H1 = pd_series_H1.value_counts()
                counts_H2 = pd_series_H2.value_counts()
                counts = pd_series.value_counts()
                entropy_H1 = stats.entropy(counts_H1 / sum(counts_H1), base=2)
                entropy_H2 = stats.entropy(counts_H2 / sum(counts_H2), base=2)
                entropy = stats.entropy(counts / sum(counts), base=2)
                result = result + entropy_H1 + entropy_H2 - entropy
                # print(dictionary[str([item,i])] )
                j = j + 1
        return result

    def entropy(self,df):

        i = 1
        entropy = 0.00
        dictionary = {}
        for item in (df):
            data = df[item]
            pd_series = pd.Series(data)
            counts = pd_series.value_counts()
            total = sum(counts)
            entropy = entropy + stats.entropy(counts / total, base=2)
            i += 1
        return entropy

    def reader(self):
        data = arff.loadarff('Training Dataset original.arff')
        self.df = pd.DataFrame(data[0])
        self.df.head()
        return self.df
