from scipy import stats
from scipy.io import arff
import pandas as pd
from tqdm import tqdm

class singleLevel:
    def entropy(self,df):

        i = 1
        entropy = 0.00
        dictionary = {}
        for item in tqdm(df):
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
