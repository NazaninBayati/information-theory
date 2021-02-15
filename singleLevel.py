from scipy import stats
from scipy.io import arff
import pandas as pd
from pyitlib import discrete_random_variable as drv

class singleLevel:

    def IG_SF(self,df, num):

        result = 0.00
        parent_entropy = singleLevel.entropy(self,df)
        for item in df:
            x = item
            y = df[item]
            df.drop([item], axis=1, inplace=True)
            child_entropy = singleLevel.entropy(self,df)
            informationGain = parent_entropy - child_entropy
            result = result + ((num * (num-1))/(2*informationGain))

            df[x] = y

        return result

    def IG(self,parent_entropy,df):
        result = 0.00
        children_entropy = singleLevel.entropy(self,df)
        result =  parent_entropy - children_entropy
        """
        if result>0:
            print("valuable: " + str(result))
        else:
            print("ignorable: " + str(result))
        """
        return result


    def MI(self,df):
        result = 0.00
        listt = df.keys()
        for item in (df):
            j = 1
            for i in listt[j:listt.__len__() - 1]:

                """
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
                """
                X = df[item].values
                Y = df[i].values
                result = result + drv.information_mutual(X, Y)
        return result

    def notselectedMI(self,NS_item,df):
        result = 0.00

        for i in df:
            """
            data1 = NS_item
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
            """
            X = NS_item.values
            Y = df[i].values
            result = result + drv.information_mutual(X, Y)

        return result

    def PCC_SF(self,df):

        result = 0.00
        listt = df.keys()
        for item in (df):
            j = 1
            for i in listt[j:listt.__len__() - 1]:
                X = df[item].values
                Y = df[i].values
                result = result + sum(stats.pearsonr(X, Y))
        return result

    def PCC(self,NS_item,df):
        result = 0.00

        for item in df:

            X = NS_item.values
            Y = df[item].values
            result = result + sum(stats.pearsonr(X, Y))
        return result

    def kendall_SF(self, df):

        result = 0.00
        listt = df.keys()
        for item in (df):
            j = 1
            for i in listt[j:listt.__len__() - 1]:
                X = df[item].values
                Y = df[i].values
                result = result + sum(stats.kendalltau(X, Y))
        return result

    def kendall(self, NS_item, df):
        result = 0.00

        for item in df:
            X = NS_item.values
            Y = df[item].values
            result = result + sum(stats.kendalltau(X, Y))
        return result

    def spearsman_SF(self, df):

        result = 0.00
        listt = df.keys()
        for item in (df):
            j = 1
            for i in listt[j:listt.__len__() - 1]:
                X = df[item].values
                Y = df[i].values
                result = result + sum(stats.spearmanr(X, Y))
        return result

    def spearsman(self, NS_item, df):
        result = 0.00

        for item in df:
            X = NS_item.values
            Y = df[item].values
            result = result + sum(stats.spearmanr(X, Y))
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
