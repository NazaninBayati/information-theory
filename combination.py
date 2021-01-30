from scipy import stats
from scipy.io import arff
import pandas as pd
from singleLevel import *
class combination:

    def MIandSD(self,df):
        # formula1 - need to be maximized

        return (39*38)/(2*singleLevel.MI(self, df))

    def avgnormalMI(self,NS_item,key_df):
        # formula 2
        return singleLevel.notselectedMI(self,NS_item,key_df)
    #return singleLevel.MI(self,self.df)/notselected features = 1
    """
    def avdSD(self,x,N):
        # formula 3
        return math.sd(x) / N-1
    """
    def entropyandSD(self,x):
        # formula 4
        return singleLevel.entropy(self,x)/39
    """
    def IG1(self,x,y,N):
        return N*(N-1)/2*IG(x,y)
    def IG2(self,x,y,N):
        return IG(x,y)/notselected=1
    def P1_MIandPCC(self,x,y,N):
        return N*(N-1)/2*PCC(x,y)
    def P2_MIandPCC(self,x,y,N):
        return PCC(x,y)/notselected = 1
        """

    def reader(self):
        data = arff.loadarff('Training Dataset original.arff')
        self.df = pd.DataFrame(data[0])
        self.df.head()
        return self.df
    def csvreader(self):
        self.csvdf = pd.read_csv('training dataset.csv')
        self.csvdf.head()
        return self.csvdf

    def __init__(self):
        features = {}
        self.df = combination.reader(self)
        self.csvdf = combination.csvreader(self)
        #for loop for each feature
        key = self.df.keys()
        csvkey = self.csvdf.keys()

        main_entropy = combination.entropyandSD(self,self.df)
        for item in tqdm(range (self.df.keys().__len__()-1)):
            key_df = self.df
            a = self.df.keys()[item]
            b = self.df[a]
            key_df.drop([self.df.keys()[item]],axis=1,inplace=True)
            entropy = combination.entropyandSD(self,key_df)
            key_df[a] = b


        for item in tqdm(range(csvkey.__len__()-1)):
            csv_df = self.csvdf
            x = self.csvdf.keys()[item]
            y = self.csvdf[x]
            csv_df.drop([self.csvdf.keys()[item]], axis=1, inplace=True)
            Mutualinformation_SD = combination.MIandSD(self, csv_df)
            AverageMutualinofrmation = combination.avgnormalMI(self, y, csv_df)
            print(Mutualinformation_SD)
            csv_df[x] = y


p = combination()