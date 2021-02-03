from scipy import stats
from scipy.io import arff
import pandas as pd
from tqdm import tqdm

from singleLevel import *
class combination:

    def MIandSD(self,df):
        # formula1 - need to be maximized

        return (29*28)/(2*singleLevel.MI(self, df))

    def avgnormalMI(self,NS_item,key_df):
        # formula 2
        return singleLevel.notselectedMI(self,NS_item,key_df)
    #return singleLevel.MI(self,self.df)/notselected features = 1

    def avdSD(self,df):
        # formula 3
        return sum(df.std(axis=0) / 29)

    def entropyandSD(self,x):
        # formula 4
        return singleLevel.entropy(self,x)/29

    def IG1(self,df):
        return singleLevel.IG_SF(self,df)

    def IG2(self,main_entropy,key_df):
        return singleLevel.IG(self,main_entropy,key_df)/1

    def P1_MIandPCC(self,df):
        return (29*28)/ (2* singleLevel.PCC_SF(self,df))

    def P2_MIandPCC(self,NS_item,df):
        return singleLevel.PCC(self,NS_item,df)
        #return PCC(x,y)/notselected = 1

    def reader(self):
        data = arff.loadarff('Training Dataset original.arff')
        self.df = pd.DataFrame(data[0])
        self.df.head()
        self.df.drop(['Result'],axis=1,inplace=True)
        return self.df

    def csvreader(self):
        self.csvdf = pd.read_csv('training dataset.csv')
        self.csvdf.head()
        self.csvdf.drop(['Result'], axis=1, inplace=True)
        return self.csvdf

    def writer(self, entropy,InformationGain, avgSTD,informationgain_SF, Mutualinformation_SD, Mutualinformation_PCC1,AverageMutualinofrmation,  Mutualinformation_PCC2):
        ds = [entropy, InformationGain,avgSTD,informationgain_SF, Mutualinformation_SD, Mutualinformation_PCC1, AverageMutualinofrmation, Mutualinformation_PCC2]
        d = {}
        for k in entropy.keys():
            d[k] = tuple(d[k] for d in ds)
        print(d)
        df = pd.DataFrame.from_dict(d, orient='index', columns=['entropy','InformationGain', 'avgSTD','informationgain_SF', 'Mutualinformation_SD', 'Mutualinformation_PCC1','AverageMutualinofrmation','Mutualinformation_PCC2'])
        df.to_csv(r'C:\Users\Lion\Documents\GitHub\information-theory\export_dataframe.csv', index=True, header=True)

    def __init__(self):
        self.df = combination.reader(self)
        self.csvdf = combination.csvreader(self)
        #for loop for each feature
        entropy={}
        InformationGain = {}
        avgSTD = {}
        informationgain_SF = {}
        Mutualinformation_SD = {}
        Mutualinformation_PCC1 = {}
        AverageMutualinofrmation = {}
        Mutualinformation_PCC2 = {}

        csvkey = self.csvdf.keys()
        parent_entropy = singleLevel.entropy(self,self.df)

        print("\n \033[1;31;40m Mutualinformation_PCC2 Execution time: ")
        csv_df = self.csvdf
        a = range(csvkey.__len__() - 1)
        for item in tqdm(range(csvkey.__len__())):

            x = self.csvdf.keys()[0]
            y = self.csvdf[x]
            csv_df.drop([csv_df.keys()[0]], axis=1, inplace=True)
            Mutualinformation_PCC2[str(x)] = combination.P2_MIandPCC(self, y, csv_df)
            csv_df[x] = y

        print("\n \033[1;31;40m Entropy Execution time: ")
        key_df = self.df
        for item in tqdm(range (self.df.keys().__len__())):

            a = self.df.keys()[0]
            b = self.df[a]
            key_df.drop([key_df.keys()[0]],axis=1,inplace=True)
            entropy[str(a)] = combination.entropyandSD(self,key_df)
            #InformationGain[str(a)] = combination.IG2(self, parent_entropy,key_df)
            key_df[a] = b

        print("\n \033[1;31;40m InformationGain Execution time: ")
        key_df = self.df
        for item in tqdm(range(self.df.keys().__len__() )):
            a = self.df.keys()[0]
            b = self.df[a]
            key_df.drop([key_df.keys()[0]], axis=1, inplace=True)
            #entropy[str(a)] = combination.entropyandSD(self, key_df)
            InformationGain[str(a)] = combination.IG2(self, parent_entropy, key_df)
            key_df[a] = b

        print("\n \033[1;31;40m avgSTD Execution time: ")
        csv_df = self.csvdf
        for item in tqdm(range(csvkey.__len__())):

            x = self.csvdf.keys()[0]
            y = self.csvdf[x]
            csv_df.drop([self.csvdf.keys()[0]], axis=1, inplace=True)
            avgSTD[str(x)] = combination.avdSD(self, csv_df)
            csv_df[x] = y

        print("\n \033[1;31;40m informationgain_SF Execution time: ")
        csv_df = self.csvdf
        for item in tqdm(range(csvkey.__len__())):

            x = self.csvdf.keys()[0]
            y = self.csvdf[x]
            csv_df.drop([self.csvdf.keys()[0]], axis=1, inplace=True)
            informationgain_SF[str(x)] = combination.IG1(self, csv_df)
            csv_df[x] = y

        print("\n \033[1;31;40m Mutualinformation_SD Execution time: ")
        csv_df = self.csvdf
        for item in tqdm(range(csvkey.__len__())):

            x = self.csvdf.keys()[0]
            y = self.csvdf[x]
            csv_df.drop([self.csvdf.keys()[0]], axis=1, inplace=True)
            Mutualinformation_SD[str(x)] = combination.MIandSD(self, csv_df)
            csv_df[x] = y

        print("\n \033[1;31;40m Mutualinformation_PCC1 Execution time: ")
        csv_df = self.csvdf
        for item in tqdm(range(csvkey.__len__())):

            x = self.csvdf.keys()[0]
            y = self.csvdf[x]
            csv_df.drop([self.csvdf.keys()[0]], axis=1, inplace=True)
            Mutualinformation_PCC1[str(x)] = combination.P1_MIandPCC(self, csv_df)
            csv_df[x] = y

        print("\n \033[1;31;40m AverageMutualinofrmation Execution time: ")
        csv_df = self.csvdf
        for item in tqdm(range(csvkey.__len__())):

            x = self.csvdf.keys()[0]
            y = self.csvdf[x]
            csv_df.drop([self.csvdf.keys()[0]], axis=1, inplace=True)
            AverageMutualinofrmation[str(x)] = combination.avgnormalMI(self, y, csv_df)
            csv_df[x] = y


        combination.writer(self, entropy, InformationGain,avgSTD,informationgain_SF,Mutualinformation_SD, Mutualinformation_PCC1, AverageMutualinofrmation,  Mutualinformation_PCC2)


p = combination()