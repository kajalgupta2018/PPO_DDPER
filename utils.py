import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import pandas as pd
import datetime as dt

class Logger:

    def __init__(self, cols, logger_name,file_name="ppo_ddper", mode=0):

       self.logger_name = logger_name 
       if mode ==0: #new file 
        
           self.log_df = pd.DataFrame(columns=cols)
           #print(self.log_df)
           self.log_df['logger_name'] = logger_name
           self.file_name = logger_name+".csv"
           print("logger file name :", self.file_name)
       else:
           self.log_df = pd.read_csv(file_name, encoding='ISO-8859-1')
           self.file_name = file_name
   
       self.log_df['date'] = dt.datetime.today().strftime("%m/%d/%Y")

    #print_msg print

    def add(self, col_name, value):
        #print("column name :", col_name)

        self.log_df[col_name] = value

    def append(self, df):
        #print("append:", df)
        self.log_df = self.log_df.append(df, ignore_index = True)

    def write_to(self):   
        self.log_df['date'] = self.log_df['date'].fillna(dt.datetime.today().strftime("%m/%d/%Y"))
        self.log_df['logger_name'] = self.log_df['logger_name'].fillna(self.logger_name)
        self.log_df.to_csv(self.file_name)
        print("saving log file ....", self.file_name)

    #def print_msg(self, message, *args):
    #    print(message% args)

    def print_log(self, col_name):
    
        if col_name== "":
            print(self.log_df.head())
        else:
            print(self.log_df[col_name])


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)




