"""
Overall setup for studying license plate data
Version: 2017-2-19a
"""

from plate_rnn.models import plate_rnn, plate_parse

import os
from collections import OrderedDict
import math
import time
import csv
import numpy as np
from scipy import stats


class plate_study():
    """    
    Base class for license plate study
    """
    def __init__(self,counter=0,resetdata=False,multiRow=False):
     
        self.resetdata = resetdata
        self.multiRow = multiRow
     
        #Load command line arguments
        self.args = plate_parse()
    
        #Overwrite file?
        if self.args.overwrite or not os.path.isfile(self.args.result_path):
            wMode = 'w'
            self.writeVarnames = True
            with open(self.args.result_path, wMode, buffering=1, newline='') as rfile:
                pass            
        else:
            self.writeVarnames = False            
        
        #Create dictionary
        if self.multiRow:
            self.results = []        
        else:
            self.results = OrderedDict()    
        
        #If not resetting data every run, load data here
        if not self.resetdata:
            self.initialize()
        
    def run(self,counter=0,noTrain=False,noEval=False,resetdata=False):
        
        self.counter = counter

        with open(self.args.result_path, 'a', buffering=1, newline='') as rfile:
            rcsv = csv.writer(rfile)        

            for i in range(self.args.runs):        
                self.counter += 1                                
                if self.counter >= self.args.start_run:    

                    #Record settings                        
                    self.recordSettings()
                    
                    start = time.time()
                    #Reset data if required
                    if self.resetdata or resetdata:
                        self.initialize()
                        
                    #Actual training
                    if not noTrain:
                        self.train()
                    
                    #Evaluate the model
                    if not noEval:
                        self.eval()
                        
                    self.results["Time"] = time.time() - start
                    
                    if self.results is not None:
                        if self.writeVarnames:
                            rcsv.writerow(list(self.results.keys()))
                            self.writeVarnames = False
                        if self.multiRow:
                            for res in self.results:
                                rcsv.writerow(list(self.results.values()))
                        else:
                            rcsv.writerow(list(self.results.values()))
    
        return self.counter    
        
    def initialize(self):
        """
        Initialize model
        """
        raise NotImplemented
    
    def train(self):
        """
        Train model
        """
        raise NotImplemented
        
    def eval(self):
        pass
        
    def recordSettings(self):
        self.results["name"] = self.args.save_model + str(self.counter)
        self.results["y_choice"] = self.args.y_choice
        self.results["learn_rate"] = self.args.learn_rate
        self.results["hidsize"] = self.args.hid_size
        self.results["embed_dim"] = self.args.embed_dim
        self.results["rlayers"] = self.args.rlayers
        self.results["flayers"] = self.args.flayers
        self.results["drop_rate"] = self.args.drop_rate
        self.results["rlayer_type"] = self.args.rlayer_type
        self.results["activation"] = self.args.activation
        self.results["epochs"] = self.args.epochs
        self.results["Time"] = 0
    
class plate_study_apm(plate_study):
    """
    Afternoon effect study    
    """
    def initialize(self):
        
        #Morning model
        self.model_morning = plate_rnn(self.args,"./data/carplate_morning_x.csv",
                                                        "./data/carplate_morning_y.csv","morning")
        
        #Afternoon model
        self.model_afternoon = plate_rnn(self.args,"./data/carplate_afternoon_x.csv",
                                                        "./data/carplate_afternoon_y.csv","afternoon")
        
        #Change settings
        #model_morning.args.learn_rate = 0.01
        #model_afternoon.args.learn_rate = 0.01
            
    def train(self):
        #Train
        self.model_morning.train()
        self.model_afternoon.train()
        
    def eval(self):
        
        self.model_morning.eval_model(self.results,"M")
        self.model_afternoon.eval_model(self.results,"A")
        
        #Cross-check
        print("Morning model, afternoon data (Valid):")
        self.model_morning.eval(self.model_afternoon.valid_set,self.model_afternoon.y_valid)
        print("Afternoon model, morning data (Valid):")
        self.model_afternoon.eval(self.model_morning.valid_set,self.model_morning.y_valid)
        print('-'*80)
        
        ytr_morning = np.append(self.model_morning.ytr_bar,
                self.model_morning.model.get_outputs(self.model_afternoon.train_set),
                axis=0)
        ytr_afternoon = np.append(self.model_afternoon.model.get_outputs(self.model_morning.train_set),
                self.model_afternoon.ytr_bar,
                axis=0)
        yv_morning = np.append(self.model_morning.yv_bar,
                self.model_morning.model.get_outputs(self.model_afternoon.valid_set),
                axis=0)
        yv_afternoon = np.append(self.model_afternoon.model.get_outputs(self.model_morning.valid_set),
                self.model_afternoon.yv_bar,
                axis=0)
        yte_morning = np.append(self.model_morning.yte_bar,
                self.model_morning.model.get_outputs(self.model_afternoon.test_set),
                axis=0)
        yte_afternoon = np.append(self.model_afternoon.model.get_outputs(self.model_morning.test_set),
                self.model_afternoon.yte_bar,
                axis=0)        
    
        ytr_ols_morning = self.model_morning.ols.predict(ytr_morning)
        ytr_ols_afternoon = self.model_afternoon.ols.predict(ytr_afternoon)        
        yte_ols_morning = self.model_morning.ols.predict(yte_morning)
        yte_ols_afternoon = self.model_afternoon.ols.predict(yte_afternoon)            
        
        #print("T-test of difference in morning and afternoon model prediction:")
            
        #Train
        run_tests(ytr_morning,ytr_afternoon,self.results,"CNN-tr")
        run_tests(ytr_ols_morning,ytr_ols_afternoon,self.results,"CNN-OLS-tr")
        
        #Test
        run_tests(yte_morning,yte_afternoon,self.results,"CNN-te")
        run_tests(yte_ols_morning,yte_ols_afternoon,self.results,"CNN-OLS-te")
    
    
class plate_study_combined(plate_study):
    """
    Run study on whole day data
    """
    def initialize(self):
        self.model = plate_rnn(self.args,"./data/carplate_x.csv",
                                            "./data/carplate_y.csv","whole_day")
        if self.args.load_model != "":
            self.model.load_model()
        
    def train(self):    
        self.model.train(self.counter)
        
    def eval(self):
        self.model.eval_model(self.results,"C")
    
    def save_all_output(self):
        self.model.save_all_output()
    
   

class plate_study_overtime(plate_study):
    """
    Run study on whole day data, retrained periodically
    """
    
    year_col = 28
    month_col = 29

    def initialize(self):
        self.model = plate_rnn(self.args,"","","overtime")        
        self.model.loaddata("./data/carplate_x.csv","./data/carplate_y.csv")        
        
        self.results["year"] = 0
        self.results["month"] = 0
        self.results["operation"] = ""
        self.results["samples"] = 0
        
    def train(self):        
        self.model.train()
        
    def run_all(self,start_num=100,retrain_mode="",win_size=0):
        
        #Train initial model
        print("Initial data setup...")
        self.model.setupdata(test_size=0,start=0,end=start_num)
        print("Initial run...")
        self.results["year"] = 0
        self.results["month"] = 0        
        self.results["operation"] = "Initial"
        self.results["samples"] = start_num
        self.run()
        
        year_prev = self.model.Zorg[start_num][self.year_col]
        month_prev = self.model.Zorg[start_num][self.month_col]
        i_prev = start_num
        
        for i in range(start_num,len(self.model.Yorg)):

            retrain = False
            
            if (self.model.Zorg[i][self.month_col]!=month_prev 
               or self.model.Zorg[i][self.year_col]!=year_prev):
                #Evaluate model for each month
                samples = i - i_prev
                dup_no = math.ceil(self.args.batch_size/samples)
                print("Y:",year_prev,"M:",month_prev,"samples:",samples,"Duplicate:",dup_no)
                
                #Inference data is duplicated to allow for large batch size
                self.model.setupdata(start=i_prev,end=i,test_size=0,
                                         duplicate=dup_no)
                
                self.results["year"] = year_prev
                self.results["month"] = month_prev
                self.results["operation"] = "Predict"
                self.results["samples"] = samples
                self.run(noTrain=True)

                if retrain_mode=="M":
                    retrain = True
                elif retrain_mode=="Y" and self.model.Zorg[i][self.year_col] != year_prev:
                    retrain = True
                
                month_prev = self.model.Zorg[i][self.month_col]
                year_prev = self.model.Zorg[i][self.year_col]
                i_prev = i
                
                
            #Retrain if necessary
            if retrain:    
                if win_size==0:
                    rtsn = 0
                else:
                    rtsn = i - win_size
                print("Retrain...")    
                self.results["operation"] = "Retrain"
                self.results["samples"] = i - rtsn
                self.model.setupdata(start=rtsn,end=i,test_size=0)
                self.run()
                #self.model.args.epochs += 20
                #self.model.fit()
            
    def eval(self):
        self.model.eval_model(self.results,"R")        
        
        
def run_tests(data1,data2,results,prefix):
    ttest = stats.ttest_rel(data1,data2)
    diff = data1 - data2
    
    results[prefix+" T"] = ttest[0][0]
    results[prefix+" p"] = ttest[1][0]
    results[prefix+" diff"] = np.mean(diff)
    results[prefix+" sd"] = np.std(diff)
    results[prefix+" N"] = diff.shape[0]        
    