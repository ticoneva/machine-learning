"""
RNN model to study license plate data
Version: 2017-2-19a
"""

from neon import NervanaObject
from neon.backends import gen_backend
from neon.data.dataiterator import ArrayIterator
#from neon.data.text_preprocessing import *
from neon.initializers import Uniform, GlorotUniform, Gaussian
from neon.layers import (GeneralizedCost, LSTM, Affine, Dropout, LookupTable,
                        RecurrentSum, RecurrentLast,
                        Recurrent, DeepBiLSTM, DeepBiRNN,GRU)
from neon.models import Model
from neon.optimizers import Adagrad,Adadelta,GradientDescentMomentum,RMSProp,Adam
from neon.transforms import (
                Rectlin, Rectlinclip, Logistic, Tanh, Softmax, Identity, Explin,
                CrossEntropyMulti, MeanSquared, LogLoss,
                Accuracy,Misclassification,TopKMisclassification
                )
from neon.callbacks.callbacks import Callbacks
from neon.util.argparser import NeonArgparser,extract_valid_args

import os
from collections import OrderedDict
import random
import numpy as np
from sklearn.linear_model import LinearRegression

import plate_rnn.loadData as ld
from plate_rnn.csv_control import *

class plate_rnn():
    
    #-----Parameters------#
    rounding = 3
    vocab_list = [" ","0","1","2","3","4","5","6","7","8","9",
        "A","B","C","D","E","F","G","H","I","J","K","L","M",
        "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
    
    # hyperparameters
    gradient_clip_value = 15
    vocab_size = len(vocab_list)
    
    #Visual settings
    hLineWidth = 80
    
    def __init__(self,args,xfile="",yfile="",name="",
                seed=12345,test_size=0.2,sentence_length=6):
        
        self.args = args
        self.myName = name
        self.randName = None
        self.seed = seed
        self.sentence_length = sentence_length
        self.y_choice = int(args.y_choice)
        self.y_cont = True if int(args.y_linear)==1 else False
        
        # setup backend if necessary
        if NervanaObject.be is None:
            self.be = gen_backend(**extract_valid_args(self.args, gen_backend))    
        else:
            self.be = NervanaObject.be        

        if xfile!="" and yfile!="":
            self.loadData(xfile,yfile)
            self.setupData(test_size)

    def loadData(self,xfile,yfile):
        """
        Load data
        """
        
        #Load data
        Xorg,Yorg,Zorg = ld.loadData(xfile,yfile)
        #y
        self.Yorg = Yorg[self.y_choice]
        #Word embeddings for RNN
        self.Xorg = ld.plateEmbeddings(Xorg[0],Xorg[1],self.vocab_list)
        #Dummies from Woo et al (2008)
        self.Dorg = ld.createWooDummies(Xorg[0],Xorg[1])
        #Dummies from Ng and Chong (2010)
        self.DNgorg = ld.createNgDummies(Xorg[0],Xorg[1])
        #Extra variables (afternoon,HSI,CPI,...)
        self.Zorg = ld.procZ2(Zorg)
        
    def setupData(self,test_size=0.2,start=None,end=None,duplicate=1):
        """
        Split data and iterators
        """
        self.test_size = test_size

        #Crop data?
        crop_data = False
        if start is not None and end is None:
            crop_data = True
            end = len(self.y)
        elif start is None and end is not None:
            crop_data = True
            start = 0
        elif start is not None and end is not None:
            crop_data = True

        #Duplicate data if requested
        if duplicate > 1:
            
            self.y = []        
            self.X = []
            self.D = np.empty([0, self.Dorg.shape[1]],dtype=int)
            self.DNg = np.empty([0, self.DNgorg.shape[1]],dtype=int)
            self.Z = []
            
            for _ in range(duplicate):        
                if crop_data:
                    self.y += self.Yorg[start:end]        
                    self.X += self.Xorg[start:end]
                    self.D = np.concatenate((self.D,self.Dorg[start:end]))
                    self.DNg = np.concatenate((self.DNg,self.DNgorg[start:end]))
                    self.Z += self.Zorg[start:end]
                else:
                    self.y += self.Yorg        
                    self.X += self.Xorg
                    self.D = np.concatenate((self.D,self.Dorg))
                    self.DNg = np.concatenate((self.DNg,self.DNgorg))
                    self.Z += self.Zorg
        
        else:
            if crop_data:
                self.y = self.Yorg[start:end]        
                self.X = self.Xorg[start:end]
                self.D = self.Dorg[start:end]
                self.DNg = self.DNgorg[start:end]
                self.Z = self.Zorg[start:end]                    
            else:
                self.y = self.Yorg
                self.X = self.Xorg
                self.D = self.Dorg
                self.DNg = self.DNgorg
                self.Z = self.Zorg
            
        #Split data
        (self.X,self.y,self.X_train,self.X_valid,self.X_test,self.y_train,
            self.y_valid,self.y_test) = ld.tvtSplit(self.X,self.y,self.y_cont,self.test_size,self.seed)

        (self.D,self.y2,self.D_train,self.D_valid,self.D_test,self.y2_train,
            self.y2_valid,self.y2_test) = ld.tvtSplit(self.D,self.y,self.y_cont,self.test_size,self.seed)
        
        (self.DNg,self.y2,self.DNg_train,self.DNg_valid,self.DNg_test,self.y2_train,
            self.y2_valid,self.y2_test) = ld.tvtSplit(self.DNg,self.y,self.y_cont,self.test_size,self.seed)        
        
        (self.Z,self.y2,self.Z_train,self.Z_valid,self.Z_test,self.y2_train,
            self.y2_valid,self.y2_test) = ld.tvtSplit(self.Z,self.y,self.y_cont,self.test_size,self.seed)

        print("X,y:",self.X.shape,self.y.shape)
        
        #Iterators
        if self.y_cont:
            self.costfunc = MeanSquared()
            self.final_activation = Identity()
        
            self.nclass = 1
            
            # set up iterators
            self.whole_set = ArrayIterator(self.X, self.y, make_onehot=False)
            self.train_set = ArrayIterator(self.X_train, self.y_train, make_onehot=False)
            if self.test_size > 0:
                self.valid_set = ArrayIterator(self.X_valid, self.y_valid, make_onehot=False)
                self.test_set = ArrayIterator(self.X_test, self.y_test, make_onehot=False)
            else:
                self.valid_set = ArrayIterator(self.X_train, self.y_train, make_onehot=False)
                self.test_set = ArrayIterator(self.X_train, self.y_train, make_onehot=False)
                
        else:
            self.costfunc = CrossEntropyMulti(usebits=True)    
            self.final_activation = Softmax()
            
            self.in_nodes = len(self.X_train[0])
            self.nclass = max(self.y_train)+1
            
            # set up iterators
            self.whole_set = ArrayIterator(self.X, self.y, nclass=self.nclass, lshape=(self.in_nodes))
            self.train_set = ArrayIterator(self.X_train, self.y_train, nclass=self.nclass, lshape=(self.in_nodes))
            if self.test_size > 0:
                self.valid_set = ArrayIterator(self.X_valid, self.y_valid, nclass=self.nclass, lshape=(self.in_nodes))
                self.test_set = ArrayIterator(self.X_test, self.y_test, nclass=self.nclass, lshape=(self.in_nodes))        
            else:
                self.valid_set = ArrayIterator(self.X_train, self.y_train, nclass=self.nclass, lshape=(self.in_nodes))
                self.test_set = ArrayIterator(self.X_train, self.y_train, nclass=self.nclass, lshape=(self.in_nodes))
                    
    def train(self,counter=0,rename=False):
        
        #Provide a random name for model file
        if self.randName is None or rename:
            self.randName = self.myName + str(random.randint(0,999999))
            self.best_state_path = self.args.best_state_dir+"best_state-"+self.randName+".pkl"

        rlayer_type = self.args.rlayer_type
        activation = self.args.activation
        depth = int(self.args.depth)
        embedding_dim = int(self.args.embed_dim)
        hidden_size = int(self.args.hid_size)
        rlayer_count = int(self.args.rlayers)
        flayer_count = int(self.args.flayers)
        all_drop = int(self.args.all_drop)
        keep_rate = float(1 - self.args.drop_rate)
        reset_cells = self.args.reset_cells
        
        my_optimizer = self.args.optimizer
        learn_rate = float(self.args.learn_rate)
        momentum_rate = float(self.args.momentum_rate)

        # weight initialization
        uni = Uniform(low=-0.1 / embedding_dim, high=0.1 / embedding_dim)
        g_uni = GlorotUniform()
        
        self.layers = [
            LookupTable(vocab_size=self.vocab_size, embedding_dim=embedding_dim, init=uni),
        ]
        
        for i in range(0,rlayer_count):
        
            if activation == 'rectlin':
                act_func = Rectlin()
            elif activation == 'rectlinclip':
                act_func = Rectlinclip()
            elif activation == 'logistic':
                act_func = Logistic()
            elif activation == 'explin':
                act_func = Explin()
            else:
                act_func = Tanh()
        
            if rlayer_type == 'lstm':
                rlayer = LSTM(hidden_size, g_uni, activation=act_func, reset_cells=reset_cells)
            elif rlayer_type == 'gru':
                rlayer = GRU(hidden_size, g_uni, activation=act_func, reset_cells=reset_cells)                                    
            elif rlayer_type == 'rnn':
                rlayer = Recurrent(hidden_size, g_uni, activation=act_func, reset_cells=reset_cells)
            elif rlayer_type == 'bilstm':
                rlayer = DeepBiLSTM(hidden_size, g_uni, activation=act_func, 
                         depth=depth, reset_cells=reset_cells)
            elif rlayer_type == 'birnn':
                rlayer = DeepBiRNN(hidden_size, g_uni, activation=act_func,
                           depth=depth, reset_cells=reset_cells, batch_norm=False)
            elif rlayer_type == 'bibnrnn':
                rlayer = DeepBiRNN(hidden_size, g_uni, activation=act_func,
                           depth=depth, reset_cells=reset_cells, batch_norm=True)
        
            self.layers.append(rlayer)
            self.layers.append(RecurrentSum())
            if all_drop > 0 and keep_rate < 1:
                self.layers.append(Dropout(keep=keep_rate))        
        
        if all_drop == 0 and keep_rate < 1:
            self.layers.append(Dropout(keep=keep_rate))
        
        for i in range(1,flayer_count):
            self.layers.append(Affine(hidden_size, g_uni, bias=g_uni, activation=Rectlin()))
            if all_drop > 0 and keep_rate < 1:
                self.layers.append(Dropout(keep=keep_rate))  
        
        self.layers.append(Affine(self.nclass, g_uni, bias=g_uni, activation=self.final_activation))    
        
        self.model = Model(layers=self.layers)
        
        self.cost = GeneralizedCost(costfunc=self.costfunc)
        
        if my_optimizer == 'gdm':
            self.optimizer = GradientDescentMomentum(learning_rate=learn_rate, 
                    momentum_coef=momentum_rate)
        elif my_optimizer == 'adadelta':
            self.optimizer = Adadelta(decay=0.95,epsilon=1e-6)
        elif my_optimizer == 'adam':
            self.optimizer = Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999)
        elif my_optimizer == 'rmsprop':
            self.optimizer = RMSProp(decay_rate=0.95, learning_rate=learn_rate)            
        else:
            self.optimizer = Adagrad(learning_rate=learn_rate,
                    gradient_clip_value=self.gradient_clip_value)
        
        
        # configure callbacks
        if self.args.callback_args['eval_freq'] is None:
            self.args.callback_args['eval_freq'] = 1
        self.callbacks = Callbacks(self.model, eval_set=self.valid_set, **self.args.callback_args)
                
        #Save best state
        if self.args.best_state:
            if not os.path.isdir(self.args.best_state_dir):
                os.mkdir(self.args.best_state_dir)                
            self.callbacks.add_save_best_state_callback(self.best_state_path)
        
        """
        if self.args.load_model != "":
            self.model.load_params(self.args.load_model)    
        """
    
        if self.args.train:
            # train model
            self.fit(counter)
            
            print('-'*self.hLineWidth)

    def fit(self,counter=0):
        """
        Fit the model and load the best state if requested
        """
        self.model.fit(self.train_set, optimizer=self.optimizer,
                  num_epochs=self.args.epochs, cost=self.cost, callbacks=self.callbacks)            

        if self.args.best_state:
            self.model.load_params(self.best_state_path)    
            os.remove(self.best_state_path)    

        if self.args.save_model != "":
            self.model.save_params(self.args.save_model+str(counter)+".pkl")
            
        if self.args.save_output != "":
            self.save_all_output()
    
    def load_model(self,load_path=""):
        if load_path=="":
            load_path = self.args.load_model
        self.model = Model(load_path+".pkl")
    
    def eval_model(self,results=None,prefix="",printResults=False):
        
        self.y_bar = self.model.get_outputs(self.whole_set)
        self.ytr_bar = self.model.get_outputs(self.train_set)
        self.yv_bar = self.model.get_outputs(self.valid_set)
        self.yte_bar = self.model.get_outputs(self.test_set)
        
        """
        #Round to nearest 1000
        self.ytrb_r = roundp_1k(self.ytr_bar)
        self.yvb_r = roundp_1k(self.yv_bar)
        self.yteb_r = roundp_1k(self.yte_bar)
        self.yb_r = roundp_1k(self.y_bar)
        """
        self.ytrb_r = (self.ytr_bar)
        self.yvb_r = (self.yv_bar)
        self.yteb_r = (self.yte_bar)
        self.yb_r = (self.y_bar)  

      
        # eval model
        if self.y_cont:
            
            if results is None:
                results = OrderedDict()
                
            self.ols = LinearRegression()
            self.ols.fit(self.ytrb_r,self.y_train)                
        
            results[prefix+" Train RMS"] = self.RMS(self.ytrb_r,self.y_train)        
            if self.test_size > 0:
                results[prefix+" Valid RMS"] = self.RMS(self.yvb_r,self.y_valid)
                results[prefix+" Test RMS"] = self.RMS(self.yteb_r,self.y_test)
                results[prefix+" Whole RMS"] = self.RMS(self.yb_r,self.y)
            
            results[prefix+" Train R2"] = self.r_sq(self.ytrb_r,self.y_train)    
            if self.test_size > 0:
                results[prefix+" Valid R2"] = self.r_sq(self.yvb_r,self.y_valid)
                results[prefix+" Test R2"] = self.r_sq(self.yteb_r,self.y_test)
                results[prefix+" Whole R2"] = self.r_sq(self.yb_r,self.y)
            
            #Woo et al (2008) comparison    
            self.ols2 = LinearRegression()
            self.ols2.fit(self.D_train,self.y_train)
            self.ytrb_r_woo = self.ols2.predict(self.D_train)
            if self.test_size > 0:
                self.yvb_r_woo = self.ols2.predict(self.D_valid)
                self.yteb_r_woo = self.ols2.predict(self.D_test)
                self.yb_r_woo = self.ols2.predict(self.D)

            results[prefix+" Woo Train RMS"] = self.RMS(self.ytrb_r_woo,self.y_train)    
            if self.test_size > 0:
                results[prefix+" Woo Valid RMS"] = self.RMS(self.yvb_r_woo,self.y_valid)
                results[prefix+" Woo Test RMS"] = self.RMS(self.yteb_r_woo,self.y_test)
                results[prefix+" Woo Whole RMS"] = self.RMS(self.yb_r_woo,self.y)            
                
            results[prefix+" Woo Train R2"] = self.ols2.score(self.D_train,self.y_train)
            if self.test_size > 0:
                results[prefix+" Woo Valid R2"] = self.ols2.score(self.D_valid,self.y_valid)
                results[prefix+" Woo Test R2"] = self.ols2.score(self.D_test,self.y_test)
                results[prefix+" Woo Whole R2"] = self.ols2.score(self.D,self.y)
                
            #Ng and Chong (2010) comparison    
            self.ols_ng = LinearRegression()
            self.ols_ng.fit(self.DNg_train,self.y_train)
            self.ytrb_r_ng = self.ols_ng.predict(self.DNg_train)
            if self.test_size > 0:
                self.yvb_r_ng = self.ols_ng.predict(self.DNg_valid)
                self.yteb_r_ng = self.ols_ng.predict(self.DNg_test)
                self.yb_r_ng = self.ols_ng.predict(self.DNg)

            results[prefix+" Ng Train RMS"] = self.RMS(self.ytrb_r_ng,self.y_train)        
            if self.test_size > 0:
                results[prefix+" Ng Valid RMS"] = self.RMS(self.yvb_r_ng,self.y_valid)
                results[prefix+" Ng Test RMS"] = self.RMS(self.yteb_r_ng,self.y_test)
                results[prefix+" Ng Whole RMS"] = self.RMS(self.yb_r_ng,self.y)            
                
            results[prefix+" Ng Train R2"] = self.ols_ng.score(self.DNg_train,self.y_train)
            if self.test_size > 0:
                results[prefix+" Ng Valid R2"] = self.ols_ng.score(self.DNg_valid,self.y_valid)
                results[prefix+" Ng Test R2"] = self.ols_ng.score(self.DNg_test,self.y_test)
                results[prefix+" Ng Whole R2"] = self.ols_ng.score(self.DNg,self.y)            
                
            #Combined model
            C_train = np.concatenate((self.ytrb_r,self.ytrb_r_woo),axis=1)
            if self.test_size > 0:
                C_valid = np.concatenate((self.yvb_r,self.yvb_r_woo),axis=1)
                C_test = np.concatenate((self.yteb_r,self.yteb_r_woo),axis=1)
                C = np.concatenate((self.yb_r,self.yb_r_woo),axis=1)
                
            self.ols3 = LinearRegression()
            self.ols3.fit(C_train,self.y_train)    
            self.ytrb_r_c = self.ols3.predict(C_train)
            if self.test_size > 0:
                self.yvb_r_c = self.ols3.predict(C_valid)
                self.yteb_r_c = self.ols3.predict(C_test)
                self.yb_r_c = self.ols3.predict(C)    

            results[prefix+" Combined Train RMS"] = self.RMS(self.ytrb_r_c,self.y_train)    
            if self.test_size > 0:
                results[prefix+" Combined Valid RMS"] = self.RMS(self.yvb_r_c,self.y_valid)
                results[prefix+" Combined Test RMS"] = self.RMS(self.yteb_r_c,self.y_test)
                results[prefix+" Combined Whole RMS"] = self.RMS(self.yb_r_c,self.y)    
                
            results[prefix+" Combined Train R2"] = self.ols3.score(C_train,self.y_train)
            if self.test_size > 0:
                results[prefix+" Combined Valid R2"] = self.ols3.score(C_valid,self.y_valid)
                results[prefix+" Combined Test R2"] = self.ols3.score(C_test,self.y_test)
                results[prefix+" Combined Whole R2"] = self.ols3.score(C,self.y)            
                            
            #Regression with additional variables            
            M_train = np.concatenate(
                (self.ytrb_r,self.ytrb_r_woo,self.Z_train),
                axis=1)
            if self.test_size > 0:
                M_valid = np.concatenate(
                    (self.yvb_r,self.yvb_r_woo,self.Z_valid),
                    axis=1)
                M_test = np.concatenate(
                    (self.yteb_r,self.yteb_r_woo,self.Z_test),
                    axis=1)    
                M = np.concatenate(
                    (self.yb_r,self.yb_r_woo,self.Z),
                    axis=1)         
                    
            self.ols4 = LinearRegression()
            self.ols4.fit(M_train,self.y_train)    
            self.ytrb_r_ce = self.ols4.predict(M_train)
            if self.test_size > 0:
                self.yvb_r_ce = self.ols4.predict(M_valid)
                self.yteb_r_ce = self.ols4.predict(M_test)
                self.yb_r_ce = self.ols4.predict(M)            
                
            results[prefix+" CE Train RMS"] = self.RMS(self.ytrb_r_ce,self.y_train)        
            if self.test_size > 0:
                results[prefix+" CE Valid RMS"] = self.RMS(self.yvb_r_ce,self.y_valid)
                results[prefix+" CE Test RMS"] = self.RMS(self.yteb_r_ce,self.y_test)
                results[prefix+" CE Whole RMS"] = self.RMS(self.yb_r_ce,self.y)            
                
            results[prefix+" CE Train R2"] = self.ols4.score(M_train,self.y_train)
            if self.test_size > 0:
                results[prefix+" CE Valid R2"] = self.ols4.score(M_valid,self.y_valid)
                results[prefix+" CE Test R2"] = self.ols4.score(M_test,self.y_test)
                results[prefix+" CE Whole R2"] = self.ols4.score(M,self.y)            

            if printResults:
                print("Train RMS:",results[prefix+" Train RMS"] )
                if self.test_size > 0:
                    print("Valid RMS:",results[prefix+" Valid RMS"])
                    print("Test RMS:",results[prefix+" Test RMS"])    
                    print("Whole RMS:",results[prefix+" Whole RMS"])
                print("")
            
                print("R-Squared Train:",round(results[prefix+" Train R2"],4))
                if self.test_size > 0:
                    print("R-Squared Valid:",round(results[prefix+" Valid R2"],4))
                    print("R-Squared Test:",round(results[prefix+" Test R2"],4))    
                    
                print('-'*self.hLineWidth)
                print('Woo et al (2008) comparison:')
                print("R-Squared Train:",round(results[prefix+" Woo Train R2"],4))
                if self.test_size > 0:
                    print("R-Squared Valid:",round(results[prefix+" Woo Valid R2"],4))
                    print("R-Squared Test:",round(results[prefix+" Woo Test R2"],4))
                print('-'*self.hLineWidth)
                
                print('Combined (OLS):')
                print("R-Squared Train:",round(results[prefix+" Combined Train R2"],4))
                if self.test_size > 0:
                    print("R-Squared Valid:",round(results[prefix+" Combined Valid R2"],4))
                    print("R-Squared Test:",round(results[prefix+" Combined Test R2"],4))
                print('-'*self.hLineWidth)
                
                print('Combined w/ extra vars (OLS):')
                print("R-Squared Train:",round(results[prefix+" CE Train R2"],4))
                if self.test_size > 0:
                    print("R-Squared Valid:",round(results[prefix+" CE Valid R2"],4))
                    print("R-Squared Test:",round(results[prefix+" CE Test R2"],4))
                print('-'*self.hLineWidth)
                                
        else:
            print("Train Accuracy - {}".format(
                round(100 * self.model.eval(self.train_set, metric=Accuracy())[0],4)))
            if self.test_size > 0:
                print("Test  Accuracy - {}".format(
                    round(100 * self.model.eval(self.valid_set, metric=Accuracy())[0],4)))
            print("Train Top-{} Accuracy - {}".format(self.args.top_k,
                round(100 * (1 - self.model.eval(self.train_set, metric=TopKMisclassification(self.args.top_k))[2]),4)))
            if self.test_size > 0:
                print("Test Top-{} Accuracy - {}".format(self.args.top_k,
                    round(100 * (1 - self.model.eval(self.valid_set, metric=TopKMisclassification(self.args.top_k))[2]),4)))
            print('-'*self.hLineWidth)
    
    def predict(self,X):
        y_out = self.model.get_outputs(X)
        if self.y_cont:
            return self.ols.predict(y_out)
        else:
            return y_out
    
    def save_model(self,model_path=""):
        if model_path == "":
            model_path = self.args.save_model
        self.model.save_params(model_path)
    
    def save_all_output(self,output_path=""):
        if output_path=="":
            output_path = self.args.save_output
        self.savePrediction(self.y,self.y_cont,self.whole_set,output_path)
        self.savePrediction(self.y_train,self.y_cont,self.train_set,output_path+"_train")
        if self.test_size > 0:
            self.savePrediction(self.y_valid,self.y_cont,self.valid_set,output_path+"_valid")
            self.savePrediction(self.y_test,self.y_cont,self.test_set,output_path+"_test")

    def savePrediction(self,y_org,y_cont,X,path):
        """
        Helper function to ouput predictions
        """
        y_out = self.model.get_outputs(X)
        if not y_cont:
            y_org = y_org.reshape((y_org.size,1))
            y_out = np.argmax(y_out,axis=1).reshape((y_org.size,1))
        y_combined = np.concatenate((y_org,y_out),axis=1)
        saveCSV(path+".csv",y_combined.tolist())    
    
    def eval(self,X,y):
        if self.y_cont:
            y_out = self.model.get_outputs(X)
            print("RMS:",self.RMS(y,y_out))
            ols = LinearRegression()
            ols.fit(y_out,y)
            print("R-Squared:",round(ols.score(y_out,y),4))        
        else:
            print("Accuracy - {}".format(
                round(100 * self.model.eval(X, metric=Accuracy())[0],4)))
            print("Top-{} Accuracy - {}".format(self.args.top_k,
                round(100 * (1 - self.model.eval(X, metric=TopKMisclassification(self.args.top_k))[2]),4)))                
    
    #Root Mean-Squared Error
    def RMS(self,y1,y2,roundP=4):
        return round(np.sqrt(np.mean(np.square(np.subtract(y1,y2)))),roundP)        

    def RMS_tensor(self,y1,y2,roundP=4):
        return round(self.be.sqrt(self.be.mean(self.be.square(y1 - y2), axis=0),roundP))
        
    #R-squared
    def r_sq(self,y_hat,y,roundP=4):
        return round(1 - (np.mean(np.square(np.subtract(y,y_hat))) / np.mean(np.square(np.subtract(y,np.mean(y))))),roundP)    
    def r2(self,*args, **kwargs)    :
        return r_sq(*args, **kwargs)

def roundp_1k(y):
    return np.log(np.maximum(1000,np.around(np.exp(y),-3)))


def plate_parse():
    """
    Load command line arguments
    """  
    
    # parse the command line arguments
    parser = NeonArgparser(__doc__)
    
    # Data settings
    parser.add_argument('--y_choice', default=9, type=int,
                help='Choose which y variable to use')            
    parser.add_argument('--y_linear', default=0, type=int,
                help='0 = discrete; 1 = linear')  
                
    #Model settings            
    parser.add_argument('--rlayer_type', default='bibnrnn',
                choices=['bilstm', 'lstm', 'birnn', 'bibnrnn', 'rnn'],
                help='type of recurrent layer to use (lstm, bilstm, rnn, birnn, bibnrnn)')
    parser.add_argument('--activation', default='rectlin',
                choices=['rectlin', 'rectlinclip', 'tanh', 'logistic','explin'],
                help='type of activation the recurrent layers use (rectlin, rectlinclip, tanh, logistic,explin)')
    parser.add_argument('--depth', default=1, type=int,
                help='Depth of bidirection recurrent layers')             
    parser.add_argument('--embed_dim', default=24, type=int,
                help='Word embedding dimension')            
    parser.add_argument('--hid_size', default=256, type=int,
                help='Number of hidden neurons')            
    parser.add_argument('--rlayers', default=5, type=int,
                help='Number of recurrent layers')  
    parser.add_argument('--flayers', default=1, type=int,
                help='Number of fully-connected layers')              
    parser.add_argument('--all_drop', default=1, type=int,
                help='1=Add a dropout layer after each recurrent layer') 
    parser.add_argument('--drop_rate', default=0.1, type=float,
                help='Dropout rate') 
    parser.add_argument('--reset_cells', dest='reset_cells',action='store_true',
                help='Reset cells. Make cells stateless')             
    parser.add_argument('--no_reset_cells', dest='reset_cells',action='store_false',
                help='No reset cells (default)')    
    parser.set_defaults(reset_cells=False)            
    
    #Optimizer settings
    parser.add_argument('--optimizer', default='adagrad',
                choices=['adagrad','adadelta','gdm','adam','rmsprop'],
                help='Optimizer')       
    parser.add_argument('--learn_rate', default=0.01, type=float,
                help='Learning rate')
    parser.add_argument('--momentum_rate', default=0.9, type=float,
                help='Momentum rate')
    parser.add_argument('--best_state', dest='best_state',action='store_true',
                help='Reload best state after training (default)')             
    parser.add_argument('--no_best_state', dest='best_state',action='store_false',
                help='Do not reload best state after training')    
    parser.set_defaults(best_state=True)    
    
    #Training settings
    parser.add_argument('--train', dest='train',action='store_true',
                help='Train model (default)')             
    parser.add_argument('--no_train', dest='train',action='store_false',
                help='Do not train model')    
    parser.set_defaults(train=True)       
    parser.add_argument('--runs', default=1, type=int,
                help='Runs for each set of hyperparameters')
    parser.add_argument('--start_run', default=1, type=int,
                help='The starting run number. Runs below this number will be skipped')                                
        
    #Report and output settings    
    parser.add_argument('--top_k', default=3, type=int,
                help='Top K Misclassification')            
    parser.add_argument('--best_state_dir', default='models/',
                help='Directory to save temporary best states.')            
    parser.add_argument('--load_model', default='',
                help='Filename to load model.')             
    parser.add_argument('--save_model', default='',
                help='Filename to save model.')             
    parser.add_argument('--save_output', default='',
                help='Filename to save output from the model. CSV extension will be added.')  
    parser.add_argument('--result_path',default='results.csv',
                help="Path to save results to.")                            
    parser.add_argument('--overwrite', dest='overwrite',action='store_true',
                help='Overwrite result file')             
    parser.add_argument('--append', dest='overwrite',action='store_false',
                help='Append to result file')    
    parser.set_defaults(overwrite=False)                                
                
    #Backend is not generated because some parameters might need to be changed below
    return parser.parse_args(gen_be=False)         