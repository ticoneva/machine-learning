"""
Import the car license plate data and format it for training
Version 2017-1-21a

Change log: 
2017-1-21: Added createNgDummies and renamed createWooDummies
2016-8-16: Now handles word embeddings and train-test split
2016-8-13: Export the remaining independent variables as Z
2016-8-2:  Imports from filename_y.csv and filename_x.csv
"""

import csv_control as cc
import numpy as np
from sklearn import cross_validation
import math
import time
from itertools import *

def loadData(x_file,y_file):

	start = time.time()

	#---------------Load data---------------#
	data_x = cc.loadCSV(x_file)
	data_y = cc.loadCSV(y_file)
	print('-'*80)
	print("X example:")
	print(data_x[0])
	print("y example:")
	print(data_y[0])
	print('-'*80)

	#Split into individual variables. 
	#Note: they are all str, even for numbers
	(letters,numbers,afternoon,ordering,date,year,
	 	month,hsi,cpi) = zip(*data_x)
	(price,price_cat10,price_cat100,unsold,avg_price_d,
		sd_price_d,median_price_d,abv_median_d,
		dbl_median_d,tri_median_d,
		price_cat5_d,price_cat10_d,price_cat20_d,
		p_std_d,p_median_ratio_d,
		y_end
		) = zip(*data_y)
		
	#Has letter part?
	has_letters = [1 if len(e)>0 else 0 for e in letters]
	
	#Replace empty letter part with double space
	#letters = [e if len(e)>0 else "  " for e in letters]
	
	#Length of number part
	num_part_len = [len(e) for e in numbers]

	#Convert to integers
	#Note that number part is not converted
	(price,price_cat10,price_cat100,unsold,median_price_d,
		abv_median_d,dbl_median_d,tri_median_d,
		price_cat5_d,price_cat10_d,price_cat20_d,
		afternoon,ordering,year,month) = [[
		int(e) if len(e)>0 else 0 for e in l] for l in 
		(price,price_cat10,price_cat100,unsold,median_price_d,
		abv_median_d,dbl_median_d,tri_median_d,
		price_cat5_d,price_cat10_d,price_cat20_d,
		afternoon,ordering,year,month)]		
		
	#Normalize values
	#NOTE: Not dropping p=0 would enormously lower R2 in estimation!
	ln_price = [math.log(1+e) for e in price]
	price_max = max(price)
	price_norm = [e/price_max for e in price]
	
	#Convert to float
	(avg_price_d,sd_price_d,p_std_d,p_median_ratio_d,
	hsi,cpi) = [[float(e) for e in l] for l in 
		(avg_price_d,sd_price_d,p_std_d,p_median_ratio_d,
		hsi,cpi)]
	
	"""
	#Normalize values
	hsi_max = max(hsi)
	cpi_max = max(cpi)
	hsi = [e/hsi_max for e in hsi]
	cpi = [e/cpi_max for e in cpi]
	"""

	#Year
	y_min = min(year)
	y_col = max(year) - y_min + 1

	x_year = []
	x_month = []
	#Loop through all observations
	l_row = len(year)
	for i in range(l_row):
		#Year dummy
		cur_row = [0] * y_col
		cur_row[year[i]-y_min] = 1
		x_year.append(cur_row)    

		#Month dummy
		cur_row = [0] * 12
		cur_row[month[i]-1] = 1
		x_month.append(cur_row)
	
	X = [letters,numbers]
	y = [price,price_cat10,price_cat100,unsold,avg_price_d,
		sd_price_d,median_price_d,abv_median_d,
		dbl_median_d,tri_median_d,
		price_cat5_d,price_cat10_d,price_cat20_d,
		p_std_d,p_median_ratio_d,ln_price]
	Z = [afternoon,ordering,has_letters,num_part_len,
		x_year,x_month,year,month,hsi,cpi]		

	"""
	[list(chain.from_iterable([p,[a],[o],xy,xm,[y],[m],[h],[c]]))
			for p,a,o,xy,xm,y,m,h,c 
			in zip(afternoon,ordering,x_year,x_month,year,month,hsi,cpi)]
	"""

	end = time.time()

	print("Time: ",end-start)

	return X,y,Z

"""
Train-Validation-Test split
"""
def tvtSplit(X,y,y_continuous,test_size=0.2,seed=12345):
	X = np.asarray(X)

	if y_continuous:
		y = npColArray(y)
	else:
		y = np.asarray(y)

	#Split data into different sets
	X_tv,X_test,y_tv,y_test = cross_validation.train_test_split(
						X,y,
						test_size=test_size,random_state=seed)
	X_train,X_valid,y_train,y_valid = cross_validation.train_test_split(
						X_tv,y_tv,
						test_size=test_size,random_state=seed)
						
	return X,y,X_train,X_valid,X_test,y_train,y_valid,y_test

"""
Set up feature matrix
"""
def plateEmbeddings(letters,numbers,vocab_list):

	vocab = {x:i for i,x in enumerate(vocab_list)}
	
	#Combine letter and number parts and pad to max plate length
	plates = ['{:6}'.format(l+n) for l,n in zip(letters,numbers)]	
		
	#Change letters and numbers to correpsond vocab num
	x_plates = [[len(vocab) + 1 if t not in vocab else vocab[t]
			for t in p] for p in plates]	
		
	return x_plates
	
"""
Convert each Z columns into appropriate numpy arrays
"""
def procZ(Z):
	Zn = []
	for i in range(0,4):
		Zn.append(npColArray(Z[i]))
	Zn.append(np.asarray(Z[4]))	
	Zn.append(np.asarray(Z[5]))
	for i in range(6,10):
		Zn.append(npColArray(Z[i]))
		
	return Zn

"""
Join all Z columns
"""
def procZ2(Z):
	afternoon,ordering,has_letters,num_part_len,x_year,x_month,year,month,hsi,cpi = Z
	return [list(chain.from_iterable([[a],[o],xy,xm,[y],[m],[math.log(h)],[math.log(c)]]))
			for a,o,xy,xm,y,m,h,c 
			in zip(afternoon,ordering,x_year,x_month,year,month,hsi,cpi)]
	
def npColArray(inList):
	return np.asarray(inList).reshape((len(inList),1))	
	
def tEq(mlist,in_val):
	return [1 if e==in_val else 0 for e in mlist]
	
def tIn(mlist,str,len_limit=-1):
	if len_limit==-1:
		return [1 if str in e else 0 for e in mlist]
	else:
		return [1 if (len(e)==len_limit and str in e) else 0 for e in mlist]
		
def tCount(mlist,in_val):
	return [np.asarray([1 if d==in_val else 0 for d in e]).sum() for e in mlist]

def createNgDummies(letters,numbers):
	"""
	Variables used in Ng and Chong 2010 JEPsych
	"""
	nn = [[int(e) for e in l] for l in numbers]
	k = [e for e in range(0,33)]
	k[1] = [1 if len(e)>0 and e[0]==e[1] else 0 for e in letters]
	k[2] = tEq(letters,"")
	k[3] = tEq(letters,"HK")
	k[4] = tEq(letters,"XX")
	k[5] = tEq(numbers,"911")
	
	n100x = ['100','200','300','400','500','600','700','800','900']	
	n1000x = ['1000','2000','3000','4000','5000','6000','7000','8000','9000']	
	k[6] = [1 if e in n100x else 0 for e in numbers]
	k[7] = [1 if e in n1000x else 0 for e in numbers]	

	k[8] = [1 if ((len(e)==3 and e[0]==e[2] and e[0]!=e[1]) or 
				    (len(e)==4 and e[0]==e[3] and e[1]==e[2] and e[0]!=e[1])) else 0 for e in nn]	#symmetric
	k[9] = [1 if (len(e)==4 and e[0]==e[1] and e[2]==e[3] and e[0]!=e[2]) else 0 for e in nn]	#aabb
	k[10] = [1 if (len(e)==4 and e[0]==e[2] and e[1]==e[3] and e[0!=e[1]]) else 0 for e in nn]	#abab
	k[11] = [1 if (len(e)==4 and e[0]==e[1] and e[0]==e[2] and e[0]!=e[3]) else 0 for e in nn]	#aaab
	k[12] = [1 if (len(e)==4 and e[0]!=e[1] and e[1]==e[2] and e[1]==e[3]) else 0 for e in nn]	#abbb		
	k[13] = [1 if (len(e)==4 and e[0]==e[1] and e[0]==e[3] and e[0]!=e[2]) else 0 for e in nn]	#aaba				
	k[14] = [1 if (len(e)==4 and e[0]==e[2] and e[0]==e[3] and e[0]!=e[1]) else 0 for e in nn]	#abaa				
	k[15] = [1 if (len(e)==3 and e[0]==e[1] and e[0]!=e[2]) else 0 for e in nn] 					#aab
	k[16] = [1 if (len(e)==3 and e[0]!=e[1] and e[1]==e[2]) else 0 for e in nn] 					#abb
	k[17] = [1 if ((len(e)==3 and e[1]==e[0]+1 and e[2]==e[1]+1) or 
					 (len(e)==4 and e[1]==e[0]+1 and e[2]==e[1]+1 and e[3]==e[2]+1)) else 0 for e in nn]	#abcd
	k[18] = [1 if ((len(e)==3 and e[1]==e[0]-1 and e[2]==e[1]-1) or 
					 (len(e)==4 and e[1]==e[0]-1 and e[2]==e[1]-1 and e[3]==e[2]-1)) else 0 for e in nn]	#dcba
	k[19] = [1 if (len(e)==2 and e[0]==e[1]) else 0 for e in nn]										#aa
	k[20] = [1 if (len(e)==3 and e[0]==e[1] and e[1]==e[2]) else 0 for e in nn]						#aaa
	k[21] = [1 if (len(e)==4 and e[0]==e[1] and e[1]==e[2] and e[2]==e[3]) else 0 for e in nn]	#aaaa
	k[22] = tIn(numbers,"13")

	k[23] = tCount(nn,1)
	k[24] = tCount(nn,2)
	k[25] = tCount(nn,3)
	k[26] = tCount(nn,4)
	k[27] = tCount(nn,5)
	k[28] = tCount(nn,6)
	k[29] = tCount(nn,7)
	k[30] = tCount(nn,8)
	k[31] = tCount(nn,9)
	k[32] = tCount(nn,0)
	
	k.pop(0)
	return np.matrix(k).T

def createWooDummies(letters,numbers):
	"""
	Variables used in Woo et al 2008 JEPsych
	"""
	nn = [[int(e) for e in l] for l in numbers]
	k = [e for e in range(0,59)]
	k[1] = tEq(letters,"")
	k[2] = [1 if len(e)>0 and e[0]==e[1] else 0 for e in letters]
	k[3] = tEq(letters,"HK")
	k[4] = tEq(letters,"XX")
	k[5] = [1 if len(e)==1 else 0 for e in nn]
	k[6] = [1 if len(e)==2 else 0 for e in nn]
	k[7] = [1 if len(e)==3 else 0 for e in nn]
	k[8] = [1 if (len(e)==2 and e[1]==0) else 0 for e in nn]
	k[9] = [1 if (len(e)==3 and e[1]==0 and e[2]==0) else 0 for e in nn]
	k[10] = [1 if (len(e)==4 and e[1]==0 and e[2]==0 and e[3]==0) else 0 for e in nn]
	k[11] = [1 if (len(e)==3 and e[1]==e[0]+1 and e[2]==e[1]+1) else 0 for e in nn]
	k[12] = [1 if (len(e)==4 and e[1]==e[0]+1 and e[2]==e[1]+1 and e[3]==e[2]+1) else 0 for e in nn]
	k[13] = [1 if (len(e)==2 and e[0]==e[1]) else 0 for e in nn]
	k[14] = [1 if (len(e)==3 and e[0]==e[1] and e[1]==e[2]) else 0 for e in nn]
	k[15] = [1 if (len(e)==4 and e[0]==e[1] and e[1]==e[2] and e[2]==e[3]) else 0 for e in nn]
	k[16] = [1 if (len(e)==4 and e[0]==e[1] and e[2]==e[3] and e[0]!=e[2]) else 0 for e in nn]			
	k[17] = [1 if (len(e)==4 and e[0]==e[2] and e[1]==e[3] and e[0]!=e[1]) else 0 for e in nn]			
	k[18] = [1 if (len(e)==3 and e[0]==e[2] and e[0]!=e[1]) else 0 for e in nn]		
	k[19] = [1 if (len(e)==4 and e[0]==e[3] and e[1]==e[2] and e[0]!=e[1]) else 0 for e in nn]	
	k[20] = tIn(numbers,"13")
	k[21] = tEq(numbers,"128")
	k[22] = tEq(numbers,"138")
	k[23] = tEq(numbers,"168")
	k[24] = tEq(numbers,"228")
	k[25] = tEq(numbers,"238")
	k[26] = tEq(numbers,"268")
	k[27] = tEq(numbers,"328")
	k[28] = tEq(numbers,"338")
	k[29] = tEq(numbers,"368")
	k[30] = tEq(numbers,"663")
	k[31] = tEq(numbers,"668")
	k[32] = tEq(numbers,"988")
	k[33] = tIn(numbers,"128",4)
	k[34] = tIn(numbers,"138",4)
	k[35] = tIn(numbers,"168",4)
	k[36] = tIn(numbers,"228",4)
	k[37] = tIn(numbers,"238",4)
	k[38] = tIn(numbers,"268",4)
	k[39] = tIn(numbers,"328",4)
	k[40] = tIn(numbers,"338",4)
	k[41] = tIn(numbers,"368",4)
	k[42] = tIn(numbers,"663",4)
	k[43] = tIn(numbers,"668",4)
	k[44] = tIn(numbers,"988",4)
	k[45] = tEq(numbers,"1628")
	k[46] = tEq(numbers,"1668")
	k[47] = tCount(nn,1)
	k[48] = tCount(nn,2)
	k[49] = tCount(nn,3)
	k[50] = tCount(nn,4)
	k[51] = tCount(nn,5)
	k[52] = tCount(nn,6)
	k[53] = tCount(nn,7)
	k[54] = tCount(nn,8)
	k[55] = tCount(nn,9)
	
	porsche = ['911','912','914','918','924','928','930','944','959','968',
			'964','993','996','997','991','986','987','981']
	ferrari = ['335','348','355','360','430','458','488',
			'550','575','599','456','612']		
	mercedes = ['200','220','230','240','250','300','400','500','600']	
	k[56] = [1 if e in porsche else 0 for e in numbers]
	k[57] = [1 if e in ferrari else 0 for e in numbers]
	k[58] = [1 if e in mercedes else 0 for e in numbers]	
	
	k.pop(0)

	return np.matrix(k).T
	