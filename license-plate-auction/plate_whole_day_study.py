#!/usr/bin/env python
"""
Train a small recurrent neural net on car plate data with whole day data.
Version 2017-1-17a

Change log:
2016-1-17:  Switched to plate_study class structure 
2016-12-29: Check if result file already exist
2016-10-21: Run multiple times and save results
2016-8-28:  Compute more t-tests
2016-8-24:  Moved the actual computation to the new plate_rnn class
"""

from plate_rnn import *

def wd_study(
		hid_size_list = [64,128,256,512,1024],
		embed_dim_list = [12,24,48,96,128,256],
		rlayers_list = [1,3,5,7,9],
		flayers_list = [1,3],
		drop_rate_list = [0.05,0.1]
		):
			
		study = plate_study_combined()

		counter = 0
		for h in hid_size_list:
			for e in embed_dim_list:
				for r in rlayers_list:
					for f in flayers_list:
						for d in drop_rate_list:

							study.args.hid_size = h
							study.args.embed_dim = e
							study.args.rlayers = r
							study.args.flayers = f
							study.args.drop_rate = d

							counter = study.run()		
							
if __name__ == '__main__':
	wd_study()							