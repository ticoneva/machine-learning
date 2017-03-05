"""
License plate study with periodic retraining


"""

from plate_rnn import *

study = plate_study_overtime()
study.args.hid_size = 1024
study.args.embed_dim = 24
study.args.rlayers = 7
study.args.flayers = 1
study.args.drop_rate = 0.05

#pre-train with 5 years of data
for _ in range(29):
	study.run_all(start_num=25990,retrain_mode="Y",win_size=25990)

