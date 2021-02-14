import numpy as np
import pandas as pd

import prep
import construct_model

df = pd.read_csv("../data/train.csv")

df = df[[c for c in df.columns if "cont" in c]+["cat1","cat2","cat3"]+["loss"]] #simplify for testing

catCols  = [c for c in df.columns if "cat" in c]
contCols = [c for c in df.columns if "cont" in c]
targetCols = ["loss"]
uselessCols=["id"]


#==Random Split==

trainDf = df.sample(frac = 0.8, random_state=0) 
testDf = df.drop(trainDf.index)

trainDf = trainDf.reset_index()
testDf = testDf.reset_index()

#==Prep the trainDf==

if True:
 for c in catCols:
  print(c)
  trainDf = prep.dummy_this_cat_col(trainDf,c)

upperWinsors={}
lowerWinsors={}
segpoints={}
shifts={}
scales={}

for c in contCols:
 print(c)
 lower, upper = prep.get_winsors_for_this_cont_col(trainDf, c, lb=0.01, ub=0.99)
 lowerWinsors[c]=lower
 upperWinsors[c]=upper
 
 trainDf = prep.apply_winsors_to_this_cont_col(trainDf, c, lower, upper)
 
 shift, scale = prep.get_shift_and_scale_for_this_cont_col(trainDf, c)
 shifts[c]=shift
 scales[c]=scale
 
 trainDf = prep.apply_shift_and_scale_to_this_cont_col(trainDf, c, shift, scale)

print(trainDf)

print(construct_model.construct_model(trainDf, [c for c in trainDf.columns if "cont" in c], "loss"))
