import numpy as np
import pandas as pd

import prep
import construct_model
import viz

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

trainDfOri = trainDf.copy()
testDfOri = testDf.copy()

#==Prep the trainDf==

uniques={}

if True:
 for c in catCols:
  print(c)
  trainDf, uus = prep.dummy_this_cat_col(trainDf,c)
  uniques[c] = uus


lowerWinsors={}
upperWinsors={}
segpoints={}
shifts={}
scales={}

for col in contCols:
 print(col)
 lower, upper = prep.get_winsors_for_this_cont_col(trainDf, col, lb=0.01, ub=0.99)
 lowerWinsors[col]=lower
 upperWinsors[col]=upper
 
 trainDf = prep.apply_winsors_to_this_cont_col(trainDf, col, lower, upper)
 
 shift, scale = prep.get_shift_and_scale_for_this_cont_col(trainDf, col)
 shifts[col]=shift
 scales[col]=scale
 
 trainDf = prep.apply_shift_and_scale_to_this_cont_col(trainDf, col, shift, scale)

model = construct_model.construct_model(trainDf, [c for c in trainDf.columns if "cont" in c], "loss", 100)

#Viz Model

for col in contCols:
 print(col)
 relasX, relasY = viz.get_cont_pdp_relativities(lowerWinsors[col], upperWinsors[col], shifts[col], scales[col], model[col]["m"], model[col]["c"], trainDfOri, col)
 intervs, prevs = viz.get_cont_pdp_prevalences(trainDfOri, col)
 print(relasX)
 print(relasY)
 print(intervs)
 print(prevs)

#Predict

for col in catCols:
 testDf = prep.dummy_this_cat_col_given_uniques(testDf, col, uniques[col])

for col in contCols:
 testDf = prep.apply_winsors_to_this_cont_col(testDf, col, lowerWinsors[col], upperWinsors[col])
 testDf = prep.apply_shift_and_scale_to_this_cont_col(testDf, col, shifts[col], scales[col])

testDf["PREDICTED"]=construct_model.predict(testDf, model)

print(testDf[["loss","PREDICTED"]])

#Viz Predictions

p, a = viz.get_Xiles(testDf, "PREDICTED", "loss", 10)

print(p)
print(a)
