import numpy as np
import pandas as pd

import prep
import seg
import construct_model
import analytics
import viz

SEGS_PER_CONT=4

#==Load in==

df = pd.read_csv("../data/train.csv")

df = df[[c for c in df.columns if "cont" in c]+["cat1","cat2","cat3"]+["loss"]] #simplify for testing

catCols  = [c for c in df.columns if "cat" in c]
contCols = [c for c in df.columns if "cont" in c]
targetCols = ["loss"]
uselessCols=["id"]


#==Random Split==

trainDf = df.sample(frac = 0.8, random_state=1) 
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

#we need winsors dead

lowerWinsors={}
upperWinsors={}
segPoints={}
shifts={}
scales={}

for col in contCols:
 print(col)
 
 #Winsors
 
 lower, upper = prep.get_winsors_for_this_cont_col(trainDf, col, lb=0, ub=1)
 lowerWinsors[col]=lower
 upperWinsors[col]=upper
 
 trainDf = prep.apply_winsors_to_this_cont_col(trainDf, col, lower, upper)
 
 #Segmentation
 
 ratioList = seg.get_ratios(SEGS_PER_CONT)
 segPointList = []
 for ratio in ratioList:
  segpt = seg.get_segpt(trainDf, col, ratio)
  roundedSegpt = seg.round_to_sf(segpt, 3)
  segPointList.append(roundedSegpt)
 segPoints[col]=segPointList
 
 #Shift, Scale
 
 shift, scale = prep.get_shift_and_scale_for_this_cont_col(trainDf, col)
 shifts[col]=shift
 scales[col]=scale
 
 trainDf = prep.apply_shift_and_scale_to_this_cont_col(trainDf, col, shift, scale)

 print(segPoints[col])
 print(lowerWinsors[col])
 print(upperWinsors[col])
 print(sum(trainDfOri[col])/len(trainDfOri))

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
print("DECILES")
print(p)
print(a)

#Analyze (i.e. get summary stats)

print("MAE")
print(analytics.get_mae(testDf["PREDICTED"],testDf["loss"]))
print("RMSE")
print(analytics.get_rmse(testDf["PREDICTED"],testDf["loss"]))
print("MEANS")
print(analytics.get_means(testDf["PREDICTED"],testDf["loss"]))
print("DRIFT COEFF")
#print(analytics.get_drift_coeff(testDf["PREDICTED"],testDf["loss"]))
print(analytics.get_drift_coeff_macro(p,a))
