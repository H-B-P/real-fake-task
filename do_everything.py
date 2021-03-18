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

df = df[["cont3","cont4"]+["cat1","cat2","cat3"]+["loss"]] #simplify for testing

#df = df[["cont3","cont4"]+["loss"]] 

catCols  = [c for c in df.columns if "cat" in c]
contCols = [c for c in df.columns if "cont" in c]
targetCols = ["loss"]
uselessCols=["id"]


#==Random Split==

trainDf = df.sample(frac = 0.8, random_state=1) 
testDf = df.drop(trainDf.index)

trainDf = trainDf.reset_index()
testDf = testDf.reset_index()

#==Prep==

#Categoricals

uniques={}

if True:
 for c in catCols:
  print(c)
  uniques[c] = prep.get_uniques_for_this_cat_col(trainDf,c)

#Segmentation

segPoints={}

for col in contCols:
 print(col)
 
 ratioList = seg.get_ratios(SEGS_PER_CONT)
 segPointList = []
 for ratio in ratioList:
  segpt = seg.get_segpt(trainDf, col, ratio)
  roundedSegpt = seg.round_to_sf(segpt, 3)
  segPointList.append(roundedSegpt)
 segPoints[col]=segPointList

#==Actually model!===

model = construct_model.construct_model(trainDf, contCols, segPoints, catCols, uniques, "loss", 100, 0.2)

#==Viz Model==

for col in contCols:
 print(col)
 intervs, prevs = viz.get_cont_pdp_prevalences(trainDf, col)
 print(intervs)
 print(prevs)

#==Predict==

testDf["PREDICTED"]=construct_model.predict(testDf, model)

print(testDf[["loss","PREDICTED"]])

#==Viz Predictions==

p, a = viz.get_Xiles(testDf, "PREDICTED", "loss", 10)
print("DECILES")
print(p)
print(a)

#==Analyze (i.e. get summary stats)==

print("MAE")
print(analytics.get_mae(testDf["PREDICTED"],testDf["loss"]))
print("RMSE")
print(analytics.get_rmse(testDf["PREDICTED"],testDf["loss"]))
print("MEANS")
print(analytics.get_means(testDf["PREDICTED"],testDf["loss"]))
print("DRIFT COEFF")
#print(analytics.get_drift_coeff(testDf["PREDICTED"],testDf["loss"]))
print(analytics.get_drift_coeff_macro(p,a))
