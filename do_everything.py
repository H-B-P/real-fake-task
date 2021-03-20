import numpy as np
import pandas as pd

import prep
import seg
import actual_modelling
import analytics
import viz
import util

SEGS_PER_CONT=5

#==Load in==

df = pd.read_csv("../data/train.csv")

df=df[[c for c in df.columns if c!="id"]]
#df = df[["cont3","cont4"]+["cat1","cat2","cat35"]+["loss"]] #simplify for testing
#df = df[[c for c in df.columns if "cont" in c]+["cat"+str(i) for i in range(1,40+1)]+["loss"]] #simplify for testing
#df = df[["cont1", "cont2"]+["cat"+str(i) for i in range(1,40+1)]+["loss"]] #simplify for testing
#df = df[[c for c in df.columns if "cont" in c]+["loss"]] #simplify for testing

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

print("PREPARING")

#Categoricals

uniques={}

if True:
 for c in catCols:
  print(c)
  uniques[c] = prep.get_uniques_for_this_cat_col(trainDf,c, 0.05)

#Segmentation

segPoints={}

for col in contCols:
 print(col)
 
 ratioList = seg.get_ratios(SEGS_PER_CONT)
 segPointList = []
 for ratio in ratioList:
  segpt = seg.get_segpt(trainDf, col, ratio)
  roundedSegpt = util.round_to_sf(segpt, 3)
  segPointList.append(roundedSegpt)
 segPoints[col]=segPointList

#==Actually model!===

model = actual_modelling.prep_starting_model(trainDf, contCols, segPoints, catCols, uniques, "loss")
model = actual_modelling.construct_model(trainDf, "loss", 10, 0.05, {"uniques":0.03, "segs":0.02, "grads":0, "contfeat":0.04, "catfeat":0.04}, model)#quickly winnow out the really useless ones
model = actual_modelling.de_feat(model)
model = actual_modelling.construct_model(trainDf, "loss", 40, 0.05, {"uniques":0.03, "segs":0.02, "grads":0, "contfeat":0.04, "catfeat":0.04}, model)
model = actual_modelling.de_feat(model)
model = actual_modelling.construct_model(trainDf, "loss", 350, 0.1, {"uniques":0, "segs":0, "grads":0, "contfeat":0, "catfeat":0}, model)

#==Viz Model==

for col in contCols:
 print(col)
 intervs, prevs = viz.get_cont_pdp_prevalences(trainDf, col)
 print([util.round_to_sf(x) for x in intervs])
 print([util.round_to_sf(x) for x in prevs])

#==Predict==

testDf["PREDICTED"]=actual_modelling.predict(testDf, model)

print(testDf[["loss","PREDICTED"]])

#==Viz Predictions==

p, a = viz.get_Xiles(testDf, "PREDICTED", "loss", 10)
print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])

#==Analyze (i.e. get summary stats)==

print("MAE")
print(util.round_to_sf(analytics.get_mae(testDf["PREDICTED"],testDf["loss"])))
print("RMSE")
print(util.round_to_sf(analytics.get_rmse(testDf["PREDICTED"],testDf["loss"])))
print("MEANS")
print(util.round_to_sf(testDf["PREDICTED"].mean()), util.round_to_sf(testDf["loss"].mean()))
print("DRIFT COEFF")
#print(analytics.get_drift_coeff(testDf["PREDICTED"],testDf["loss"]))
print(util.round_to_sf(analytics.get_drift_coeff_macro(p,a)))
