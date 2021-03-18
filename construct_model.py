import pandas as pd
import numpy as np
import calculus
import enforce_constraints
#EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2}, "cont2":{"m":-1, "c":4}}

#EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2,"z":0.35}, "cont2":{"m":-1, "c":4, "z":0.43}}

#EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2,"z":0.35, "segs":[[0.5,0.01],[0.7,0.02]]}, "cont2":{"m":-1, "c":4, "z":0.43}, "cat1":{"wstfgl":1.05, "forpalorp":0.92}}

EXAMPLE_MODEL={"BIG_C":1700,"conts":{"cont1":{"m":1,"c":2,"z":0.35, "segs":[[0.5,0.01],[0.7,0.02]]}, "cont2":{"m":-1, "c":4, "z":0.43}}, "cats":{"cat1":{"uniques":{"wstfgl":1.05, "forpalorp":0.92}, "OTHER":1.04}}}


def predict(inputDf, model):
 preds = pd.Series([model["BIG_C"]]*len(inputDf))
 for col in model["conts"]:
  effectOfCol = get_effect_of_this_cont_col(inputDf, model, col)
  preds = preds*effectOfCol
 for col in model["cats"]:
  effectOfCol = get_effect_of_this_cat_col(inputDf, model, col)
  preds = preds*effectOfCol
 return preds

def get_effect_of_this_cont_col(inputDf, model, col):
 effectOfCol = model["conts"][col]["m"]*(inputDf[col]-model["conts"][col]["z"])+model["conts"][col]["c"]
 if "segs" in model["conts"][col]:
  for seg in model["conts"][col]["segs"]:
   effectOfCol = effectOfCol+abs(inputDf[col]-seg[0])*seg[1]
 return effectOfCol

def get_effect_of_this_cat_col(inputDf, model, col):
 effectOfCol = pd.Series([model["cats"][col]["OTHER"]]*len(inputDf))
 for unique in model["cats"][col]["uniques"]:
  effectOfCol[inputDf[col]==unique] = model["cats"][col]["uniques"][unique]
 return effectOfCol

def explain(model):
 print(model["BIG_C"])
 for col in model["conts"]:
  print(col, model["conts"][col])
 for col in model["cats"]:
  print(col, model["cats"][col])
 print("-")

def construct_model(inputDf, conts, segs, cats, uniques, target, nrounds=100, lr=0.1, grad=calculus.Gamma_grad):
 
 #Prep the model
 model={"BIG_C":inputDf[target].mean(), "conts":{}, "cats":{}}
 
 for col in conts:
  model["conts"][col]={}
  model["conts"][col]["c"]=1
  model["conts"][col]["m"]=0
  model["conts"][col]["z"]=inputDf[col].mean()
  model["conts"][col]["segs"]=[]
  for seg in segs[col]:
   model["conts"][col]["segs"].append([seg,0])
 
 for col in cats:
  model["cats"][col]={"OTHER":1}
  model["cats"][col]["uniques"]={}
  for unique in uniques[col]:
   model["cats"][col]["uniques"][unique]=1
 
 for i in range(nrounds):
  print("epoch: "+str(i)+"/"+str(nrounds))
  explain(model)
  preds = predict(inputDf, model)
  grads = grad(np.array(preds), np.array(inputDf[target]))
  for col in conts:
   xEffect = get_effect_of_this_cont_col(inputDf, model, col)
   model["conts"][col]["c"]-=sum(grads*preds/xEffect)*lr/len(inputDf)
   model["conts"][col]["m"]-=sum(grads*preds*((inputDf[col]-model["conts"][col]["z"])/inputDf[col].std(ddof=0))/xEffect)*lr/len(inputDf)
   if "segs" in model["conts"][col]:
    for seg in model["conts"][col]["segs"]:
     seg[1]-=sum(grads*preds*(abs(inputDf[col]-seg[0])/inputDf[col].std(ddof=0))/xEffect)*lr/len(inputDf)
  for col in cats:
   xEffect = get_effect_of_this_cat_col(inputDf, model, col)
   model["cats"][col]["OTHER"]-=sum((grads*preds/xEffect)[~inputDf[col].isin(model["cats"][col]["uniques"].keys())])*lr/len(inputDf)
   for unique in model["cats"][col]["uniques"]:
    model["cats"][col]["uniques"][unique]-=sum((grads*preds/xEffect)[inputDf[col]==unique])*lr/len(inputDf)
  #model=enforce_constraints.enforce_all_constraints(inputDf, model)
 
 return model


