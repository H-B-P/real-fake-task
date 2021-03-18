import pandas as pd
import numpy as np
import calculus
import enforce_constraints
#EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2}, "cont2":{"m":-1, "c":4}}

#EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2,"z":0.35}, "cont2":{"m":-1, "c":4, "z":0.43}}

EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2,"z":0.35, "segs":[[0.5,0.01],[0.7,0.02]]}, "cont2":{"m":-1, "c":4, "z":0.43}, "cat1":{"wstfgl":1.05, "forpalorp":0.92}}

def predict(inputDf, model):
 preds = pd.Series([model["BIG_C"]]*len(inputDf))
 for col in model:
  if col!="BIG_C":
   effectOfCol = model[col]["m"]*(inputDf[col]-model[col]["z"])+model[col]["c"]
   if "segs" in model[col]:
    for seg in model[col]["segs"]:
     effectOfCol = effectOfCol+abs(inputDf[col]-seg[0])*seg[1]
   preds = preds*effectOfCol
   
 return preds

def explain(model, explanatories):
 for x in explanatories:
  print(x, model[x])
 print("-")

def construct_model(inputDf, segs, explanatories, target, nrounds=100, lr=0.1, grad=calculus.Gamma_grad):
 
 #Prep the model
 model={"BIG_C":inputDf[target].mean()}
 
 for x in explanatories:
  model[x]={}
  model[x]["c"]=1
  model[x]["m"]=0
  model[x]["z"]=inputDf[x].mean()
  model[x]["segs"]=[]
  for seg in segs[x]:
   model[x]["segs"].append([seg,0])
 
 for i in range(nrounds):
  print("epoch: "+str(i)+"/"+str(nrounds))
  explain(model, explanatories)
  preds = predict(inputDf, model)
  grads = grad(np.array(preds), np.array(inputDf[target]))
  for x in explanatories:
   xEffect = model[x]["m"]*(inputDf[x]-model[x]["z"])+model[x]["c"]
   if "segs" in model[x]:
    for seg in model[x]["segs"]:
     xEffect = xEffect+abs(inputDf[x]-seg[0])*seg[1]
   model[x]["c"]-=sum(grads*preds/xEffect)*lr/len(inputDf)
   model[x]["m"]-=sum(grads*preds*(inputDf[x]/inputDf[x].std(ddof=0))/xEffect)*lr/len(inputDf)
   if "segs" in model[x]:
    for seg in model[x]["segs"]:
     seg[1]-=sum(grads*preds*(abs(inputDf[x]-seg[0])/inputDf[x].std(ddof=0))/xEffect)*lr/len(inputDf)
  model=enforce_constraints.enforce_all_constraints(inputDf, model)
 
 return model
  
