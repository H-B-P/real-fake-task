import pandas as pd
import numpy as np
import calculus
import enforce_constraints
EXAMPLE_MODEL={"BIG_C":1700,"cont1":{"m":1,"c":2}, "cont2":{"m":-1, "c":4}}

def predict(inputDf, model):
 preds = pd.Series([model["BIG_C"]]*len(inputDf))
 for col in model:
  if col!="BIG_C":
   preds = preds*(inputDf[col]*model[col]["m"]+model[col]["c"])
 return preds

def explain(model, explanatories):
 for x in explanatories:
  print(model[x])
 print("-")

def construct_model(inputDf, explanatories, target, nrounds=100, lr=0.1, grad=calculus.Gamma_grad):
 
 #Prep the model
 model={"BIG_C":inputDf[target].mean()}
 
 for x in explanatories:
  model[x]={}
  model[x]["c"]=1
  model[x]["m"]=0
 
 for i in range(nrounds):
  print("epoch: "+str(i)+"/"+str(nrounds))
  #explain(model, explanatories)
  preds = predict(inputDf, model)
  grads = grad(np.array(preds), np.array(inputDf[target]))
  for x in explanatories:
   model[x]["c"]-=sum(grads*preds/(model[x]["m"]*inputDf[x]+model[x]["c"]))*lr/len(inputDf)
   model[x]["m"]-=sum(grads*preds*inputDf[x]/(model[x]["m"]*inputDf[x]+model[x]["c"]))*lr/len(inputDf)
  model=enforce_constraints.enforce_all_constraints(inputDf, model)
 
 return model
  
