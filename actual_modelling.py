import pandas as pd
import numpy as np
import math


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

def round_to_sf(x, sf=5):
 if x==0:
  return 0
 else:
  return round(x,sf-1-int(math.floor(math.log10(abs(x)))))

def roundify_dict(dyct, sf=5):
 opdyct=dyct.copy()
 for k in opdyct:
  if k=="segs":
   for seg in opdyct[k]:
    seg[1]=round_to_sf(seg[1])
  elif k=="uniques":
   for unique in opdyct[k]:
    opdyct[k][unique] = round_to_sf(opdyct[k][unique])
  else:
   opdyct[k]=round_to_sf(opdyct[k])
 return opdyct

def explain(model, sf=5):
 print("BIG_C", round_to_sf(model["BIG_C"]))
 for col in model["conts"]:
  print(col, roundify_dict(model["conts"][col]))
 for col in model["cats"]:
  print(col, roundify_dict(model["cats"][col]))
 print("-")

def update_using_pena(amt, push, pena, multiplier=1, zeroVal=0, zeroCheck=1, i=0):
 if amt==zeroVal:
  if (i%zeroCheck)==0:
   if abs(pena)>abs(push):
    return zeroVal
   else:
    if push>zeroVal:
     return amt+(push-pena)*multiplier
    else:
     return amt+(push+pena)*multiplier
 else:
  if amt>zeroVal:
   wouldBeOutPut = amt+(push-pena)*multiplier
  else:
   wouldBeOutPut = amt+(push+pena)*multiplier
  if ((wouldBeOutPut-zeroVal)*(amt-zeroVal))<0: #i.e. if amt and wouldBeOutPut have different signs, i.e. if moving from former to latter would take you past a zero
   return zeroVal #"don't bother" is sticky!
  else:
   return wouldBeOutPut  

def update_using_pena_incl_feature(amt, push, pena, numerator, denominator, featlim, multiplier=1, zeroVal=0, zeroCheck=1, i=0):
 if denominator>0:
  return update_using_pena(amt, push, pena+numerator/denominator, multiplier, zeroVal, zeroCheck, i)
 else:
  return update_using_pena(amt, push, pena+featlim, multiplier, zeroVal, zeroCheck, i)

def construct_model(inputDf, conts, segs, cats, uniques, target, nrounds=100, lr=0.1, pena={"uniques":0.1, "grads":0.01, "segs":0.01, "overalls":0.01, "features":0.01}, grad=calculus.Gamma_grad, startingModel=None):
 
 #Prep the model if we aren't given one to start
 
 if startingModel==None:
 
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
 
 else:
  
  model=startingModel.copy()
 
 
 
 for i in range(nrounds):
  
  print("epoch: "+str(i)+"/"+str(nrounds))
  explain(model)
  
  preds = predict(inputDf, model)
  grads = grad(np.array(preds), np.array(inputDf[target]))
  
  for col in conts:
   effectOfCol = get_effect_of_this_cont_col(inputDf, model, col)
   
   gpoe = grads*preds/effectOfCol #these terms keep appearing together
   
   #You know what never mind this
   #specZeroValC = model["conts"][col]["c"]+(1-sum(effectOfCol)/len(effectOfCol))# What do we need to add to the current average effect to get the output we want (i.e. 1)?
   #model["conts"][col]["c"] = update_using_pena(model["conts"][col]["c"], -sum(gpoe)*1*mult, pena["overalls"]*len(inputDf), mult, specZeroValC)
   #if we end up doing this sort of thing after all update twice in a row
   
   featurePenaDenominator = model["conts"][col]["m"]**2 + (model["conts"][col]["c"]-1)**2
   for seg in model["conts"][col]["segs"]:
    featurePenaDenominator += seg[1]**2
   featurePenaDenominator = math.sqrt(featurePenaDenominator)
   
   model["conts"][col]["c"] = update_using_pena_incl_feature(model["conts"][col]["c"], -sum(gpoe)*1/len(inputDf), 0, abs(model["conts"][col]["c"]-1)*pena["features"], featurePenaDenominator, pena["features"], lr, 1)
   model["conts"][col]["m"] = update_using_pena_incl_feature(model["conts"][col]["m"], -sum(gpoe*((inputDf[col]-model["conts"][col]["z"])/inputDf[col].std(ddof=0)))/len(inputDf), pena["grads"], abs(model["conts"][col]["m"])*pena["features"], featurePenaDenominator, pena["features"],  lr, 0)
   if "segs" in model["conts"][col]:
    for seg in model["conts"][col]["segs"]:
     seg[1] = update_using_pena_incl_feature(seg[1], -sum(gpoe*(abs(inputDf[col]-seg[0])/inputDf[col].std(ddof=0)))/len(inputDf), pena["segs"], abs(seg[1])*pena["features"], featurePenaDenominator, pena["features"], lr, 0)
  
  for col in cats:
   effectOfCol = get_effect_of_this_cat_col(inputDf, model, col)
   
   gpoe = grads*preds/effectOfCol
   mult = lr/len(inputDf)
   
   #We could probably do a pure-overall approach, but hybridizing just seems like the wrong move. Check overall feat effects at end; if no weirdness, don't bother solving nonproblems.
   #specZeroValO = model["cats"][col]["OTHER"]+(1-sum(effectOfCol)/len(effectOfCol))
   model["cats"][col]["OTHER"]=update_using_pena_incl_feature(model["cats"][col]["OTHER"], -sum((gpoe)[~inputDf[col].isin(model["cats"][col]["uniques"].keys())])/len(inputDf), pena["uniques"], abs(model["cats"][col]["OTHER"])*pena["features"], featurePenaDenominator, pena["features"], lr,1)
   for unique in model["cats"][col]["uniques"]:
    model["cats"][col]["uniques"][unique] = update_using_pena_incl_feature(model["cats"][col]["uniques"][unique], -sum((gpoe)[inputDf[col]==unique])/len(inputDf), pena["uniques"], abs(model["cats"][col]["uniques"][unique])*pena["features"], featurePenaDenominator, pena["features"], lr,1)
  #model=enforce_constraints.enforce_all_constraints(inputDf, model)
 
 return model


