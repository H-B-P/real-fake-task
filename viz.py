import pandas as pd
import numpy as np

def convert_value(lowerWinsor, upperWinsor, shift, scale, x):
 if x<lowerWinsor:
  return (lowerWinsor-shift)/scale
 if x>upperWinsor:
  return (upperWinsor-shift)/scale
 return (x-shift)/scale

def get_relativity(x, m, c):
 return m*x+c

def get_cont_pdp_relativities(lW, uW, shift, scale, m, c, df, col):
 Xs=[min(df[col]), lW, uW, max(df[col])]
 Ys=[get_relativity(convert_value(lW,uW,shift,scale,x),m,c) for x in Xs]
 return Xs, Ys

def get_cont_pdp_prevalences(df, col, intervals=10, weightCol=None):
 cdf= df.copy()
 
 if type(intervals)==type([1,2,3]):
  intervs=intervals
 else:
  gap=(max(df[col])-min(df[col]))/intervals
  intervs=list(np.arange(min(df[col]), max(df[col])+gap, gap))
 
 if weightCol==None:
  cdf["weight"]=1
 else:
  cdf["weight"]=cdf[weightCol]
 
 prevs=[]
 
 for i in range(len(intervs)-1):
  loInt = intervs[i]
  hiInt = intervs[i+1]
  if i==(len(intervs)-2):
   prevs.append(sum(cdf[(cdf[col]<=hiInt) & (cdf[col]>=loInt)]["weight"]))
  else:
   prevs.append(sum(cdf[(cdf[col]<hiInt) & (cdf[col]>=loInt)]["weight"]))
 
 return intervs, prevs


def get_Xiles(df, predCol, actCol, X=10):
 cdf = df.copy()
 cdf = cdf.sort_values([predCol, actCol])
 cdf = cdf.reset_index()
 preds = []
 acts = []
 for i in range(X):
  lowerlim = int(((i)*len(cdf))/X)
  upperlim = int(((i+1)*len(cdf))/X)
  subset = cdf[lowerlim:upperlim]
  preds.append(sum(subset[predCol])/len(subset))
  acts.append(sum(subset[actCol])/len(subset))
 return preds, acts
