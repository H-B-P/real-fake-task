import pandas as pd
import numpy as np

def get_uniques_for_this_cat_col(inputDf, col, threshold=0):
 uniques = pd.unique(inputDf[col])
 passingUniques = []
 for unique in uniques:
  if (float(len(inputDf[inputDf[col]==unique]))/float(len(inputDf)))>=threshold:
   passingUniques.append(unique)
 return passingUniques

def dummy_this_cat_col(inputDf, col, threshold=0):
 cdf = inputDf.copy()
 uniques = pd.unique(cdf[col])
 usedUniques=[]
 for unique in uniques:
  if (float(len(cdf[cdf[col]==unique]))/float(len(cdf)))>=threshold:
   cdf[col+"_is_"+str(unique)] = (cdf[col]==unique).astype(int)
   usedUniques.append(unique)
 cdf=cdf.drop(col, axis=1)
 return cdf, usedUniques

def dummy_this_cat_col_given_uniques(inputDf, col, uniques):
 cdf = inputDf.copy()
 uniques = pd.unique(cdf[col])
 for unique in uniques:
   cdf[col+"_is_"+str(unique)] = (cdf[col]==unique).astype(int)
 cdf=cdf.drop(col, axis=1)
 return cdf


def get_winsors_for_this_cont_col(inputDf, col, lb=0.01, ub=0.99):
 lower = inputDf[col].quantile(lb)
 upper = inputDf[col].quantile(ub)
 return lower,upper

def apply_winsors_to_this_cont_col(inputDf, col, lower, upper):
 cdf = inputDf.copy()
 cdf.loc[cdf[col] < lower, col]=lower
 cdf.loc[cdf[col] > upper, col]=upper
 return cdf

def get_shift_and_scale_for_this_cont_col(inputDf,col):
 shift = inputDf[col].mean()
 scale = inputDf[col].std(ddof=0)
 return shift, scale

def apply_shift_and_scale_to_this_cont_col(inputDf, col, shift, scale):
 cdf = inputDf.copy()
 cdf[col]=cdf[col]-shift
 cdf[col]=cdf[col]/scale
 return cdf
