import numpy as np
import pandas as pd

def convert_lines_to_points(df, col, m, c):
 xlo=min(df[col])
 xhi=max(df[col])
 ylo=m*xlo+c
 yhi=m*xhi+c
 return ylo, yhi

def convert_points_to_lines(df, col, ylo, yhi):
 xlo=min(df[col])
 xhi=max(df[col])
 m=(yhi-ylo)/(xhi-xlo)
 c=ylo-m*xlo
 return m, c

def enforce_all_constraints(df, model):
 cmodel=model.copy()
 cmodel=enforce_minimum_effect(df,cmodel)
 cmodel=enforce_maximum_effect(df,cmodel)
 return cmodel

def enforce_minimum_effect(df, model, minimum=0.1):
 cmodel=model.copy()
 for col in model:
  if col!="BIG_C":
   ylo, yhi = convert_lines_to_points(df, col, model[col]["m"], model[col]["c"])
   if ylo<minimum:
    ylo=minimum
   if yhi<minimum:
    yhi=minimum
   m,c=convert_points_to_lines(df,col,ylo,yhi)
   cmodel[col]["m"]=m
   cmodel[col]["c"]=c
 return cmodel

def enforce_maximum_effect(df, model, maximum=10):
 cmodel=model.copy()
 for col in model:
  if col!="BIG_C":
   ylo, yhi = convert_lines_to_points(df, col, model[col]["m"], model[col]["c"])
   if ylo>maximum:
    ylo=maximum
   if yhi>maximum:
    yhi=maximum
   m,c=convert_points_to_lines(df,col,ylo,yhi)
   cmodel[col]["m"]=m
   cmodel[col]["c"]=c
 return cmodel
