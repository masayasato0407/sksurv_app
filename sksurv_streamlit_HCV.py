#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st


# In[2]:


#ライブラリ
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.linear_model.coxph import BreslowEstimator
import sksurv
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


# In[4]:


#機械学習モデルの保存
import pickle


# In[5]:


#予測モデル読み込み
#savファイルを入れたフォルダを読み込む
#savファイルはpyと同じフォルダ内に格納
rsf = pickle.load(open("rfmodel20220304.sav", 'rb'))


# In[6]:


#title
st.title('Prediction model for post-SVR HCC') 


# In[9]:


st.markdown("Enter the following items to display the predicted HCC risk")


# In[10]:


#変数の設定
with st.form('user_inputs'): 
  gender = st.selectbox('gender', options=['female', 'male']) 
  Age=st.number_input('age (year)', min_value=0) 
  BMI=st.number_input('Body mass index', min_value=10.0) 
  alc60 = st.selectbox('Daily alcoholic consumption', options=['Less than 60g', '60g or more']) 
  PLT=st.number_input('Platelet count (×10^4/µL)', min_value=0.0)
  AFP=st.number_input('AFP (ng/mL)', min_value=0.0) 
  ALB=st.number_input('Albumin (g/dL)', min_value=0.0) 
  AST=st.number_input('AST (IU/L)', min_value=0)
  ALT=st.number_input('ALT (IU/L)', min_value=0)
  GGT=st.number_input('GGT (IU/L)', min_value=0)
  TBil=st.number_input('Total bilirubin (mg/dL)', min_value=0.0)
  DM= st.selectbox('Diabetes', options=['absent', 'present']) 
  st.form_submit_button() 


# In[11]:


#男性なら1,女なら0
性別 = 0 

if gender == 'male': 
  gender = 1

elif gender == 'female':
  gender = 0 


# In[12]:


#alc60未満0,以上1
alc60 = 0 

if alc60 == '60g or more': 
  alc60 = 1 

elif alc60 == 'Less than 60g': 
  alc60 = 0 


# In[13]:


#DMなし0,あり1
DM = 0 

if DM == 'present': 
  DM = 1 

elif DM == 'absent': 
  DM = 0 


# In[46]:


#dataframe作成
X_test_selall = pd.DataFrame(
    data={'gender': [gender], 
          'Age': [Age],
          'BMI': [BMI],
          'alc60': [alc60],
          'PLT': [PLT],
          'AFP': [AFP],
          'ALB': [ALB],
          'AST': [AST],
          'ALT': [ALT],
          'GGT': [GGT],
          'TBil': [TBil],
          'DM': [DM]
         }
)


# In[48]:


#個別カプラン表示
surv = rsf.predict_survival_function(pd.DataFrame(
    data={'gender': [gender], 
          'Age': [Age],
          'BMI': [BMI],
          'alc60': [alc60],
          'PLT': [PLT],
          'AFP': [AFP],
          'ALB': [ALB],
          'AST': [AST],
          'ALT': [ALT],
          'GGT': [GGT],
          'TBil': [TBil],
          'DM': [DM]
         }
), return_array=True)

for i, s in enumerate(surv):
    plt.step(rsf.event_times_, s, where="post", label=str(i))
plt.xlim(0,5)
plt.ylim(0,1)
plt.ylabel("predicted HCC development")
plt.xlabel("years")
plt.grid(True)
#上下逆にする
plt.gca().invert_yaxis()
#yラベルの変更
plt.yticks([0.0, 0.2, 0.4,0.6,0.8,1.0],
            ['100%', '80%', '60%', '40%', '20%', '0%'])
plt.savefig("img.png")


# In[ ]:


st.subheader("HCC risk for submitted patient")


# In[ ]:


st.image ("img.png")

