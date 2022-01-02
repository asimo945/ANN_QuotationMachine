#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 14:32:58 2021

@author: linli-shang
"""
import pandas as pd
import numpy as np

x_feat=['order quantity', 
                'Customer', 
                'By_way', 
               'Types',
                'SPEC_1st_L1',
                'SPEC_1st_L2', 
                'SPEC_1st_L3',
               'SPEC_1st_N', 
                'SPEC_2nd_L1', 
               'SPEC_2nd_L2',
                'SPEC_2nd_L3',
                'Ap_outer_r',
                'Ap_L',
               'cen_outter_r',
                'cen_L',
                'cen_material',
                'Shell_outer_r', 
               'shell_L',
                'shell_material']
x_num_feat=['order quantity', 
            'SPEC_1st_N',
            'Ap_outer_r', 
            'Ap_L',
            'cen_outter_r',
            'cen_L',
            'Shell_outer_r',
            'shell_L']
x_cat_feat=['Customer',
            'By_way',
            'Types',
            'SPEC_1st_L1',
            'SPEC_1st_L2',
            'SPEC_1st_L3',
            'cen_material',
            'shell_material',
            'SPEC_2nd_L1',
            'SPEC_2nd_L2',
            'SPEC_2nd_L3']

def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif

def temp_to_features(data):
    
    ## Rename Mand_col to Eng_col ##
    rename_col={'數量':'order quantity','規格':'SPEC', '客戶':'Customer', 
            '通路':'By_way', '型態':'Types', '外觀外徑(mm)':'Ap_outer_r',
            '外觀長度(mm)':'Ap_L', '重量(kg)':'AP_w',
            '軸心外徑(mm)':'cen_outter_r', '軸心長度(mm)':'cen_L',
         '軸心材質':'cen_material', '外殼外徑(mm)':'Shell_outer_r', 
         '外殼長度(mm)':'shell_L', '外殼材質':'shell_material'}
    
    data=data.rename(columns=rename_col)
    data['SPEC'] = data['SPEC'].str.split("/", n = 1, expand = True)[0]
    data['SPEC'] = data['SPEC'].str.split("(", n = 1, expand = True)[0].astype('str')    
    ## SPEC_split 
    
    data[['SPEC_1st_L', 'SPEC_1st_N', 'SPEC_2nd_L']] = data['SPEC'].str.extract('([A-Za-z]+)(\d+\.?\d*)([A-Za-z]*)', expand = True)
    data[['SPEC_1st_L1', 'SPEC_1st_L2', 'SPEC_1st_L3']] = data['SPEC_1st_L'].str.extract('([A-Za-z]+)([A-Za-z]+)([A-Za-z]+)', expand = True)
    data[['SPEC_2nd_L1', 'SPEC_2nd_L2', 'SPEC_2nd_L3']] = data['SPEC_2nd_L'].str.extract('([A-Za-z]+)([A-Za-z]+)([A-Za-z]+)', expand = True)
    data['SPEC_1st_N']= pd.to_numeric(data['SPEC_1st_N'],errors='coerce',downcast='float')
    data['Ap_outer_r']= pd.to_numeric(data['Ap_outer_r'],errors='coerce',downcast='float')
    data['cen_outter_r']= pd.to_numeric(data['cen_outter_r'],errors='coerce',downcast='float')
    data['Shell_outer_r']= pd.to_numeric(data['Shell_outer_r'],errors='coerce',downcast='float')

    
    ## only Keep useful features
    data=data[x_feat]
    
    ## make sure the data type is correct ##
    data[x_num_feat]=data[x_num_feat].astype('float32')
    
    data[x_cat_feat]=data[x_cat_feat].astype('object')
    
    
    print ('The null data from each columns are:\n\n', data.isnull().sum(),'\n\n')
    print('The data types of each columns are:\n\n', data.dtypes,'\n\n')
    
    return data



def feature_col_clean_split (dataframe):
    
    from sklearn.preprocessing import MinMaxScaler
    numerical = dataframe.select_dtypes(exclude=['object'])
    numerical.fillna(0,inplace = True)
    numerical.round(4)
    categoric = dataframe.select_dtypes(include=['object'])
    categoric.fillna('NONE',inplace = True)
    dataframe = numerical.merge(categoric, left_index = True, right_index = True)
    
    dummies = pd.get_dummies(categoric[x_cat_feat],drop_first=True)
    
    data = pd.concat([numerical,dummies],axis=1)
    
    return data,numerical,categoric,dummies


def thousands(np_array):
    lis=[float(i) for i in np.round(np_array.astype(float),0)]
    thous=["{:,}".format(i) for i in lis]
    return thous
    
    



    
    