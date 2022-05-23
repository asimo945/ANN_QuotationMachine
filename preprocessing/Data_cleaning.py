import numpy as np
import pandas as pd

path_test =r'Data/Real_test.csv'
df_test=pd.read_csv(path_test)

df_test = df_test.rename(columns={
    '成本分析後單價':'price per unit',
                        '品名':'item_name',
                        '結案碼':'Deal condition',
                        '客戶簡稱':'Customer',
                        '型態別':'Types',
                        '通路別':'By_way'})



df_test['Deal']= 1
cols=['price per unit','item_name', '規格', 'Customer', 'By_way', 'Types','Deal']

df_test=df_test[cols]

path_test_X_pkl= r'Data/Real_test_df_x.pkl'
df_test.to_pickle(path_test_X_pkl)

