#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

#%run shared_functions.py


# In[2]:


df = pd.read_csv('data/Risky Dealer Case Study/Risky Dealer Case Study Transactions.csv')
df.head()


# In[3]:


df.shape


# In[ ]:





# # Preprocessing
# 1. convert categorical attributes: 'SellingLocation', 'CarMake','JDPowersCat','CarYear','LIGHTG', 'LIGHTY', \
#        'LIGHTR', 'Arbitrated', 'Salvage', 'OVE', 'Simulcast', 'InLane',
#        'PSIEligible'
# 2. selected features: 'Mileage', 'SalePrice', 'MMR', 'PSI', 'SellingLocation', 'CarMake','JDPowersCat','CarYear','LIGHTG', 'LIGHTY', \
#        'LIGHTR', 'Arbitrated', 'Salvage', 'OVE', 'Simulcast', 'InLane',
#        'PSIEligible'
# 3. output feature: 'Returned'
# 4. ignored features: 
#     4.1 Missing values: Autocheck_score	ConditionReport
#     4.2 Uncertain features: DSEligible
#     4.3 unrelated features: 'BuyerID', 'SellerID', 'CarYear', 'SaleDate'

# In[4]:


from sklearn import preprocessing
def convert_categorical(data_df):
    label_encoder = preprocessing.LabelEncoder()
    for c in ['SellingLocation', 'CarMake','JDPowersCat','CarYear','LIGHTG', 'LIGHTY',            'LIGHTR', 'Arbitrated', 'Salvage', 'OVE', 'Simulcast', 'InLane',
           'PSIEligible']:
        data_df[c+'_label']= label_encoder.fit_transform(data_df[c])
    #oh_encoder = preprocessing.OneHotEncoder()
    oh_encoder = preprocessing.OneHotEncoder(categories='auto') 
    np_labels = oh_encoder.fit_transform(data_df[['SellingLocation_label', 'CarMake_label','JDPowersCat_label',                            'CarYear_label','LIGHTG_label', 'LIGHTY_label', 'LIGHTR_label', 'Arbitrated_label',                             'Salvage_label', 'OVE_label', 'Simulcast_label', 'InLane_label',                            'PSIEligible_label']]).toarray()
    data_df = pd.concat([data_df, pd.DataFrame(np_labels)],axis=1)
    return data_df


# In[5]:


# df_clean = pd.concat([df_clean, pd.DataFrame(np_labels)])
# df_clean.shape


# In[6]:


df_clean = convert_categorical(df)
df_clean.shape


# In[7]:


df_clean.columns


# In[ ]:





# In[8]:


output_feature="Returned"

input_features=['Mileage', 'SalePrice', 'MMR', 'PSI'] #, 'Autocheck_score'] #'ConditionReport',]

names = {}
for i in range(332):
    input_features.append(str(i))
    names[i] = str(i)
df_clean = df_clean.rename(columns=names)


# In[9]:


df_clean.shape


# Note that green, yellow, and red lights should not appear in combination. If this occurs it is a data error. In this case default to the highest warning when multiple lights exist.

# In[10]:


df_clean['check-GYR'] = df_clean[['LIGHTG','LIGHTY','LIGHTR']].apply(lambda s:False if s[0]==s[1]==1 or s[0]==s[2]==1 or s[1]==s[2]==1 else True, axis=1)
df_clean = df_clean[df_clean['check-GYR']].drop('check-GYR',axis=1)
df_clean.shape


# Statistics and correlation of the data show that all the attribures are independent.
# 

# In[11]:


df_selected = df_clean.dropna(subset=['Returned','JDPowersCat'])[input_features+[output_feature]]
df_selected = df_selected.fillna(value=0)
df_selected.shape


# In[12]:


df_selected[input_features].dropna().describe()


# In[13]:


df_selected[input_features].corr()


# In[14]:


df_selected.head()


# In[ ]:





# In[ ]:





# In[15]:



df_positive = df_selected[df_selected['Returned']==0].copy()
df_negative = df_selected[df_selected['Returned']==1].copy()


# In[16]:


df_positive.shape, df_negative.shape


# In[17]:


df_positive.columns


# # Sample Selection
# Select 2000 records with Returned=0 as positive samples
# Select 2000 records with Returned=1 as negative samples

# In[18]:


train_df = pd.concat([df_positive[:2000],df_negative[:2000]])
test_df = pd.concat([df_positive[2000:4000],df_negative[2000:]])


# In[19]:


train_df[input_features].dropna().shape, test_df[input_features].dropna().shape


# In[20]:


train_df.head()


# In[21]:


test_df.head()


# # Model Training
# 1. selected machine learning models: 'Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Deep Learning'
# 2. The best model: 'XGBoost'

# In[22]:



def scaleData(train,test,features):
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(train[features])
    train[features]=scaler.transform(train[features])
    test[features]=scaler.transform(test[features])
    
    return (train,test)


# In[23]:


fitted_models_and_predictions_dictionary={}


# In[24]:


def performance_assessment(predictions_df, output_feature='Returned', 
                           prediction_feature='predictions'):
    
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])
    
    performances = pd.DataFrame([[AUC_ROC, AP]], 
                           columns=['AUC ROC','Average precision'])

    performances = performances.round(3)
    
    return performances



def performance_assessment_model_collection(fitted_models_and_predictions_dictionary, 
                                            transactions_df, 
                                            type_set='test'):

    performances=[]
    
    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
    
        predictions_df=transactions_df
            
        predictions_df['predictions']=model_and_predictions['predictions_'+type_set]
        
        performances_model=performance_assessment(predictions_df, output_feature=output_feature,                                                    prediction_feature='predictions')
        performances_model.index=[classifier_name]
        
        #performances=performances.append(performances_model)
        performances.append(performances_model)
    return pd.concat(performances) 


# In[25]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical as one_hot
import yolov4


# In[26]:


#import nvidia_smi
import GPUtil
import psutil


# In[27]:


cpu_usage = psutil.cpu_percent()
mem_usage = psutil.virtual_memory().percent
print(f'cpu_usage: {cpu_usage}, memory_usage: {mem_usage}')

GPUtil.showUtilization()


# In[ ]:





# In[28]:


devices = tf.config.experimental.list_physical_devices('GPU')
for d in devices:
    tf.config.set_logical_device_configuration(
    d,
    [tf.config.LogicalDeviceConfiguration(100),
     tf.config.LogicalDeviceConfiguration(100)])
devices_names = [d.name.split("e:")[1] for d in devices]
mirrored_strategy = tf.distribute.MirroredStrategy(devices=devices_names)


# In[29]:



N_CLASSES = 2

hidden_size = 293

class DeepLearningModel(tf.keras.Model):
    def __init__(self):
        super(DeepLearningModel, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer1 = keras.layers.Dense(
            self.hidden_size,
            activation='relu',
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0., stddev=1.)
        )
        self.hidden_size = self.hidden_size//2
        self.hidden_layer2 = keras.layers.Dense(
            self.hidden_size,
            activation='relu',
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0., stddev=1.)
        )
        
        self.hidden_size = self.hidden_size//2
        self.hidden_layer3 = keras.layers.Dense(
            self.hidden_size,
            activation='relu',
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0., stddev=1.)
        )
        self.prediction_layer = keras.layers.Dense(
            2,
            activation='softmax',
            kernel_initializer=keras.initializers.TruncatedNormal(mean=0., stddev=1.)
        )
        
        
        self.model = keras.Sequential()
        self.model.add(self.hidden_layer1)
        self.model.add(self.hidden_layer2)
        self.model.add(self.hidden_layer3)
        
    def call(self, inputs):
        
        latent = self.model(inputs)
        return self.prediction_layer(latent)

with mirrored_strategy.scope():
    model = DeepLearningModel()
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=0.001),
                  metrics=['accuracy'])
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                                  verbose=1, min_delta=.01)

history = model.fit(train_df[input_features], one_hot(train_df[output_feature]),
                    epochs=100,
                    batch_size=30,
                    validation_data=(test_df[input_features],one_hot(test_df[output_feature])),
                   callbacks = [earlystopping])


# In[30]:



history = model.fit(train_df[input_features], one_hot(train_df[output_feature]),
                    epochs=100,
                    batch_size=30,
                    validation_data=(test_df[input_features],one_hot(test_df[output_feature])),
                   callbacks = [earlystopping])


predictions_test=model.predict(test_df[input_features])
predictions_test = [np.argmax(i) for i in predictions_test]

predictions_train=model.predict(train_df[input_features])
predictions_train = [np.argmax(i) for i in predictions_test]


model_and_predictions_dictionary = {'classifier': 'Deep Learning',
                                    'predictions_test': predictions_test,
                                    'predictions_train': predictions_train
                                   }

fitted_models_and_predictions_dictionary['Deep Learning'] = model_and_predictions_dictionary


# In[31]:


import sklearn
from sklearn import *

import math
import sys
import time
#import graphviz
import xgboost

# For imbalanced learning
import imblearn

import warnings
warnings.filterwarnings('ignore')


IR=1.0/4.0
IR=1.0/1.0

class_weight={0:IR,1:1.0}

classifiers_dictionary={'Logistic regression':sklearn.linear_model.LogisticRegression(random_state=0), 
                        'Decision tree with depth of two': sklearn.tree.DecisionTreeClassifier(max_depth=5,class_weight=class_weight,random_state=0),#sklearn.tree.DecisionTreeClassifier(max_depth=2,random_state=0), 
                        'Decision tree - unlimited depth':sklearn.tree.DecisionTreeClassifier(class_weight=class_weight,random_state=0), #sklearn.tree.DecisionTreeClassifier(random_state=0), 
                        'Random forest':sklearn.ensemble.RandomForestClassifier(class_weight=class_weight,random_state=0,n_jobs=-1),
                        'XGBoost':xgboost.XGBClassifier(random_state=0,n_jobs=-1),
                       }


for classifier_name in classifiers_dictionary:
    
    (train_df, test_df)=scaleData(train_df,test_df,input_features)
    
    classifier = classifiers_dictionary[classifier_name]
    classifier.fit(train_df[input_features], train_df[output_feature])

    predictions_test=classifier.predict_proba(test_df[input_features])[:,1]
    
    predictions_train=classifier.predict_proba(train_df[input_features])[:,1]

    model_and_predictions_dictionary = {'classifier': classifier,
                                        'predictions_test': predictions_test,
                                        'predictions_train': predictions_train
                                       }
    
    fitted_models_and_predictions_dictionary[classifier_name] = model_and_predictions_dictionary


# In[32]:


# performances on test set with all features
df_performances=performance_assessment_model_collection(fitted_models_and_predictions_dictionary, test_df, 
                                                        type_set='test')
df_performances


# In[ ]:





# In[ ]:





# # Model Performance Evaluation
# By comparing the distribution of original data set and the misclassified data set, the mean, mean+std and mean-std become smaller in misclassified data set. In other words, the reason for misclassifying records is because the sample set is biased. More samples with the values less than averages should be added.

# In[33]:


train_df.head()


# In[34]:


test_df.head()


# In[35]:


classifier = classifiers_dictionary['XGBoost']#['Random forest']#
df_selected = df_selected.fillna(value=0)
df_selected['prediction'] = classifier.predict_proba(df_selected[input_features])[:,1]

df_selected.head()


# In[36]:


df_clean_eval = df_selected.copy()
df_clean_eval['check'] = df_clean_eval[['Returned','prediction']].apply(lambda s: True if (s[0]<0.5 and s[1]<0.5) or (s[0]>=0.5 and s[1]>=0.5) else False,axis=1 )
df_clean_eval[df_clean_eval['check']==False].shape


# In[37]:


df_clean_false = df_clean_eval[df_clean_eval['check']==False].copy() #pd.read_csv('data/test_false.csv')
df_clean_false.describe()


# In[38]:


df_clean_eval.describe()


# In[39]:


df_stats = df_clean_eval.describe()
df_stats


# In[40]:


df_stats_false = df_clean_false.describe()
for c in df_stats_false.columns:
    df_stats_false = df_stats_false.rename(columns={c:c+'_false'})
df_stats_false


# In[41]:


df_summary = pd.concat([df_stats_false.T, df_stats.T])
df_summary = df_summary[['mean','std','min','max']]
df_summary['mean-std'] = df_summary[['mean','std']].apply(lambda s: s[0]-s[1],axis=1)
df_summary['mean+std'] = df_summary[['mean','std']].apply(lambda s: s[0]+s[1],axis=1)
df_summary = df_summary[['min','mean-std','mean','mean+std','max']].T
df_summary = df_summary.reset_index()
df_summary = df_summary.rename(columns={'index':'Values'})
df_summary


# In[42]:


from ipywidgets import widgets
import matplotlib.pyplot as plt
import numpy as np

import altair as alt

alt.data_transformers.disable_max_rows()


# In[43]:


size_selector = alt.selection_multi(encodings=['x'])

def summary_plot_alt(data):
    def plotter(column):
        #valid_rows = X[column].notna()
        #plt.plot(X.loc[valid_rows, column], y[valid_rows], '.', color='k')
        #if column.find('_shift')<0:
        scatter = alt.Chart(data, width=500).mark_point().encode(
            x=column,
            y="Values",
            #selected sizes are colored according to the "smoker" column, others are rendered in white
            #color = alt.condition( size_selector , "Time (min)", alt.value("white"))
        )


        scatter_shift = alt.Chart(data, width=500).mark_point().encode(
            x= column+'_false',
            y="Values",
            #selected sizes are colored according to the "smoker" column, others are rendered in white
            #color = alt.condition( size_selector , "Time (min)", alt.value("white"))
        ).add_selection(size_selector)
        
        return scatter & scatter_shift

    return plotter


# For DSEligible, PSIEligibal and other attributes, the stats of misclassified data decrease. Records with smaller values are not trained completely.

# In[44]:


columns = [c for c in df_summary.drop('Values',axis=1).columns if c.find('_false')<0]
#dropdown_values = {"{0}: {1}".format(k, features_dict[k]):k for k in X.columns}
widgets.interact(summary_plot_alt(df_summary), column=columns);


# # Dealer Score Computation
# 1. The attribute DSEligible can be adjusted by using Risky Dealer Score
# 2. Risky Dealer Score can be defined with the attribute Returned. 
#     2.1 Predict the values of the attribute Returned with the trained model.
#     2.2 Using predicted values of the attribute Returned to define new DSEligible values: When the attribute Returned is Yes=1, DSEligible=0, otherwise, DSEligible=1.
#     2.3 Compute Risky Dealer Score: based on new DSEligible values, for each dealer, compute the ratio of count(new DSEligible == DSEligible)/count(records for the dealer) 
#     2.4 if the ratio for a dealer is less than 0.1, the dealer is considered as a risky dealer.
# 3. Conclusion: In 16000 dealers, 845 dealers are considered as risky dealers.

# In[45]:


df_classified = df_clean.dropna(subset=['JDPowersCat'])
classifier = classifiers_dictionary['XGBoost']#['Random forest']#
df_classified = df_classified.fillna(value=0)
df_classified['prediction'] = classifier.predict_proba(df_classified[input_features])[:,1]

df_classified['check'] = df_classified[['DSEligible','prediction']].apply(lambda s: True if (s[0]>0.5 and s[1]<=0.5) or (s[0]<=0.5 and s[1]>0.5) else False, axis=1)
df_classified.head()


# In[ ]:





# In[46]:


score_list = []
df_classified = df_classified[list(df.columns)+['prediction','check']]
for k, v in df_classified.groupby(['SellerID']):
    sellerID = v['SellerID'].values[0]
    pos = v[v['check']==True].shape[0]
    neg = v[v['check']==False].shape[0]
    if pos+neg>0:
        score = pos/(pos+neg)
    else: 
        score = 0
    score_list.append((sellerID,score))


# In[47]:


len(score_list),df_classified['SellerID'].nunique()


# In[48]:


df_seller_score = pd.DataFrame(score_list)
df_seller_score.head()


# In[49]:


df_seller_score[df_seller_score[1]>0.1].shape


# In[50]:


df_seller_score[df_seller_score[1]<=0.1].shape


# In[51]:


df_risky = pd.merge(df_classified, df_seller_score, left_on='SellerID',right_on=0, how='left')
df_risky = df_risky.rename(columns={0:'SellerID0',1:'RiskyDealerScore'})
df_risky = df_risky.drop(['SellerID0','check'],axis=1)
df_risky.head()


# In[52]:


df_risky.to_csv('data/Risky Dealer Case Study Risky Dealer Score.csv',index=False)


# # Seller Profiling and Behavior Analysis
# Most of risky dealers work on
# 1. LightG=No, LightY=No, LightR=Yes,
# 2. Arbitrated=No, Salvage=Yes, OVE=No, Simulcast=No, InLane=No, PSIEligible=No, DSEligible=No
# 3. high/low mileage, low/medium saleprice, low/medium MMR, low/medium autocheck score
# 
# Because Mileage and SalePrice and MMR are related, we want to see how risky dealers act on the three attributes
# Most of risky dealers work on 
# 1. SalePrice=low MMR=low
# 2. SalePrice=high MMR=high

# In[53]:


df_risky = pd.read_csv('data/Risky Dealer Case Study Risky Dealer Score.csv')
df_risky['risky'] = df_risky['RiskyDealerScore'].apply(lambda s: True if s<=0.1 else False)
df_risky['risky_yes'] = df_risky['RiskyDealerScore'].apply(lambda s: 1 if s<=0.1 else 0)
df_risky['risky_no'] = df_risky['RiskyDealerScore'].apply(lambda s: 0 if s<=0.1 else 1)


# In[54]:


columns = ['LIGHTG', 'LIGHTY', 'LIGHTR', 'Arbitrated', 'Salvage', 'OVE', 'Simulcast', 'InLane','PSIEligible','DSEligible']
index = 0
fig, axs = plt.subplots(2, 5,figsize=(24, 12))
fontsize = 8

for ax in axs.flat:
    column = columns[index]
    df = df_risky.dropna(subset=[column])        
    ax.hist([df[df['risky_yes']==1][column],df[df['risky_no']==1][column]],bins=2,label=['Risky=Yes','Risky=No'])
    ax.set_xlabel(column, fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_title('Risky+'+column, fontsize=fontsize)
    ax.legend()
    index = index + 1


# In[55]:


import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

columns = ['Mileage', 'SalePrice', 'MMR','Autocheck_score']
index = 0
fig, axs = plt.subplots(1,4,figsize=(24, 12))
fontsize = 8

for ax in axs.flat:
    column = columns[index]
    df = df_risky.dropna(subset=[column])
    Xt = np.array(df[column],dtype=np.float32).reshape(-1,1)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(Xt)
    df[column+'_cluster'] = kmeans.predict(Xt)
    centers = kmeans.cluster_centers_
    df[column+'_cluster_center'] = df[column+'_cluster'].apply(lambda s: centers[s][0])
    ax.hist([df[df['risky_yes']==1][column+'_cluster_center'],df[df['risky_no']==1][column+'_cluster_center']],bins=3,label=['Risky=Yes','Risky=No'])
    ax.set_xlabel(column+'_cluster', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_title('Risky+'+column+'_cluster', fontsize=fontsize)
    ax.legend()
    index = index + 1


# In[56]:


columns = [['Mileage', 'SalePrice'],['Mileage', 'MMR'],[ 'SalePrice', 'MMR']]
index = 0
fig, axs = plt.subplots(1,3,figsize=(24, 12))
fontsize = 8

for ax in axs.flat:
    column = columns[index]
    df = df_risky.dropna(subset=column)
    Xt = np.array(df[column],dtype=np.float32).reshape(-1,len(column))

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(Xt)
    df['cluster'] = kmeans.predict(Xt)
    centers = kmeans.cluster_centers_
    print(column, centers)

    df['cluster_center'] = df['cluster'].apply(lambda s: ', '.join([str(c) for c in centers[s]]))
    ax.hist([df[df['risky_yes']==1]['cluster_center'],df[df['risky_no']==1]['cluster_center']],bins=3,label=['Risky=Yes','Risky=No'])
    ax.set_xlabel('cluster', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_title('Risky Dealers behave on '+ ', '.join(column), fontsize=fontsize)
    ax.legend()
    index = index + 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




