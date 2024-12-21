# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:24:52 2024

@author: cbatti545
"""

import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
#import mglearn
from IPython.display import display
import sklearn
import sys
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error 
#https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, RocCurveDisplay, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from math import sqrt
import multiprocessing
import pip

import xgboost as xgb
#################import
player_info=pd.read_csv('players.csv')
player_play=pd.read_csv('player_play.csv').iloc[:,[0,1,2,37,38,39]]
plays_ = pd.read_csv('plays_processed.csv')
games = pd.read_csv('games.csv')


###############join to create base df
plays_ = pd.merge(plays_, games, how= 'left', on='gameId')
plays_['gameDate']=pd.to_datetime(plays_['gameDate'])

plt.figure(figsize=(15,12))
plt.hist(plays_['play_type_design_'], bins=4, color='skyblue', edgecolor='black')
plt.xlabel('Play Type')
plt.ylabel('Frequency')
plt.title('Playcall Distribution')
plt.show()

####personnel hack
play_w_position= pd.merge(player_play, player_info, how= 'left', on='nflId').drop(columns=['height','weight','birthDate','collegeName','displayName'])

te_group=play_w_position.groupby(['gameId','playId'])['position'].apply(lambda x: (x=='TE').sum()).reset_index(name='te_count')
rb_group=play_w_position.groupby(['gameId','playId'])['position'].apply(lambda x: (x=='RB').sum()).reset_index(name='rb_count')

personnel=pd.merge(te_group,rb_group,how='left', on=['gameId','playId'])

personnel['personnel']=(personnel['rb_count'].astype(str))+(personnel['te_count'].astype(str))

#####pre snap
motion_atsnap = play_w_position.groupby(['gameId','playId'])['inMotionAtBallSnap'].apply(lambda x: any(x == True))
motion_presnap = play_w_position.groupby(['gameId','playId'])['motionSinceLineset'].apply(lambda x: any(x == True))
shift_presnap = play_w_position.groupby(['gameId','playId'])['shiftSinceLineset'].apply(lambda x: any(x == True))

pre_snap = pd.concat([motion_atsnap,motion_presnap,shift_presnap],axis=1)


### join personnel + presnap

pre_personnel = pd.merge(personnel,pre_snap, how='left', on=['gameId','playId'])

###join to pbp

plays= pd.merge(plays_, pre_personnel, how='left', on=['gameId','playId'])

plays.info()
plays.describe()


#########
#########
#########
def quarter_minutes(gameClock):
    "converts time stamp into minutes float (by qtr)"
    minutes, seconds = map(int, gameClock.split(":"))
    return minutes + (seconds/60)

plays['qtr_minutes']= plays['gameClock'].apply(quarter_minutes)

plays['game_minutes']= ((plays['quarter']-1)*15)+plays['qtr_minutes']

plays['yds_to_goaline']= np.where(plays['possessionTeam'] == plays['yardlineSide'], plays['yardlineNumber']+50, plays['yardlineNumber'])

plays['poss_team_winning']= np.where((plays['possessionTeam'] == plays['homeTeamAbbr']) & (plays['preSnapHomeScore'] > plays['preSnapVisitorScore']), 
                                     1,np.where((plays['possessionTeam'] == plays['visitorTeamAbbr']) & (plays['preSnapHomeScore'] < plays['preSnapVisitorScore']),1,0))

plays['poss_team_home']= np.where(plays['possessionTeam'] == plays['homeTeamAbbr'], 1, 0)



###cat_dummys
dummy_downs= pd.get_dummies(plays['down'], prefix='down', drop_first=True)
dummy_qtr= pd.get_dummies(plays['quarter'], prefix='qtr', drop_first=True)
dummy_form= pd.get_dummies(plays['offenseFormation'], prefix='form', drop_first=True)
dummy_wr_align= pd.get_dummies(plays['receiverAlignment'], prefix='wr_align', drop_first=True)
dummy_pers= pd.get_dummies(plays['personnel'], prefix='personnel', drop_first=True)

dummy_cols=pd.concat([dummy_downs,dummy_qtr,dummy_form,dummy_wr_align,dummy_pers],axis=1)

####drop columns
plays= plays.drop(columns=['te_count', 'rb_count', 'personnel', 'gameClock','playClockAtSnap',
                           'passLength','targetX','targetY', 'receiverAlignment','offenseFormation','down',
                           'playAction','dropbackType','passLocationType','qbSneak','pff_runConceptPrimary',
                           'pff_runConceptSecondary','pff_passCoverage','pff_runPassOption','pff_manZone','week'])

#plays= plays.drop(columns=['season', 'gameTimeEastern', 'gameDate'])

###grouping by poss team and sorting game date
plays = plays.groupby('possessionTeam').apply(lambda x: x.sort_values('gameDate')).reset_index(level=0, drop=True)

#########################################szn to date avg by team
plays_szn=plays

###################run
running_szn_run_avg = (
    plays_szn[plays_szn['play_type_design'] == "run"]
    .groupby(['possessionTeam', 'gameId'], as_index=False)
    .agg({'yardsGained': 'mean'})  # Average yards gained per game
    .rename(columns={'yardsGained': 'avg_yards_per_game'})
)

running_szn_run_avg['rsh_running_avg_szn'] = (
    running_szn_run_avg.groupby('possessionTeam')['avg_yards_per_game']
    .expanding()
    .mean()
    .reset_index(level=0, drop=True)
)

##################pass
##comps
running_szn_pass_comp_cnt = (
    plays_szn[(plays_szn['play_type_design'] == "pass") & (plays_szn['passResult']=="C")]
    .groupby(['possessionTeam', 'gameId'], as_index=False)
    .agg({'passResult': 'count'})  # Average yards gained per game
    .rename(columns={'passResult': 'comps_per_game'})
)

running_szn_pass_comp_cnt['pass_running_cnt_szn'] = (
    running_szn_pass_comp_cnt.groupby('possessionTeam')['comps_per_game']
    .expanding()
    .sum()
    .reset_index(level=0, drop=True)
)

##att
running_szn_pass_att_cnt = (
    plays_szn[(plays_szn['play_type_design'] == "pass") & ((plays_szn['passResult']!="S") & (plays_szn['passResult']!="R"))]
    .groupby(['possessionTeam', 'gameId'], as_index=False)
    .agg({'passResult': 'count'})  # Average yards gained per game
    .rename(columns={'passResult': 'atts_per_game'})
)

running_szn_pass_att_cnt['pass_att_running_cnt_szn'] = (
    running_szn_pass_att_cnt.groupby('possessionTeam')['atts_per_game']
    .expanding()
    .sum()
    .reset_index(level=0, drop=True)
)

### add play action comp_pct and pa_run call %
running_szn_action_cnt = (
    plays_szn[(plays_szn['play_type_design_'] == "pa_pass")]
    .groupby(['possessionTeam', 'gameId'], as_index=False)
    .agg({'play_type_design_': 'count'})  # Average yards gained per game
    .rename(columns={'play_type_design_': 'pa_per_game'})
)

running_szn_action_cnt['action_running_cnt_szn'] = (
    running_szn_action_cnt.groupby('possessionTeam')['pa_per_game']
    .expanding()
    .sum()
    .reset_index(level=0, drop=True)
)


### play cnt
running_szn_play_cnt = (
    plays_szn[(plays_szn['play_type_design_'] != "")]
    .groupby(['possessionTeam', 'gameId'], as_index=False)
    .agg({'play_type_design_': 'count'})  
    .rename(columns={'play_type_design_': 'play_cnt_per_game'})
)

running_szn_play_cnt['play_cnt_szn'] = (
    running_szn_play_cnt.groupby('possessionTeam')['play_cnt_per_game']
    .expanding()
    .sum()
    .reset_index(level=0, drop=True)
)


###pass call ratio szn to date
running_szn_pass_call_cnt = (
    plays_szn[(plays_szn['play_type_design'] == "pass")]
    .groupby(['possessionTeam', 'gameId'], as_index=False)
    .agg({'play_type_design_': 'count'})  # Average yards gained per game
    .rename(columns={'play_type_design_': 'pass_call_per_game'})
)

running_szn_pass_call_cnt['pass_call_cnt_szn'] = (
    running_szn_pass_call_cnt.groupby('possessionTeam')['pass_call_per_game']
    .expanding()
    .sum()
    .reset_index(level=0, drop=True)
)

#####merge and calc
##comp pct
szn_to_date_pass=pd.merge(running_szn_pass_comp_cnt, running_szn_pass_att_cnt,how="left",on=['possessionTeam','gameId'])
szn_to_date_pass['cmp_pct_szn']=running_szn_pass_comp_cnt['pass_running_cnt_szn']/running_szn_pass_att_cnt['pass_att_running_cnt_szn']
szn_to_date_pass=szn_to_date_pass.drop(columns=['pass_running_cnt_szn','pass_att_running_cnt_szn','comps_per_game','atts_per_game'])
#run and pass running szn
szn_to_date=pd.merge(szn_to_date_pass, running_szn_run_avg,how='left',on=['possessionTeam','gameId']).drop(columns=['avg_yards_per_game'])
#action %
szn_to_date_pa_pct=pd.merge(running_szn_action_cnt,running_szn_play_cnt,how='left',on=['possessionTeam','gameId'])
szn_to_date_pa_pct['pa_call_pct']=szn_to_date_pa_pct['action_running_cnt_szn']/szn_to_date_pa_pct['play_cnt_szn']
szn_to_date_pa_pct=szn_to_date_pa_pct.drop(columns=['pa_per_game','action_running_cnt_szn','play_cnt_szn','play_cnt_per_game'])
szn_to_date=pd.merge(szn_to_date, szn_to_date_pa_pct,how='left',on=['possessionTeam','gameId'])
#pass ratio
szn_to_date_pass_ratio=pd.merge(running_szn_pass_call_cnt,running_szn_play_cnt,how='left',on=['possessionTeam','gameId'])
szn_to_date_pass_ratio['pass_call_pct']=szn_to_date_pass_ratio['pass_call_cnt_szn']/szn_to_date_pass_ratio['play_cnt_szn']
szn_to_date_pass_ratio=szn_to_date_pass_ratio.drop(columns=['pass_call_per_game','pass_call_cnt_szn','play_cnt_szn','play_cnt_per_game'])
szn_to_date=pd.merge(szn_to_date, szn_to_date_pass_ratio,how='left',on=['possessionTeam','gameId'])
################################game to date avg by team
plays_test=plays

plays_test = plays_test.sort_values(['possessionTeam', 'gameId', 'playId'])

# Compute the running average in order
plays_test['rsh_running_avg_gm'] = (
    plays_test[plays_test['play_type_design'] == "run"]  
    .groupby(['possessionTeam', 'gameId'])['yardsGained']
    .expanding()
    .mean()
    .reset_index(level=[0, 1], drop=True)  
)

plays_test['rsh_running_avg_gm'] = plays_test['rsh_running_avg_gm'].fillna(method='ffill')

plays_test['comp_pct_running_avg_gm'] = (
    plays_test[(plays_test['play_type_design'] == "pass") & (plays_test['passResult']=="C")]  # Apply condition
    .groupby(['possessionTeam', 'gameId'])['passResult']
    .expanding()
    .count()
    .reset_index(level=[0, 1], drop=True) /
    plays_test[(plays_test['play_type_design'] == "pass") & ((plays_test['passResult']!="S") & (plays_test['passResult']!="R"))]  # Apply condition
    .groupby(['possessionTeam', 'gameId'])['passResult']
    .expanding()
    .count()
    .reset_index(level=[0, 1], drop=True)
    
)

plays_test['comp_pct_running_avg_gm'] = plays_test['comp_pct_running_avg_gm'].fillna(method='ffill')

###########################final_df with drops
##excluding 2nd, 4th, OT < 2 mins
###exlude first 9 plays of each game??? "3 drives", spikes, kneels

plays_final= pd.merge(plays_test, szn_to_date, how='left',on=['possessionTeam','gameId'])

plays_final= plays_final[plays['quarter'].isin([1,3]) | (plays_final['quarter'].isin([2,4,5]) & plays_final['qtr_minutes']<2)]

plays_final=plays_final[(plays_final['qbSpike'] != True) & (plays_final['qbKneel'] == 0)]

n=6 #two drives

plays_final = plays_final.groupby(['gameId', 'possessionTeam']).apply(lambda x: x.iloc[n:]).reset_index(drop=True)

plays_final=plays_final[(plays_final['rsh_running_avg_szn'] >= 2.5) & (plays_final['rsh_running_avg_szn'] <= 6) & 
                        (plays_final['rsh_running_avg_gm'] <= 10) & (plays_final['rsh_running_avg_gm'] >= 0) &
                        (plays_final['cmp_pct_szn'] >= .55) & (plays_final['cmp_pct_szn'] <= .75)&
                        (plays_final['comp_pct_running_avg_gm'] >= .51) & (plays_final['comp_pct_running_avg_gm'] <= .9)]

#first_occurrences = plays_final.groupby('possessionTeam').head(1)

plays_final=pd.concat([plays_final,dummy_cols],axis=1).drop(columns=['gameId','playId','possessionTeam','defensiveTeam',
                                                                      'yardlineSide','yardlineNumber','preSnapHomeScore',
                                                                      'preSnapVisitorScore','passResult','play_type_design',
                                                                      'season','gameDate','gameTimeEastern','homeTeamAbbr',
                                                                      'visitorTeamAbbr','homeFinalScore','visitorFinalScore',
                                                                      'quarter','qbSpike','qbKneel','absoluteYardlineNumber',
                                                                      'yardsGained']).dropna()

####################
####################
####################EDA
plays_final.info()
descriptive=plays_final.describe()
plays_final.shape
plays_final['inMotionAtBallSnap']=plays_final['inMotionAtBallSnap'].astype(bool)
plays_final['motionSinceLineset']=plays_final['motionSinceLineset'].astype(bool)
plays_final['shiftSinceLineset']=plays_final['shiftSinceLineset'].astype(bool)
# Set Seaborn style
sns.set_style("darkgrid")

numerical_columns = plays_final.select_dtypes(include=["int64", "float64"]).columns

# Plot distribution of each numerical feature
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(plays_final[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(plays_final[feature].skew(), 2)}")

plt.tight_layout()
plt.show()


###
plt.figure(figsize=(10, 8))

sns.violinplot(x="play_type_design_", y="offenseFormation", data=plays_, palette={
               'run': 'lightcoral', 'pa_run': 'lightblue', 'pa_pass': 'lightgreen', 'pass': 'gold'}, alpha=0.7)

plt.title('Violin Plot for Play Type and Formation')
plt.xlabel('Type')
plt.ylabel('Formation')
plt.show()

# Using Seaborn to create a violin plot
sns.violinplot(x="play_type_design_", y="receiverAlignment", data=plays_, palette={
               'run': 'lightcoral', 'pa_run': 'lightblue', 'pa_pass': 'lightgreen', 'pass': 'gold'}, alpha=0.7)

plt.title('Violin Plot for Play Type and WR Alignment')
plt.xlabel('Type')
plt.ylabel('Receiver ALignment')
plt.show()

#####
numeric_columns = plays_final.select_dtypes(include=['float64', 'int64']).columns.tolist()
corr_matrix = plays_final[numerical_columns].corr()

# visualize correlation matrix
plt.figure(figsize=(25, 20))
sns.heatmap(corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            fmt=".2f", 
            annot_kws={"size": 14}
           )
plt.xticks(rotation=90, fontsize=20)
plt.yticks(rotation=0, fontsize=20)
plt.title("Correlation Matrix for Continuous Variables", fontsize=20)
plt.show()

####################
####################
####################model creation

plays_final['play_type_design_'].unique()
label_encoder = LabelEncoder() 
plays_final['play_type_design_']= label_encoder.fit_transform(plays_final['play_type_design_'])

y=plays_final['play_type_design_'].to_frame()
X=plays_final.drop(columns=['play_type_design_'])

X_train, X_test, y_train, y_test = train_test_split(X,y,
    random_state=13, shuffle = True, test_size=.2,  stratify=y)

n_features_base=plays_final.shape[1]
gb_model_base=GradientBoostingClassifier(random_state=13,learning_rate=.01,n_estimators=50, max_features=n_features_base, max_depth=4)

gb_model_base.fit(X_train, y_train)


y_pred_train_base = gb_model_base.predict(X_train)
y_pred_test_base = gb_model_base.predict(X_test)

print("Accuracy on training set: {:.3f}".format(accuracy_score(y_train, y_pred_train_base)))
print("Accuracy on test set: {:.3f}".format(accuracy_score(y_test, y_pred_test_base)))
print("Precision on test set: {:.3f}".format(precision_score(y_test,y_pred_test_base, average='micro')))
print("Recall on test set: {:.3f}".format(recall_score(y_test,y_pred_test_base, average='micro')))

conf_matrix = confusion_matrix(y_test, y_pred_test_base)
print("Confusion Matrix for Test Set:")
print(conf_matrix)

##reverse encode pass:2, run:3, pa_pass:0, pa_run:1

target_names=label_encoder.inverse_transform(plays_final['play_type_design_'])


plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix) ## base params plot
disp.plot(cmap=plt.cm.Blues)
plt.title('Gradient Boost_Base Parameter')
plt.yticks([0,1,2],['PA_Pass','Pass','Run'])
plt.xticks([0,1,2],['PA_Pass','Pass','Run'])
plt.show()


importances = gb_model_base.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

###########
threshold = 0.025
filtered_indices = [i for i in indices if importances[i] >= threshold]  # Filtered indices

X_refit=X.iloc[:,filtered_indices]

n_features=X_refit.shape[1]
# ###splitting into test/train
# X_train_refit, X_test_refit, y_train, y_test = train_test_split(X_refit,y,
#     random_state=13, shuffle = True, test_size=.2)

# ###refit model with no hypertuning, but with feature selection (max_features based on >= .025)
# gb_model_refit=GradientBoostingClassifier(learning_rate=.01,n_estimators=500, max_features=n_features,
#                                          random_state=13).fit(X_train_refit,y_train)

# y_pred_train_refit = gb_model_refit.predict(X_train_refit)
# y_pred_test_refit = gb_model_refit.predict(X_test_refit)

# print("Accuracy on training set_refit: {:.3f}".format(gb_model_refit.score(X_train_refit, y_train)))
# print("Accuracy on test set_refit: {:.3f}".format(gb_model_refit.score(X_test_refit, y_test)))

#####################
#####################
#####################
##gradient boost gridseatch
gb_clf=GradientBoostingClassifier(random_state=13)

parameters = {
    'n_estimators': [25,50,75,100],
    'max_depth':[2,3,5,None],
    'max_features': [n_features,7,5],
    'learning_rate': [.1,.05,.01]
    }


# grid_search =GridSearchCV(gb_clf, parameters, n_jobs=10, cv=5)
# grid_search.fit(X_train, y_train)

# print(grid_search.best_params_)
# print(grid_search.best_score_)

##using best params
gb_clf_best=GradientBoostingClassifier(learning_rate= 0.05, max_depth= 5, max_features= 7, n_estimators= 100,random_state=13)

gb_clf_best.fit(X_train, y_train)

y_pred_train_best = gb_clf_best.predict(X_train)
y_pred_test_best = gb_clf_best.predict(X_test)

print("Accuracy on training set: {:.3f}".format(accuracy_score(y_train, y_pred_train_best)))
print("Accuracy on test set: {:.3f}".format(accuracy_score(y_test, y_pred_test_best)))
print("Precision on test set: {:.3f}".format(precision_score(y_test,y_pred_test_best, average='micro')))
print("Recall on test set: {:.3f}".format(recall_score(y_test,y_pred_test_best, average='micro')))

conf_matrix_best = confusion_matrix(y_test, y_pred_test_best)
print("Confusion Matrix for Test Set:")
print(conf_matrix_best) 

########
importances_clf_best = gb_clf_best.feature_importances_
features_clf_best = X.columns
indices_clf_best = np.argsort(importances_clf_best)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances: Gradient Boost')
plt.barh(range(len(indices_clf_best)), importances_clf_best[indices_clf_best], color='b', align='center')
plt.yticks(range(len(indices_clf_best)), [features_clf_best[i] for i in indices_clf_best])
plt.xlabel('Relative Importance')
plt.show()



# ##### insert orc_auc cuver
# ##train
# y_scoreT=gb_clf_best.predict_proba(X_train)[:,1]
# fprT,tprT,_=roc_curve(y_train,y_scoreT)
# roc_aucT=auc(fprT,tprT)

# ##test
# y_score=gb_clf_best.predict_proba(X_test)[:,1]
# fpr,tpr,_=roc_curve(y_train,y_score)
# roc_auc=auc(fpr,tpr)

# ##viz
# plt.figure()
# plt.plot(fprT, tprT, color='navy', lw=2, label=f'Train ROC (area = {roc_aucT:.4f})')
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Test ROC (area = {roc_auc:.4f})')
# plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')  # Diagonal line for random classifier
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for Gradient Boost on Play Prediction')
# plt.legend(loc="lower right")
# plt.show()
# ##end

target_names=label_encoder.inverse_transform(plays_final['play_type_design_'])

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best) ##best params plot
disp.plot(cmap=plt.cm.Blues)
plt.title('Gradient Boost: Best Parameters\n Test Set Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_test_best)))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks([0,1,2],['PA_Pass','Pass','Run'])
plt.xticks([0,1,2],['PA_Pass','Pass','Run'])
plt.show()


################
################
################
################
#######xg boost model



xgb_model_test = xgb.XGBClassifier(
    n_estimators=75,  # Number of boosting rounds
    max_depth=5,       # Maximum depth of each tree
    learning_rate=0.075, # Step size shrinkage used in update to prevents overfitting
    objective='multi:softmax',  # Multiclass classification objective
    num_class=3,
    random_state=13       # Number of classes in the target variable
)

xgb_model_test.fit(X_train, y_train)

y_pred_xgb_test = xgb_model_test.predict(X_test)
y_pred_xgb_train = xgb_model_test.predict(X_train)

#accuracy_test = accuracy_score(y_test, y_pred_xgb_test)
#print("Accuracy:", accuracy_test)
print("XGB Accuracy on training set: {:.3f}".format(accuracy_score(y_train, y_pred_xgb_train)))
print("XGB Accuracy on test set: {:.3f}".format(accuracy_score(y_test, y_pred_xgb_test)))
print("XGB Precision on test set: {:.3f}".format(precision_score(y_test,y_pred_xgb_test, average='micro')))
print("XGB Recall on test set: {:.3f}".format(recall_score(y_test,y_pred_xgb_test, average='micro')))



##########
############
##xgb grid search
# xgb_model = xgb.XGBClassifier(objective='multi:softmax',  num_class=3,
#     random_state=13)

# parameters_xgb = {
#     'n_estimators': [50,75,100,200,500],
#     'max_depth':[3,5,7,10],
#     'learning_rate': [0.01,0.05,0.075,0.1],
#     'subsample': [0.6, 0.8, 1.0]
#     }

# strata_fold=StratifiedKFold(n_splits=5)
# grid_search_xgb =GridSearchCV(xgb_model, parameters_xgb, 
#                               n_jobs=10, cv=strata_fold,
#                               scoring='accuracy', verbose=0)

# grid_search_xgb.fit(X_train, y_train)

# print('xgb best')
# print(grid_search_xgb.best_params_)
# print(grid_search_xgb.best_score_)

##########################gmai
##########################
##predct xgb
xgb_model_best = xgb.XGBClassifier(
    n_estimators=500,  # Number of boosting rounds
    max_depth=3,       # Maximum depth of each tree
    learning_rate=0.05,
    subsample=.08,# Step size shrinkage used in update to prevents overfitting
    objective='multi:softmax',  # Multiclass classification objective
    num_class=3,
    random_state=13       # Number of classes in the target variable
)
xgb_model_best.fit(X_train, y_train)


# Make predictions on the test set
y_pred_xgb_best = xgb_model_best.predict(X_test)
y_pred_xgb_best_train = xgb_model_best.predict(X_train)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_xgb_best)
#print("Accuracy:", accuracy)
print("XGB Accuracy on training set: {:.3f}".format(accuracy_score(y_train, y_pred_xgb_best_train)))
print("XGB Accuracy on test set: {:.3f}".format(accuracy_score(y_test, y_pred_xgb_best)))

conf_matrix_best_xgb = confusion_matrix(y_test, y_pred_xgb_best)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best_xgb) ##best params plot
disp.plot(cmap=plt.cm.Blues)
plt.title('xGBoost: Best Parameters\n Test Set Accuracy: {:.2f}'.format(accuracy_score(y_test, y_pred_xgb_best)))
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks([0,1,2],['PA_Pass','Pass','Run'])
plt.xticks([0,1,2],['PA_Pass','Pass','Run'])
plt.show()

importances_xg_best = xgb_model_best.feature_importances_
features_xg_best = X.columns
indices_xg_best = np.argsort(importances_xg_best)
plt.figure(figsize=(10, 8))
plt.title('Feature Importances: XG Boost')
plt.barh(range(len(indices_xg_best)), importances_xg_best[indices_xg_best], color='b', align='center')
plt.yticks(range(len(indices_xg_best)), [features_xg_best[i] for i in indices_xg_best])
plt.xlabel('Relative Importance')
plt.show()

 
#########post EDA
# play_dict = {0: 'pa_pass', 1: 'pass', 2: 'run'}

# y_pred_names= pd.Series(y_pred_test_best,name="pred_play").to_frame()

# y_pred_names['play_type']=y_pred_names['pred_play'].map(play_dict)
