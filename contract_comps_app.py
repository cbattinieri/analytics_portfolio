import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
import sklearn
import sys
import seaborn as sns
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from math import sqrt
import xgboost as xgb
import requests
import datetime
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
#import mglearn
#import sys
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, RocCurveDisplay, precision_score, recall_score
from sklearn.ensemble import GradientBoostingClassifier
import requests
import datetime
##############################nhl stats/bio
############################################
############################################
############################################
##########nhl_stats
seasons= ['20032004','20052006','20062007','20072008','20082009','20092010','20102011','20112012',
          '20122013','20132014','20142015','20152016','20162017','20172018','20182019','20192020',
          '20202021','20212022','20222023','20232024','20242025']

url_nhl = "https://api.nhle.com/stats/rest/en/skater/summary?sort=points&limit=-1&cayenneExp=seasonId="


all_data = []


#https://api.nhle.com/stats/rest/en/skater/summary?sort=points&limit=-1&cayenneExp=seasonId=20232024

##########
for season in seasons:
    url = f"{url_nhl}{season}%20and%20gameTypeId=2"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "data" in data and data["data"]:  # Check if data exists
            all_data.extend(data["data"])
        else:
            print(f"No data for season {season}")
    else:
        print(f"Error {response.status_code} for season {season}")

# Convert to DataFrame
df = pd.DataFrame(all_data)
df_base=df.copy()
######################nhl career wrangle
df_base[['career_gp','career_points','career_goals','career_assists',
         'ev_career_points','ev_career_goals']]=df_base.groupby('playerId')[['gamesPlayed','points','goals','assists',
                                                                                                 'evPoints','evGoals']].transform('sum')

df_base['career_assists'] = df_base['career_points'] - df_base['career_goals']
df_base['ev_career_assists'] = df_base['ev_career_points'] - df_base['ev_career_goals']
                                                                             
df_base=df_base.drop(['lastName','skaterFullName','shootsCatches'],axis=1)

df_base['july_1_']=df['seasonId'].astype('str').str.slice(0, 4).str.cat(["-07-01"]* len(df), sep="")
df_base['july_1_']=pd.to_datetime(df_base['july_1_'], format='%Y-%m-%d')

df_stats=df_base.iloc[:, [16,9,13,21,5,6,0,11,12,22,20,19,10,8,3,1,2,14,15,17,18,4,7,23,24,25,26,27,28,29,30]] 

##########
########bio data
bios_url= "https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&cayenneExp=seasonId="
bios_all_data=[]

for season in seasons:
        bio_url_base = f"{bios_url}{season}%20and%20gameTypeId=2"
        response = requests.get(bio_url_base)

        if response.status_code == 200:
            data = response.json()
            if "data" in data and data["data"]:  # Check if data exists
                bios_all_data.extend(data["data"])
            else:
                print(f"No data for season {season}")
        else:
            print(f"Error {response.status_code} for season {season}")

    # Convert to DataFrame
bios_df = pd.DataFrame(bios_all_data)

bios_df=bios_df.drop(['assists','goals','gamesPlayed','points','isInHallOfFameYn','birthCity','birthStateProvinceCode','positionCode'],axis=1).drop_duplicates()

bios_df['birthDate']=pd.to_datetime(bios_df['birthDate'], format='%Y-%m-%d')
########################################################
###main data wrangle
nhl_skater_info_stats=pd.merge(df_stats,bios_df,how='left',on='playerId')

nhl_skater_info_stats['july_1_age']=((nhl_skater_info_stats['july_1_']-nhl_skater_info_stats['birthDate']).dt.days/365.25).astype(int)

nhl_skater_info_stats['evPointsPerGame']=nhl_skater_info_stats['evPoints']/nhl_skater_info_stats['gamesPlayed']

nhl_skater_info_stats['szn_no'] = (
    nhl_skater_info_stats.groupby('playerId').cumcount()+1)

nhl_skater_info_stats['career_to_date_gp'] = (
    nhl_skater_info_stats.groupby('playerId')['gamesPlayed'].cumsum())

nhl_skater_info_stats['career_to_date_points'] = (
    nhl_skater_info_stats.groupby('playerId')['points'].cumsum())

nhl_skater_info_stats['career_to_date_ev_points'] = (
    nhl_skater_info_stats.groupby('playerId')['evPoints'].cumsum())

nhl_skater_info_stats['career_to_date_toi_avg'] = ((
    nhl_skater_info_stats.groupby('playerId')['timeOnIcePerGame'].cumsum()-
    nhl_skater_info_stats['timeOnIcePerGame'])/(nhl_skater_info_stats['szn_no']-1)).fillna(0).round(2)

nhl_skater_info_stats['career_to_date_p_pg'] = ((
    nhl_skater_info_stats.groupby('playerId')['points'].cumsum()-
    nhl_skater_info_stats['points'])/nhl_skater_info_stats['career_to_date_gp']).fillna(0).round(2)

nhl_skater_info_stats['career_to_date_ev_p_pg'] = ((
    nhl_skater_info_stats.groupby('playerId')['evPoints'].cumsum()-
    nhl_skater_info_stats['evPoints'])/nhl_skater_info_stats['career_to_date_gp']).fillna(0).round(2)

nhl_skater_info_stats['pct_gp']= nhl_skater_info_stats['career_to_date_gp']/(nhl_skater_info_stats['szn_no']*82).round(2)  

##l3 and l2 avg loop
for_columns=['gamesPlayed','points','pointsPerGame','evPoints','evPointsPerGame','timeOnIcePerGame']

for columns in for_columns:

    nhl_skater_info_stats[f'{columns}_avg_L3']=((
    nhl_skater_info_stats.groupby('playerId')[columns].shift(0)
    +nhl_skater_info_stats.groupby('playerId')[columns].shift(1)
    +nhl_skater_info_stats.groupby('playerId')[columns].shift(2))/3)


for columns in for_columns:

    nhl_skater_info_stats[f'{columns}_avg_L2']=((
    nhl_skater_info_stats.groupby('playerId')[columns].shift(0)
    +nhl_skater_info_stats.groupby('playerId')[columns].shift(1))/2)
    
###### import CONTRACTS
####################################
api_key = "7cbd0c72c53346c4700590fa074dcf"
url = f"https://puckpedia.com/api/players2?api_key={api_key}"

##
response = requests.get(url)

if response.status_code == 200:
    data = response.json()  # Parse JSON response

else:
    print(f"Error: {response.status_code}, {response.text}")
##

records = data.get("data", []) 

if isinstance(records, list):
     df = pd.DataFrame(records)
    
else:
     print("Expected a list but got:", type(records))
     

#df.info()

df_exploded = df.explode("current")  # Expand lists into rows
df_exploded = pd.concat([df_exploded.drop(columns=["current"]), df_exploded["current"].apply(pd.Series)], axis=1)
#print(df_exploded)

df_temp=df_exploded[['player_id','future']]

df_exploded_new = df_temp.explode("future")  # Expand lists into rows
df_exploded_new = pd.concat([df_exploded_new.drop(columns=["future"]), df_exploded_new["future"].apply(pd.Series)], axis=1)

df_exploded_new=df_exploded_new.drop('player_id',axis=1).add_suffix('_future')
#print(df_exploded_new)

df_exploded=df_exploded.drop('future',axis=1)

contracts_df_base =pd.concat([df_exploded,df_exploded_new],axis=1).drop('0_future',axis=1)

###############################
current_contract_year=20242025
##active contracts, future contracts not incldued excluding current ELC delas
contracts_active=contracts_df_base.iloc[:,0:38]
contracts_active = contracts_active[contracts_active['contract_id'].notna()]
#contracts_active = contracts_active[contracts_active['signing_status'] != False]

###contract year hack
contracts_active['contract_year_hack']=contracts_active['contract_end'].astype('string').str.split('-').str[0]

contracts_active['contract_year1']=contracts_active['contract_year_hack'].astype(int)-contracts_active['length'].astype(int)

contracts_active['contract_year2']=contracts_active['contract_year1']+1

contracts_active['contract_year_hack']=contracts_active['contract_year1'].astype('string')+contracts_active['contract_year2'].astype('string')

contracts_active=contracts_active.drop(columns=['contract_year1','contract_year2','height','weight','city','country','state_province','jersey_number','shoots'],axis=1)

contracts_active['contract_year_hack']=contracts_active['contract_year_hack'].astype(int)
contracts_active['nhl_id']=contracts_active['nhl_id'].astype(int)

contracts_active['contract_year']=np.where(contracts_active['contract_end']=='2024-2025', 20242025,contracts_active['contract_year_hack'])

###need to change expiring FAs this year expiry status as signing status

fa_current_szn=contracts_active[contracts_active['contract_year'] == 20242025]

fa_current_szn['signing_status'] = fa_current_szn['expiry_status']

#fa_current_szn_check = fa_current_szn[fa_current_szn['last_name'].isin(['Peterka', 'Cates', 'Marner'])]
  
fa_prior_szn=contracts_active[contracts_active['contract_year'] != 20242025]

contracts_active=pd.concat([fa_prior_szn,fa_current_szn])

#contracts_active_check = contracts_active[contracts_active['last_name'].isin(['Peterka', 'Cates', 'Marner','Bedard'])]

##
######false are entry level slides
contracts_stats_df=pd.merge(contracts_active, nhl_skater_info_stats, how='inner',left_on=['nhl_id', 'contract_year'],right_on=['playerId', 'seasonId'])

contracts_stats_df=contracts_stats_df[contracts_stats_df['position'] != 'Goaltender']

player_ids_df = contracts_stats_df.loc[:, ['playerId', 'skaterFullName','position','signing_status','contract_year']]
player_ids_df['grouped_pos']=np.where(player_ids_df['position']!='Defense', 'Forward', 'Defense')

players_dict = dict(zip(player_ids_df["playerId"], player_ids_df["skaterFullName"]))

contracts_stats_df=contracts_stats_df.drop(columns=['team_id','nhl_team_id','team_name','career_gp','career_points',
                                                    'career_assists','career_goals','ev_career_assists','ev_career_goals','ev_career_goals',
                                                    'shots','shootingPct','faceoffWinPct','birthDate','birthCountryCode','currentTeamName'],axis=1)
#'team_name.1',

contracts_stats_df['position']=np.where(contracts_stats_df['position'].str.contains('ing'),"Winger",contracts_stats_df['position'])

##removing slide players
contracts_stats_df= contracts_stats_df[contracts_stats_df['signing_status']!=False]

#####features POSITION, SIGNING STATUS,  AGE, career to date gp, career to date p pg (ev), pct_gp
numeric_features=contracts_stats_df.iloc[:,[62,65,69,70,68,71]]

cat_features=contracts_stats_df.iloc[:,[2,26]]

#############model
#############
#############
from sklearn.neighbors import NearestNeighbors
std_scaler = StandardScaler()
    
####splitting (current and previous FAs to ensure scaling)
current_contract_year=20242025

#salary cap number- cap hit % will be aav/ contract year +1 upper limit
cap_data_dict = {'contract_year':[current_contract_year-30003, current_contract_year-20002, current_contract_year-10001,current_contract_year,current_contract_year+10001],
        'upper_limit':[82500000, 83500000, 88000000, 95500000,104000000]}
cap_data=pd.DataFrame(cap_data_dict)

comparables_info=contracts_stats_df.iloc[:,[30,60,62,56,61,31,59,28,13,14,24,64,33,34,35,36,37,38,65,66,67,68,69,70,71]]
comparables_info=pd.merge(comparables_info, cap_data,how='left',on='contract_year')
comparables_info['cap_hit_pct']=(((comparables_info['value'].astype(int)/comparables_info['length'].astype(int))/comparables_info['upper_limit'])*100).round(2)
comparables_info=comparables_info.drop(columns=['upper_limit'],axis=1)


###ufa rfa (excluding no QOs as it causes one off changes to signing status)
RFAs=contracts_stats_df[contracts_stats_df['signing_status']=='RFA']

UFAs=contracts_stats_df[contracts_stats_df['signing_status']=='UFA']

##position
RFAs_fwd=RFAs[RFAs['position']!='Defense'].reset_index().drop(columns='index',axis=1)

RFAs_dmen=RFAs[RFAs['position']=='Defense'].reset_index().drop(columns='index',axis=1)

UFAs_fwd=UFAs[UFAs['position']!='Defense'].reset_index().drop(columns='index',axis=1)

UFAs_dmen=UFAs[UFAs['position']=='Defense'].reset_index().drop(columns='index',axis=1)

####
numeric_features=(RFAs_fwd.iloc[:,[62,65,69,70,68,71]].columns)

###RFA FWD MODEL
RFAs_fwd_scaled=pd.DataFrame(std_scaler.fit_transform(RFAs_fwd[numeric_features]))
RFAs_fwd_new=pd.concat([RFAs_fwd,RFAs_fwd_scaled],axis=1)

RFAs_fwd_pending=RFAs_fwd_new[RFAs_fwd_new['contract_year']==current_contract_year]
RFAs_fwd_prev=RFAs_fwd_new[(RFAs_fwd_new['contract_year']!=current_contract_year)&(RFAs_fwd_new['contract_year']>=(current_contract_year-30003))]

RFAs_fwd_pending_X=RFAs_fwd_pending.iloc[:,-6:]
RFAs_fwd_prev_X=RFAs_fwd_prev.iloc[:,-6:]

RFAs_fwd_knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(RFAs_fwd_prev_X)

RFAs_fwd_distances, RFAs_fwd_indices = RFAs_fwd_knn.kneighbors(RFAs_fwd_pending_X)

    
###RFA DMAN MODEL
RFAs_dmen_scaled=pd.DataFrame(std_scaler.fit_transform(RFAs_dmen[numeric_features]))
RFAs_dmen_new=pd.concat([RFAs_dmen,RFAs_dmen_scaled],axis=1)

RFAs_dmen_pending=RFAs_dmen_new[RFAs_dmen_new['contract_year']==current_contract_year]
RFAs_dmen_prev=RFAs_dmen_new[(RFAs_dmen_new['contract_year']!=current_contract_year)&(RFAs_dmen_new['contract_year']>=(current_contract_year-30003))]

RFAs_dmen_pending_X=RFAs_dmen_pending.iloc[:,-6:]
RFAs_dmen_prev_X=RFAs_dmen_prev.iloc[:,-6:]

RFAs_dmen_knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(RFAs_dmen_prev_X)

RFAs_dmen_distances, RFAs_dmen_indices = RFAs_dmen_knn.kneighbors(RFAs_dmen_pending_X)

########UFA FWD MODEL-- PLATFORM STATS INCLUDED NO POINTS WHEN LOOKING IN SEASON (36)
ufa_numeric_features=(RFAs_fwd.iloc[:,[62,65,69,70,68,71,33,37,38]].columns)

###UFA FWD MODEL
UFAs_fwd_scaled=pd.DataFrame(std_scaler.fit_transform(UFAs_fwd[ufa_numeric_features]))
UFAs_fwd_scaled.iloc[:,0]=UFAs_fwd_scaled.iloc[:,0]*30
UFAs_fwd_scaled.iloc[:,1]=UFAs_fwd_scaled.iloc[:,1]*10
UFAs_fwd_scaled.iloc[:,2]=UFAs_fwd_scaled.iloc[:,2]*10
UFAs_fwd_scaled.iloc[:,6]=UFAs_fwd_scaled.iloc[:,6]*.25
UFAs_fwd_scaled.iloc[:,7]=UFAs_fwd_scaled.iloc[:,7]*.25
UFAs_fwd_scaled.iloc[:,8]=UFAs_fwd_scaled.iloc[:,8]*.25
UFAs_fwd_new=pd.concat([UFAs_fwd,UFAs_fwd_scaled],axis=1)

UFAs_fwd_pending=UFAs_fwd_new[UFAs_fwd_new['contract_year']==current_contract_year]
UFAs_fwd_prev=UFAs_fwd_new[(UFAs_fwd_new['contract_year']!=current_contract_year)&(UFAs_fwd_new['contract_year']>=(current_contract_year-30003))]

UFAs_fwd_pending_X=UFAs_fwd_pending.iloc[:,-6:]
UFAs_fwd_prev_X=UFAs_fwd_prev.iloc[:,-6:]

UFAs_fwd_knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(UFAs_fwd_prev_X)

UFAs_fwd_distances, UFAs_fwd_indices = UFAs_fwd_knn.kneighbors(UFAs_fwd_pending_X)

########UFA DMAN MODEL-- SHOULD ADD PLATFORM YEARS    
UFAs_dmen_scaled=pd.DataFrame(std_scaler.fit_transform(UFAs_dmen[ufa_numeric_features]))
#UFAs_dmen_scaled.iloc[:,0]=UFAs_dmen_scaled.iloc[:,0]*1.25
#UFAs_dmen_scaled.iloc[:,7]=UFAs_dmen_scaled.iloc[:,7]*1.1
UFAs_dmen_new=pd.concat([UFAs_dmen,UFAs_dmen_scaled],axis=1)

UFAs_dmen_pending=UFAs_dmen_new[UFAs_dmen_new['contract_year']==current_contract_year]
UFAs_dmen_prev=UFAs_dmen_new[(UFAs_dmen_new['contract_year']!=current_contract_year)&(UFAs_dmen_new['contract_year']>=(current_contract_year-30003))]

UFAs_dmen_pending_X=UFAs_dmen_pending.iloc[:,-6:]
UFAs_dmen_prev_X=UFAs_dmen_prev.iloc[:,-6:]

UFAs_dmen_knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(UFAs_dmen_prev_X)

UFAs_dmen_distances, UFAs_dmen_indices = UFAs_dmen_knn.kneighbors(UFAs_dmen_pending_X)

#########################

#########################comps distnaces
num_similar_players = len(UFAs_fwd_indices[0])  # Number of similar players to include
columns = ['Player'] + [f'Comp. {i+1}' for i in range(num_similar_players)]
ufa_fwd_comps = pd.DataFrame(columns=columns)

for i, (distance, index) in enumerate(zip(UFAs_fwd_distances, UFAs_fwd_indices)):
    row = [UFAs_fwd_pending.iloc[i]['playerId']]  
    row.extend([UFAs_fwd_prev.iloc[idx]['playerId'] for idx in index])
    ufa_fwd_comps.loc[i] = row
##ufa d 
num_similar_players_ufa_d = len(UFAs_dmen_indices[0])  # Number of similar players to include
columns_ufa_d = ['Player'] + [f'Comp. {i+1}' for i in range(num_similar_players_ufa_d)]
data_ufa_d = []

for i, (distance, index) in enumerate(zip(UFAs_dmen_distances, UFAs_dmen_indices)):
    row = [UFAs_dmen_pending.iloc[i]['playerId']]  
    row.extend([UFAs_dmen_prev.iloc[idx]['playerId'] for idx in index])
    data_ufa_d.append(row)

ufa_dmen_comps = pd.DataFrame(data_ufa_d, columns=columns_ufa_d)

##rfa d
num_similar_players_rfa_d = len(RFAs_dmen_indices[0])  # Number of similar players to include
columns_rfa_d = ['Player'] + [f'Comp. {i+1}' for i in range(num_similar_players_rfa_d)]
data_rfa_d = []

for i, (distance, index) in enumerate(zip(RFAs_dmen_distances, RFAs_dmen_indices)):
    row = [RFAs_dmen_pending.iloc[i]['playerId']]  
    row.extend([RFAs_dmen_prev.iloc[idx]['playerId'] for idx in index])
    data_rfa_d.append(row)

rfa_dmen_comps = pd.DataFrame(data_rfa_d, columns=columns_rfa_d)

##rfa fwd
num_similar_players_rfa_f = len(RFAs_fwd_indices[0])  # Number of similar players to include
columns_rfa_f = ['Player'] + [f'Comp. {i+1}' for i in range(num_similar_players_rfa_f)]
data_rfa_f = []

for i, (distance, index) in enumerate(zip(RFAs_fwd_distances, RFAs_fwd_indices)):
    row = [RFAs_fwd_pending.iloc[i]['playerId']]  
    row.extend([RFAs_fwd_prev.iloc[idx]['playerId'] for idx in index])
    data_rfa_f.append(row)

rfa_fwd_comps = pd.DataFrame(data_rfa_f, columns=columns_rfa_f)

all_comps=pd.concat([rfa_fwd_comps,rfa_dmen_comps,ufa_fwd_comps,ufa_dmen_comps])

all_comps=all_comps.astype(int)

##info to display, showing old contract details for 
comparables_info=contracts_stats_df.iloc[:,[30,60,62,56,61,31,59,28,13,14,24,64,33,34,35,36,37,38,65,66,67,68,69,70,71]]
comparables_info=pd.merge(comparables_info, cap_data,how='left',on='contract_year')
comparables_info['cap_hit_pct']=(((comparables_info['value'].astype(int)/comparables_info['length'].astype(int))/comparables_info['upper_limit'])*100).round(2)
comparables_info=comparables_info.drop(columns=['upper_limit'],axis=1)


filtered_ufas_fwd = UFAs_fwd[UFAs_fwd['contract_year'] == current_contract_year]
# Create a dictionary with playerId as the key and skaterFullName as the value
players_dict = dict(zip(filtered_ufas_fwd['playerId'], filtered_ufas_fwd['skaterFullName']))

stats_dict={'pointsPerGame':'Pts perGame','gamesPlayed':'GP','points':'Pts','career_to_date_p_pg':'CareerPtsPerGame','career_to_date_gp':'CareerGP','career_to_date_points':'CareerPts'}
#######################
########################
########################app    
import matplotlib.pyplot as plt
import numpy as np
from shiny.express import ui, input, render
from shiny import reactive, req
from shiny_validate import InputValidator, check
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import make_interp_spline
import plotly.express as px
from shinywidgets import render_plotly
iv= InputValidator()

ui.page_opts(title="NHL Contract Comparables Tool", fillable=True, id="page")

with ui.sidebar():
    ui.input_radio_buttons(  
    "fa_radio",  
    "Free Agent Status:",  
    {"UFA": "UFA", "RFA": "RFA"},  
)  
    ui.input_radio_buttons(  
    "pos_radio",  
    "Position:",  
    {"Forward": "Forward", "Defense": "Defensemen"},  
)

    ui.input_selectize(
            "players",
            "Search for player:",
            multiple=False,
            choices=players_dict,
            selected=next(iter(players_dict.keys()), None),
            width="100%",
)  
with ui.navset_pill(id="tab"):  
    with ui.nav_panel("Contract&Stats"):
        
        with ui.layout_columns():
            with ui.card(full_screen=True,fill=True,height="200px"):
                ui.card_header("Player going to Market:")
                @render.data_frame
                def table_single():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_player().drop(columns=['Height','Weight','Shoots','Position'],axis=1))

        with ui.layout_columns():
            with ui.card(full_screen=True):
                ui.card_header("Comparables:")
                @render.data_frame
                def table_comps():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_comps().drop(columns=['Height','Weight','Shoots','Position'],axis=1))
                ui.card_footer("Career stats are career totals until listed 'contract year' (Contract Year included)")      

        with ui.layout_columns():
            with ui.card():
                ui.card_header(ui.input_select("stat_dropdown", None, choices=stats_dict, selected='Pts perGame', width="auto",),"by Age:")
                @render.plot
                def points_plot():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, "", 
                                fontsize=12, ha='center', va='center', transform=ax.transAxes)
                        ax.set_axis_off()  # Hide axes
                        return fig  # Return the placeholder plot
                    
                    unique_ids = career_plots_df()['skaterFullName'].unique()
                    
                    fig, ax = plt.subplots()

                    for i, player_id in enumerate(unique_ids):
                        player_data = career_plots_df()[career_plots_df()['skaterFullName'] == player_id]
                        x = player_data['july_1_age']
                        y = player_data[input.stat_dropdown()]
                        
                        ax.plot(x, y, label=f'{player_id}', color='gray' if i > 0 else 'black', linewidth=1 if i > 0 else 2)

                    ax.set_xlabel('July1stAge')
                    ax.set_ylabel('Points perGame')
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    ax.set_xlim(int(career_plots_df()['july_1_age'].min()), int(career_plots_df()['july_1_age'].max()) + 1)
                    ax.legend()
                    
                    return fig

                    # ax.set_xlabel('July1stAge')
                    # ax.set_ylabel('Points perGame')
                    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    # ax.set_xlim(int(career_plots_df()['july_1_age'].min()), int(career_plots_df()['july_1_age'].max()) + 1)

                    # return fig

    with ui.nav_panel("Stats"):
        with ui.layout_columns():
            with ui.card(full_screen=True,fill=True,height="200px"):
                ui.card_header("Player going to Market:")
                @render.data_frame
                def table_single_stats():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_player().drop(columns=['Height','Weight','Shoots','Position','Term','Value','AAV','Cap Hit % (*Estimate)'],axis=1))

        with ui.layout_columns():
            with ui.card(full_screen=True):
                ui.card_header("Comparables:")
                @render.data_frame
                def table_comps_stats():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_comps().drop(columns=['Height','Weight','Shoots','Position','Term','Value','AAV','Cap Hit %'],axis=1))
                ui.card_footer("Career stats are career totals until listed 'contract year' (Contract Year included)")      

    with ui.nav_panel("Contract"):
        with ui.layout_columns():
            with ui.card(full_screen=True,fill=True,height="150px"):
                ui.card_header("Player going to Market:")
                @render.data_frame
                def table_single_contracts():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_player().iloc[:,[0,1,6,7,8,9,10]])
        with ui.layout_columns():        
            with ui.card(full_screen=True):
                ui.card_header("Comparables:")
                @render.data_frame
                def table_comps_contracts():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_comps().iloc[:,[0,1,6,7,8,9,10]])
        with ui.layout_columns():
            with ui.card():
                ui.card_header("Comparables Contracts:")
                @render.plot
                def contract_scatter_plot():
                    req(iv.is_valid())
                    if not input.players():  
                        return plt.subplots()
                    fig, ax = plt.subplots()
                    #ax.scatter(final_comps_df_player()['Cap Hit % (*Estimate)'], final_comps_df_comps()['Term'].mean(), color='black')
                    # Plot the rest of the rows with thinner lines and gray color
                    ax.scatter(final_comps_df_comps()['Cap Hit %'], final_comps_df_comps()['Term'], color='gray')
                    ax.set_xlabel('Cap Hit%')
                    ax.set_ylabel('Term') 
                    
                    return fig
            with ui.card(full_screen=True):
                ui.card_header('Estimated Cap Hit % Future Values:')
                @render.plot
                def cap_hit_future_plot():
                    req(iv.is_valid())
                    if not input.players():
                        return plt.subplots()
                         # Generate a bar chart for these columns
                    fig, ax = plt.subplots()
                    seasons = ['Cap Hit % Season1', 'Cap Hit % Season2', 'Cap Hit % Season3']
                    player_future_df = final_comps_df_player_future()
                    if not isinstance(player_future_df, pd.DataFrame) or not all(season in player_future_df.columns for season in seasons):
                        ax.text(0.5, 0.5, "No data available", fontsize=12, ha='center', va='center', transform=ax.transAxes)
                        ax.set_axis_off()
                        return fig
                    values = player_future_df[seasons].values.flatten()
                    colors = ['#1f77b4', '#4a90d9', '#87bff3']  # Different shades of the same color
                    bars = ax.bar(seasons, values, color=colors)
                    
                    # Add data labels on top of bars
                    for bar, value in zip(bars, values):
                        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height()-bar.get_height() + 0.5, f'{value:.2f}', 
                                ha='center', va='bottom', fontsize=10)
                    
                    ax.set_ylabel('Cap Hit %')
                    ax.set_title('Cap Hit % into Future Seasons')
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    return fig   
    with ui.nav_panel("Bio"):
        with ui.layout_columns():
            with ui.card(full_screen=True,fill=True,height="150px"):
                ui.card_header("Player going to Market:")
                @render.data_frame
                def table_single_bios():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_player().iloc[:,:5])
        with ui.layout_columns():        
            with ui.card(full_screen=True):
                ui.card_header("Comparables:")
                #ui.navset_card_tab(id="tab2")
                @render.data_frame
                def table_comps_bios():
                    req(iv.is_valid())
                    if not input.players():  # Check if the player input is blank
                        return pd.DataFrame()
                    return render.DataGrid(final_comps_df_comps().iloc[:,:5]) 
    with ui.nav_panel("Info & Notes"):
        with ui.layout_columns():
            with ui.card(full_screen=True,fill=True):
                ui.markdown(
                """
                ## Model
                - Comparables are based on a Machine Learning model that uses defined features to find similar players.  
                - There are 4 models used: UFA Dmen, UFA Fwds, RFA Dmen, RFA Fwds to help eliminate bias and normalize the data.  
                - UFA Models were trained to use age and career to date stats as the most influential features with platform year stats also factored in.
                - RFA Models were trained to use career to date stats as the most influential features with no platform year stats included.
                - All players will have exactly 5 comparables.
                - Could see unconventional comparables in "superstar" players, mid 30's aged players, & RFA's with very few games played both in career and platform years.
                ## Cap Hit Percentage Estimates and AAV Calculation 
                - Cap Hit Percntage Estimate is a weighted average of the comparables Cap Hit %. 
                - The weights of each comparable contract are 30%, 25%, 20%, 15%, 10%, following in order the comparables listed. 
                - AAV for player coming to market calculated as Cap Hit Percentage x $95.5M.    
                ## Data
                - The model uses the last 3 seasons of free agent data to find similar players.  
                - Contract data from PuckPedia.com, Stats and Bio data from NHL.com.
                - Players who have signed in-season extensions still listed as FAs.
                """
                )     
###########################
@reactive.calc
def players_id():
    fa_status = input.fa_radio()
    position = input.pos_radio()
    idx = (
        (player_ids_df["grouped_pos"] == position)
        & (player_ids_df["signing_status"] == fa_status)
        & (player_ids_df["contract_year"] == current_contract_year)
    )
    return player_ids_df[idx]

# Update available players when careers data changes
@reactive.effect
def _():
    players = dict(zip(players_id()["playerId"], players_id()["skaterFullName"]))
    ui.update_selectize("players", choices=players, selected=input.players() or next(iter(players.keys()), None))

##df for player comparables
@reactive.calc
def final_comps_df_initial():
    # Use the reactive selected player ID
    player_id = input.players()
    player_id = int(player_id) 
    
    return player_id

@reactive.calc
def final_comps_df():
    comp_stats = all_comps[all_comps['Player'] == final_comps_df_initial()]
    
    comp_stats_2= pd.DataFrame({"player_id":[comp_stats.iloc[0,0],comp_stats.iloc[0,1],comp_stats.iloc[0,2],comp_stats.iloc[0,3],comp_stats.iloc[0,4],comp_stats.iloc[0,5]]})
    #return comp_stats_2    
  
    # # Merge the comparable stats with the player information
    final_comps = pd.merge(comp_stats_2, comparables_info, how='left', left_on='player_id', right_on='playerId')
    final_comps = final_comps.drop(columns=['player_id'],axis=1)
    # Calculate weighted cap hit percentage
    if len(final_comps) > 5:  # Ensure there are enough comparable players
        final_comps['cap_hit_pct'][0] = (
            (final_comps['cap_hit_pct'][1] * 0.3) +
            (final_comps['cap_hit_pct'][2] * 0.25) +
            (final_comps['cap_hit_pct'][3] * 0.20) +
            (final_comps['cap_hit_pct'][4] * 0.15) +
            (final_comps['cap_hit_pct'][5] * 0.1)
        ).round(2)

    final_comps['pct_gp']=(final_comps['pct_gp']*100).round(2)
    final_comps['timeOnIcePerGame']=final_comps['timeOnIcePerGame'].round(2)
    final_comps['pointsPerGame']=final_comps['pointsPerGame'].round(2)
    final_comps['timeOnIcePerGame']=(final_comps['timeOnIcePerGame']/60).round(2)
    final_comps.rename(columns={'skaterFullName': 'Player', 'value': 'Value', 'length': 'Term', 'cap_hit_pct': 'Cap Hit %','july_1_age':'July1stAge',
                                                  'positionCode':'Position','shootsCatches':'Shoots','pct_gp':'CareerGP %','timeOnIcePerGame':'TOI perGame',
                                                  'contract_year':'Contract Year','weight':'Weight','height':'Height','current_season_aav':'AAV',
                                                  'szn_no':'Season #','gamesPlayed':'GP','goals':'G','assists':'A','points':'Pts','pointsPerGame':'Pts perGame',
                                                  'career_to_date_gp':'CareerGP','career_to_date_points':'CareerPts','career_to_date_ev_points':'CareerEVPts',
                                                  'career_to_date_toi_avg':'CareerTOIAvg','career_to_date_p_pg':'CareerPtsPerGame','career_to_date_ev_p_pg':'CareerEVPtsPerGame'}, inplace=True)
    if 'Cap Hit %' in final_comps.columns:
        column_to_move = final_comps.pop('Cap Hit %') # Remove the column and store it
        final_comps.insert(8, 'Cap Hit %', column_to_move) # Insert at the specified position with column name
    final_comps=final_comps.drop(columns=['playerId'],axis=1)
    return final_comps

@reactive.calc
def final_comps_df_single():
    comp_stats = all_comps[all_comps['Player'] == final_comps_df_initial()]
    
    comp_stats_2= pd.DataFrame({"player_id":[comp_stats.iloc[0,0],comp_stats.iloc[0,1],comp_stats.iloc[0,2],comp_stats.iloc[0,3],comp_stats.iloc[0,4],comp_stats.iloc[0,5]]})
    #return comp_stats_2    
  
    # # Merge the comparable stats with the player information
    final_comps = pd.merge(comp_stats_2, comparables_info, how='left', left_on='player_id', right_on='playerId')
    final_comps = final_comps.drop(columns=['player_id'],axis=1)
    # Calculate weighted cap hit percentage
    if len(final_comps) > 5:  # Ensure there are enough comparable players
        final_comps['cap_hit_pct'][0] = (
            (final_comps['cap_hit_pct'][1] * 0.3) +
            (final_comps['cap_hit_pct'][2] * 0.25) +
            (final_comps['cap_hit_pct'][3] * 0.20) +
            (final_comps['cap_hit_pct'][4] * 0.15) +
            (final_comps['cap_hit_pct'][5] * 0.1)
        ).round(2)

    final_comps['pct_gp']=(final_comps['pct_gp']*100).round(2)
    final_comps['timeOnIcePerGame']=final_comps['timeOnIcePerGame'].round(2)
    final_comps['pointsPerGame']=final_comps['pointsPerGame'].round(2)
    final_comps['timeOnIcePerGame']=(final_comps['timeOnIcePerGame']/60).round(2)
    final_comps.rename(columns={'skaterFullName': 'Player', 'value': 'Value', 'length': 'Term', 'cap_hit_pct': 'Cap Hit %','july_1_age':'July1stAge',
                                                  'positionCode':'Position','shootsCatches':'Shoots','pct_gp':'CareerGP %','timeOnIcePerGame':'TOI perGame',
                                                  'contract_year':'Contract Year','weight':'Weight','height':'Height','current_season_aav':'AAV',
                                                  'szn_no':'Season #','gamesPlayed':'GP','goals':'G','assists':'A','points':'Pts','pointsPerGame':'Pts perGame',
                                                  'career_to_date_gp':'CareerGP','career_to_date_points':'CareerPts','career_to_date_ev_points':'CareerEVPts',
                                                  'career_to_date_toi_avg':'CareerTOIAvg','career_to_date_p_pg':'CareerPtsPerGame','career_to_date_ev_p_pg':'CareerEVPtsPerGame'}, inplace=True)
    if 'Cap Hit %' in final_comps.columns:
        column_to_move = final_comps.pop('Cap Hit %') # Remove the column and store it
        final_comps.insert(8, 'Cap Hit %', column_to_move) # Insert at the specified position with column name
    final_comps=final_comps.drop(columns=['playerId'],axis=1)
    return final_comps

@reactive.calc
def final_comps_df_comps():
    comps_df = final_comps_df()
    if not comps_df.empty:
        return comps_df.iloc[-5:]  # Use iloc to select the last 4 rows
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data is available

@reactive.calc
def final_comps_df_player():
    player_comps_df = final_comps_df_single()
    player_comps_df['Term']="?"
    player_comps_df['Value']="?"
    player_comps_df['AAV'] = ((player_comps_df['Cap Hit %'] * 95500000) / 100).round(2)
    player_comps_df.rename(columns={'Cap Hit %':'Cap Hit % (*Estimate)'}, inplace=True)
    if not player_comps_df.empty:
        return player_comps_df.iloc[:1]
    else:
        return "Search for player"  # Return an empty DataFrame if no data is available
    
@reactive.calc
def final_comps_df_player_future():
    player_comps_df = final_comps_df_player()
    player_comps_df['Term']="?"
    player_comps_df['Value']="?"
    #player_comps_df['AAV'] = ((player_comps_df['Cap Hit %'] * 95500000) / 100).round(2)
    #player_comps_df.rename(columns={'Cap Hit %':'Cap Hit % (*Estimate)'}, inplace=True)
    player_comps_df['Cap Hit % Season1'] = player_comps_df['Cap Hit % (*Estimate)']
    player_comps_df['Cap Hit % Season2'] = ((player_comps_df['AAV'] / 104000000) * 100).round(2)
    player_comps_df['Cap Hit % Season3'] = ((player_comps_df['AAV'] / 113500000) * 100).round(2)
    if not player_comps_df.empty:
        return player_comps_df.iloc[:1]
    else:
        return "Search for player"  # Return an empty DataFrame if no data is available
    
@reactive.calc
def career_plots_df():
    comp_stats = all_comps[all_comps['Player'] == final_comps_df_initial()]
    
    comp_stats_2= pd.DataFrame({"player_id":[comp_stats.iloc[0,0],comp_stats.iloc[0,1],comp_stats.iloc[0,2],comp_stats.iloc[0,3],comp_stats.iloc[0,4],comp_stats.iloc[0,5]]})
    all_stats=nhl_skater_info_stats.iloc[:,[1,43,45,4,7,8,48,49,52]]
    comp_stats_all_stats=pd.merge(comp_stats_2, all_stats,how='left',left_on='player_id',right_on='playerId')

    
    return comp_stats_all_stats


