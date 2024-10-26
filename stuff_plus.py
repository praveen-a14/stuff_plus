import os
import numpy as np
import pandas as pd
from pybaseball import statcast, statcast_pitcher_arsenal_stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import sqlite3
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll.base import scope

# Establish connection to SQLite DB. CSVs have already been imported to tables in the db via terminal
conn = sqlite3.connect('/Users/praveenarunshankar/Library/CloudStorage/OneDrive-Personal/Documents/my-code/python-code/baseball/stuff_plus.db')
cursor = conn.cursor()

# Select relevant cols, Group by pitcher and pitch and caculate averages and identify primary pitch. Calculate run values for model's target variable
cursor.execute('''CREATE TEMP TABLE temp_results AS
                  SELECT pitcher, player_name, strftime('%Y', game_date) AS year, p_throws,
                        CASE WHEN pitch_type = 'SL' AND ROUND(AVG(CASE WHEN p_throws = 'L' THEN pfx_x * -12 ELSE pfx_x * 12 END), 2) > 9 THEN 'ST'
                        WHEN pitch_type = 'ST' AND ROUND(AVG(CASE WHEN p_throws = 'L' THEN pfx_x * -12 ELSE pfx_x * 12 END), 2) <= 9 THEN 'SL'
                        ELSE pitch_type END AS pitch_type,
                        COUNT(*) AS count, ROUND(AVG(release_speed), 2) AS velo, 
                        ROUND(AVG(pfx_z) * 12, 2) AS iVB,
                        ROUND(AVG(CASE WHEN p_throws = 'L' THEN pfx_x * -12 ELSE pfx_x * 12 END), 2) AS HB,
                        ROUND(AVG(release_spin_rate), 0) AS spin, ROUND(AVG(release_pos_z), 2) AS rel_z, 
                        ROUND(AVG(CASE WHEN p_throws = 'L' THEN release_pos_x * -1 ELSE release_pos_x * 1 END), 2) AS rel_x, 
                        ROUND(LN(AVG(release_extension)), 2) AS ext, 
                        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY pitcher, strftime('%Y', game_date)), 1) AS usg, 
                        FIRST_VALUE(pitch_type) OVER (PARTITION BY pitcher, strftime('%Y', game_date) ORDER BY COUNT(*) FILTER (WHERE pitch_type IN ('FF', 'SI')) DESC) AS pri,
                        ROUND((SUM(delta_run_exp)/COUNT(*)) * -100, 3) AS crv_100
                  FROM full_sc
                  WHERE game_type = "R"
                        AND pitch_type IN ('FF', 'SI', 'FC', 'CU', 'CH', 'SL', 'SV', 'KC', 'FS', 'ST', 'FO', 'SC', 'CS', 'KN') 
                  GROUP BY pitcher, player_name, pitch_type, strftime('%Y', game_date)''')

# Join the table with arm angle data
cursor.execute('''CREATE TEMP TABLE temp_aa AS
                  SELECT t.*, a.arm_angle  
                  FROM temp_results t
                  INNER JOIN arm_data a ON t.pitcher = a.player_id AND t.year = a.year''')

# Calculate velocity and movement differences from primary pitch and create a permanent table with the full data
cursor.execute('DROP TABLE IF EXISTS full_data;')
cursor.execute('''CREATE TABLE full_data AS
                  SELECT t.*, 
                         ROUND(((MAX(CASE WHEN (t.pitch_type = t.pri AND t.year = t.year) THEN t.velo END) OVER (PARTITION BY t.pitcher, t.year)) - t.velo), 2) AS v_dif,
                         ROUND(t.iVB - MAX(CASE WHEN (t.pitch_type = t.pri AND t.year = t.year) THEN t.iVB END) OVER (PARTITION BY t.pitcher, t.year), 2) AS iVB_dif,
                         ROUND(t.HB - MAX(CASE WHEN (t.pitch_type = t.pri AND t.year = t.year) THEN t.HB END) OVER (PARTITION BY t.pitcher, t.year), 2) AS HB_dif
                  FROM 
                      temp_aa t;''')

# Query the db to get pandas dfs of each pitch. Used 2021-2023 to train the model and 2024 to test
#fourseamers
ff = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'FF' AND pri = 'FF') AND count >= 100 AND year <> '2024';", conn)
ff_24 = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'FF' AND pri = 'FF')  AND count >= 15 AND year = '2024';", conn)
#sinkers
si = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'SI' AND pri = 'SI') AND count >= 100 AND year <> '2024';", conn)
si_24 = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'SI' AND pri = 'SI')  AND count >= 15 AND year = '2024';", conn)
# cutters sliders
fc = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'FC' OR pitch_type = 'SL') AND count >= 100 AND year <> '2024';", conn)
fc_24 = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'FC' OR pitch_type = 'SL')  AND count >= 15 AND year = '2024';", conn)
# sweepers
st = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'ST') AND count >= 100 AND year <> '2024';", conn)
st_24 = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type = 'ST')  AND count >= 15 AND year = '2024';", conn)
# changeups splitters forkballs
ch = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type IN ('CH', 'FS', 'FO')) AND count >= 100 AND year <> '2024';", conn)
ch_24 = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type IN ('CH', 'FS', 'FO'))  AND count >= 15 AND year = '2024';", conn)
# curveballs knuckle curves slurves
cu = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type IN ('CU', 'KC', 'SV')) AND count >= 100 AND year <> '2024';", conn)
cu_24 = pd.read_sql_query("SELECT * FROM full_data WHERE (pitch_type IN ('CU', 'KC', 'SV'))  AND count >= 15 AND year = '2024';", conn)
#full data
full = pd.read_sql_query('SELECT * FROM full_data WHERE (year <> "2024" AND count >= 100);', conn)
test_24 = pd.read_sql_query('SELECT * FROM full_data WHERE (year = "2024" AND count >= 15);', conn)

conn.close()

# SQL might make some numeric columns into characters, take care of that here
numeric_columns = ['velo','spin', 'rel_z', 'rel_x', 'ext', 'crv_100', 'arm_angle', 'v_dif', 'iVB_dif', 'HB_dif', 'count', 'pitcher', 'year']
for column in numeric_columns:
    ff[column] = pd.to_numeric(ff[column], errors='coerce')
    ff_24[column] = pd.to_numeric(ff_24[column], errors='coerce')
    full[column] = pd.to_numeric(full[column], errors='coerce')
    test_24[column] = pd.to_numeric(test_24[column], errors='coerce')
    si[column] = pd.to_numeric(si[column], errors='coerce')
    si_24[column] = pd.to_numeric(si_24[column], errors='coerce')
    fc[column] = pd.to_numeric(fc[column], errors='coerce')
    fc_24[column] = pd.to_numeric(fc_24[column], errors='coerce')
    st[column] = pd.to_numeric(st[column], errors='coerce')
    st_24[column] = pd.to_numeric(st_24[column], errors='coerce')
    ch[column] = pd.to_numeric(ch[column], errors='coerce')
    ch_24[column] = pd.to_numeric(ch_24[column], errors='coerce')
    cu[column] = pd.to_numeric(cu[column], errors='coerce')
    cu_24[column] = pd.to_numeric(cu_24[column], errors='coerce')

############################ FASTBALL MODEL ##############################

# drop irrelevant columns to select features
ff_X = ff.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'v_dif', 'iVB_dif', 'HB_dif', 'rel_z', 'rel_x', 'usg']) 
# identify target variable
ff_y = ff['crv_100']
# select numerical columns (all)
ff_numerical_cols = ff_X.select_dtypes(exclude=['object']).columns.tolist()
# split data into test and train sets
ff_X_train, ff_X_test, ff_y_train, ff_y_test = train_test_split(ff_X, ff_y, test_size=0.2, random_state=50)
# create xgboost model. Used hyperopt to tune hyperparameters for each model
ff_preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', ff_numerical_cols)])
ff_model = Pipeline(steps=[('preprocessor', ff_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.02,
    max_depth = 3,
    subsample = 0.75,
    min_child_weight = 9,
    colsample_bytree = 0.7,
    gamma = 0.1,
    reg_lambda = 0.5,
    reg_alpha = 0.9,
    random_state = 50))])

# fit the model, predict on the test set and caluclate r-squared
ff_model.fit(ff_X_train, ff_y_train)
ff_y_pred = ff_model.predict(ff_X_test)
ff_r2 = r2_score(ff_y_test, ff_y_pred)
#print(ff_r2)

# run and print feature importances
ff_importances = ff_model.named_steps['regressor'].feature_importances_
ff_feature_names = ff_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(ff_numerical_cols).tolist()
ff_feature_importance_df = pd.DataFrame({'Feature': ff_feature_names, 'Importance': ff_importances})
ff_feature_importance_df = ff_feature_importance_df.sort_values(by='Importance', ascending=False)
#print(ff_feature_importance_df) 

# test data on 2024 fastballs
ff_24['xrv'] = ff_model.predict(ff_24)
# use z-score to create stuff+, xrv on the wrc+ scale
ff_24['stuff'] = (ff_24['xrv'])/(abs(ff_24['xrv'].mean())) * 100
ff_24['z'] = (ff_24['xrv'] - ff_24['xrv'].mean())/ff_24['xrv'].std()
ff_24['stuff'] = (ff_24['z'] * 10) + 100
# print leaderboard
#print(ff_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30)) 

################## SINKER MODEL ####################

si_X = si.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'v_dif', 'iVB_dif', 'HB_dif', 'rel_z', 'rel_x', 'usg', 'spin'])  # Features
si_y = si['crv_100']
si_numerical_cols = si_X.select_dtypes(exclude=['object']).columns.tolist()

si_X_train, si_X_test, si_y_train, si_y_test = train_test_split(si_X, si_y, test_size=0.2, random_state=50)
si_preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', si_numerical_cols)])
si_model = Pipeline(steps=[('preprocessor', si_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.02,
    max_depth = 3,
    subsample = 0.75,
    min_child_weight = 9,
    colsample_bytree = 0.7,
    gamma = 0.1,
    reg_lambda = 0.5,
    reg_alpha = 0.9,
    random_state = 50))])

si_model.fit(si_X_train, si_y_train)
si_y_pred = si_model.predict(si_X_test)
si_r2 = r2_score(si_y_test, si_y_pred)
#print(si_r2)

si_importances = si_model.named_steps['regressor'].feature_importances_
si_feature_names = si_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(si_numerical_cols).tolist()

si_feature_importance_df = pd.DataFrame({'Feature': si_feature_names, 'Importance': si_importances})
si_feature_importance_df = si_feature_importance_df.sort_values(by='Importance', ascending=False)
#print(si_feature_importance_df) 

si_24['xrv'] = si_model.predict(si_24)
si_24['stuff'] = (si_24['xrv'])/(abs(si_24['xrv'].mean())) * 100
si_24['z'] = (si_24['xrv'] - si_24['xrv'].mean())/si_24['xrv'].std()
si_24['stuff'] = (si_24['z'] * 10) + 100
#print(si_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30)) 

################# Cutters and Sliders #####################

fc_X = fc.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'rel_z', 'rel_x', 'usg'])  # Features
fc_y = fc['crv_100']
fc_numerical_cols = fc_X.select_dtypes(exclude=['object']).columns.tolist()

fc_X_train, fc_X_test, fc_y_train, fc_y_test = train_test_split(fc_X, fc_y, test_size=0.2, random_state=60)
fc_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', fc_numerical_cols) # Keep numerical columns as is
    ]
)
fc_model = Pipeline(steps=[('preprocessor', fc_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 80,
    learning_rate = 0.02,
    max_depth = 3,
    subsample = 0.75,
    min_child_weight = 9,
    colsample_bytree = 0.75,
    gamma = 0.1,
    reg_lambda = 0.9,
    reg_alpha = 0.8,
    random_state = 60))])

fc_model.fit(fc_X_train, fc_y_train)
fc_y_pred = fc_model.predict(fc_X_test)
fc_r2 = r2_score(fc_y_test, fc_y_pred)
#print(fc_r2)

fc_importances = fc_model.named_steps['regressor'].feature_importances_
fc_feature_names = fc_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(fc_numerical_cols).tolist()

fc_feature_importance_df = pd.DataFrame({'Feature': fc_feature_names, 'Importance': fc_importances})
fc_feature_importance_df = fc_feature_importance_df.sort_values(by='Importance', ascending=False)
#print(fc_feature_importance_df) 

fc_24['xrv'] = fc_model.predict(fc_24)
fc_24['stuff'] = (fc_24['xrv'])/(abs(fc_24['xrv'].mean())) * 100
fc_24['z'] = (fc_24['xrv'] - fc_24['xrv'].mean())/fc_24['xrv'].std()
fc_24['stuff'] = (fc_24['z'] * 10) + 100
#print(fc_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30)) 

######################### SWEEPERS #######################

st_X = st.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'rel_z', 'rel_x', 'usg'])  # Features
st_y = st['crv_100']
st_numerical_cols = st_X.select_dtypes(exclude=['object']).columns.tolist()

st_X_train, st_X_test, st_y_train, st_y_test = train_test_split(st_X, st_y, test_size=0.2, random_state=70)
st_preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', st_numerical_cols)])
st_model = Pipeline(steps=[('preprocessor', st_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.02,
    max_depth = 3,
    subsample = 0.75,
    min_child_weight = 3,
    colsample_bytree = 0.75,
    gamma = 0.1,
    reg_lambda = 0.9,
    reg_alpha = 0.8,
    random_state = 70))])

st_model.fit(st_X_train, st_y_train)
st_y_pred = st_model.predict(st_X_test)
st_r2 = r2_score(st_y_test, st_y_pred)
#print(st_r2)

st_importances = st_model.named_steps['regressor'].feature_importances_
st_feature_names = st_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(st_numerical_cols).tolist()

st_feature_importance_df = pd.DataFrame({'Feature': st_feature_names, 'Importance': st_importances})
st_feature_importance_df = st_feature_importance_df.sort_values(by='Importance', ascending=False)
#print(st_feature_importance_df) 

st_24['xrv'] = st_model.predict(st_24)
st_24['pct'] = st_24['xrv'].rank(pct=True) * 100
st_24['z'] = (st_24['xrv'] - st_24['xrv'].mean())/st_24['xrv'].std()
st_24['stuff'] = (st_24['z'] * 10) + 100
#print(st_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30)) 

#################### CHANGEUPS, SPLITTERS, FORKBALLS ###################

ch_X = ch.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'rel_z', 'rel_x', 'usg', 'spin'])  # Features
ch_y = ch['crv_100']
ch_numerical_cols = ch_X.select_dtypes(exclude=['object']).columns.tolist()

ch_X_train, ch_X_test, ch_y_train, ch_y_test = train_test_split(ch_X, ch_y, test_size=0.2, random_state=10)
ch_preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', ch_numerical_cols)])
ch_model = Pipeline(steps=[('preprocessor', ch_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.02,
    max_depth = 3,
    subsample = 0.75,
    min_child_weight = 10,
    colsample_bytree = 0.75,
    gamma = 0.1,
    reg_lambda = 0.9,
    reg_alpha = 0.8,
    random_state = 11))])

ch_model.fit(ch_X_train, ch_y_train)
ch_y_pred = ch_model.predict(ch_X_test)
ch_r2 = r2_score(ch_y_test, ch_y_pred)
#print(ch_r2)

ch_importances = ch_model.named_steps['regressor'].feature_importances_
ch_feature_names = ch_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(ch_numerical_cols).tolist()

ch_feature_importance_df = pd.DataFrame({'Feature': ch_feature_names, 'Importance': ch_importances})
ch_feature_importance_df = ch_feature_importance_df.sort_values(by='Importance', ascending=False)
#print(ch_feature_importance_df) 

ch_24['xrv'] = ch_model.predict(ch_24)
ch_24['pct'] = ch_24['xrv'].rank(pct=True) * 100
ch_24['z'] = (ch_24['xrv'] - ch_24['xrv'].mean())/ch_24['xrv'].std()
ch_24['stuff'] = (ch_24['z'] * 10) + 100
#print(ch_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30))

############# CURVEBALLS, SLURVES, KNUCKLE CURVES, SLOW CURVES

cu_X = cu.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'rel_z', 'rel_x', 'usg'])  # Features
cu_y = cu['crv_100']
cu_numerical_cols = cu_X.select_dtypes(exclude=['object']).columns.tolist()

cu_X_train, cu_X_test, cu_y_train, cu_y_test = train_test_split(cu_X, cu_y, test_size=0.2, random_state=17)
cu_preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', cu_numerical_cols)])
cu_model = Pipeline(steps=[('preprocessor', cu_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.02,
    max_depth = 3,
    subsample = 0.75,
    min_cuild_weight = 12,
    colsample_bytree = 0.75,
    gamma = 0.1,
    reg_lambda = 0.9,
    reg_alpha = 0.8,
    random_state = 17))])

cu_model.fit(cu_X_train, cu_y_train)
cu_y_pred = cu_model.predict(cu_X_test)
cu_r2 = r2_score(cu_y_test, cu_y_pred)
#print(cu_r2)

cu_importances = cu_model.named_steps['regressor'].feature_importances_
cu_feature_names = cu_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(cu_numerical_cols).tolist()

cu_feature_importance_df = pd.DataFrame({'Feature': cu_feature_names, 'Importance': cu_importances})
cu_feature_importance_df = cu_feature_importance_df.sort_values(by='Importance', ascending=False)
#print(cu_feature_importance_df) 

cu_24['xrv'] = cu_model.predict(cu_24)
cu_24['pct'] = cu_24['xrv'].rank(pct=True) * 100
cu_24['z'] = (cu_24['xrv'] - cu_24['xrv'].mean())/cu_24['xrv'].std()
cu_24['stuff'] = (cu_24['z'] * 10) + 100
#print(cu_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30))

# Bind all pitch models
full_stuff = pd.concat([ff_24, si_24, fc_24, st_24, ch_24, cu_24], axis=0)
print(full_stuff.drop(columns=['z', 'pct']).sort_values(by='stuff', ascending=False).head(30))

######################### Full model, not great, heavily overvalues breaking balls #########################

""" full_X = full.drop(columns=['pitcher', 'year', 'player_name', 'pri', 'count', 'p_throws', 'pitch_type', 'crv_100', 'v_dif', 'iVB_dif', 'HB_dif', 'rel_z', 'rel_x', 'usg', 'spin'])  # Features
full_y = full['crv_100']
full_numerical_cols = full_X.select_dtypes(exclude=['object']).columns.tolist()

full_X_train, full_X_test, full_y_train, full_y_test = train_test_split(full_X, full_y, test_size=0.2, random_state=49)
full_preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', full_numerical_cols) # Keep numerical columns as is
    ]
)
full_model = Pipeline(steps=[('preprocessor', full_preprocessor),
    ('regressor', XGBRegressor(
    n_estimators = 100,
    learning_rate = 0.01,
    max_depth = 3,
    subsample = 0.75,
    min_child_weight = 3,
    colsample_bytree = 0.75,
    gamma = 0.1,
    reg_lambda = 0.9,
    reg_alpha = 0.8,
    random_state = 49))])

full_model.fit(full_X_train, full_y_train)
full_y_pred = full_model.predict(full_X_test)
full_r2 = r2_score(full_y_test, full_y_pred)
print(full_r2)

full_importances = full_model.named_steps['regressor'].feature_importances_
full_feature_names = full_model.named_steps['preprocessor'].transformers_[0][1].get_feature_names_out(full_numerical_cols).tolist()

full_feature_importance_df = pd.DataFrame({'Feature': full_feature_names, 'Importance': full_importances})
full_feature_importance_df = full_feature_importance_df.sort_values(by='Importance', ascending=False)
print(full_feature_importance_df)

test_24['xrv'] = full_model.predict(test_24)
test_24['stuff'] = (test_24['xrv'])/(abs(test_24['xrv'].mean())) * 100
test_24['z'] = (test_24['xrv'] - test_24['xrv'].mean())/test_24['xrv'].std()
test_24['stuff'] = (test_24['z'] * 10) + 100
print(test_24.drop(columns=['z']).sort_values(by='xrv', ascending=False).head(30))  """

# Hyper opt parameter tuning, was run for each model

""" def objective(space):
    model = XGBRegressor(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        learning_rate=space['learning_rate'],
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        gamma=space['gamma'],
        min_child_weight=space['min_child_weight'],
        reg_alpha=space['reg_alpha'],
        reg_lambda=space['reg_lambda'],
        objective='reg:squarederror'
    )
    score = -cross_val_score(model, ch_X_train, ch_y_train, cv=3, scoring='neg_mean_squared_error').mean()
    return {'loss': score, 'status': STATUS_OK}

space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1.0),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
    'gamma': hp.uniform('gamma', 0, 0.3),
    'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 10, 1)),
    'reg_alpha': hp.uniform('reg_alpha', 0, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=200,  # Increase if needed
            trials=trials)

print("Best Parameters:", best) """