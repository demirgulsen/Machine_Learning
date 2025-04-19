"""
Bu dosyada maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
oyuncularının maaş tahminleri için farklı değişken kombinasyonları denenerek çeşitli modeller test edilmiştir.

Amaç: En etkili değişken grubunu ve modeli belirlemek.

"""
# Başlıyoruz...

import numpy as np
import warnings
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.exceptions import ConvergenceWarning
from itertools import combinations

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = int(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = int(up_limit)
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
def data_scaler(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    scaler = StandardScaler()
    num_cols = [col for col in num_cols if col not in 'Salary']
    X_scaled = scaler.fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)
    return df
def data_prep(dataframe):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    # Aykırı değerleri temizleyelim
    for col in num_cols:
        replace_with_thresholds(dataframe, col)
    # Eksik değerleri silelim
    dataframe.dropna(inplace=True)
    return dataframe
def base_model_result(X, y):
    models = [('LR', LinearRegression()),
              ("Ridge", Ridge()),
              ("Lasso", Lasso()),
              ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor(verbosity=-1)),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")
def rf_model_result(X, y):
    rf_model = RandomForestRegressor(random_state=17)
    rf_params = {"max_depth": [5, 8, 10, 12],
                 "max_features": [12, 15, 20],
                 "min_samples_split": [2, 3, 5],
                 "n_estimators": [100, 200, 300, 400]}

    rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
    rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"Random Forest - RMSE: {round(rmse, 4)}")
    return rf_final

def gbm_model_results(X, y):
    gbm_model = GradientBoostingRegressor(random_state=17)
    gbm_params = {"learning_rate": [0.01, 0.1],
                  "max_depth": [3, 8],
                  "n_estimators": [500, 1000],
                  "subsample": [1, 0.5, 0.7]}

    gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
    rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"GBM - RMSE: {round(rmse, 4)}")
    return gbm_final

def add_cross_combination_salary_features(df, imp_cols, low_corr_cols, target='Salary'):
    new_columns = []
    for imp in imp_cols:
        for r in [1, 2]:  # 1'li ve 2'li kombinasyonlar
            for cols in combinations(low_corr_cols, r):
                comb_cols = [imp] + list(cols)
                new_col_name = '_'.join(comb_cols) + f'_{target}_mean'
                salary_mean = df.groupby(comb_cols)[target].transform('mean')
                new_columns.append(salary_mean.rename(new_col_name))

    df = pd.concat([df] + new_columns, axis=1)
    return df

def add_salary_groupby_features(df, base_cols, target='Salary', comb_sizes=[2, 3]):
    new_cols = []

    for r in comb_sizes:
        for cols in combinations(base_cols, r):
            comb_name = '_'.join(cols) + f'_{target}_mean'
            df[comb_name] = df.groupby(list(cols))[target].transform('mean')
            new_cols.append(comb_name)

    return df, new_cols

##########################################################################################################################################

hitters = pd.read_csv('Github_Machine_Learning/Bonus_1/dataset/hitters.csv')
df = data_prep(hitters)

##########################################################################################################################################
### Step 1  #############################################################################################################################
# Kendi oluşturduğumuz değişkenlerin model başarısına etkisini kontrol edelim
def step_1_create_feature(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    new_num_cols=[col for col in num_cols if col!="Salary"]
    df[new_num_cols]=df[new_num_cols]+0.0000000001

    df['NEW_Hits'] = df['Hits'] / df['CHits'] + df['Hits']
    df['NEW_RBI'] = df['RBI'] / df['CRBI']
    df['NEW_Walks'] = df['Walks'] / df['CWalks']
    df['NEW_PutOuts'] = df['PutOuts'] * df['Years']
    df["Hits_Success"] = (df["Hits"] / df["AtBat"]) * 100
    df["NEW_CRBI*CATBAT"] = df['CRBI'] * df['CAtBat']
    df["NEW_RBI"] = df["RBI"] / df["CRBI"]
    df["NEW_Chits"] = df["CHits"] / df["Years"]
    df["NEW_CHmRun"] = df["CHmRun"] * df["Years"]
    df["NEW_CRuns"] = df["CRuns"] / df["Years"]
    df["NEW_Chits"] = df["CHits"] * df["Years"]
    df["NEW_RW"] = df["RBI"] * df["Walks"]
    df["NEW_CH_CB"] = df["CHits"] / df["CAtBat"]
    df["NEW_CHm_CAT"] = df["CHmRun"] / df["CAtBat"]
    df['NEW_Diff_Runs'] = df['Runs'] - (df['CRuns'] / df['Years'])
    df['NEW_Diff_RBI'] = df['RBI'] - (df['CRBI'] / df['Years'])
    df["NEW_RBI_WALK"] = df["RBI"] / df["Walks"]
    df['NEW_Diff_Atbat'] = df['AtBat'] - (df['CAtBat'] / df['Years'])
    df['NEW_Diff_Walks'] = df['Walks'] - (df['CWalks'] / df['Years'])
    df['NEW_Diff_Hits'] = df['Hits'] - (df['CHits'] / df['Years'])
    df['NEW_Diff_HmRun'] = df['HmRun'] - (df['CHmRun'] / df['Years'])

    df['NEW_Career_Offensive_Impact'] = df['CRBI'] + df['CWalks']
    df['TOTAL_Successful_Shot_Number'] = (df['CAtBat'] / df['CHits'])
    df['Successful_Shot_Number'] = (df['AtBat'] / df['Hits'])   # başarılı atış sayısı
    df['Net_Defense_Skill'] = df['CWalks'] - df['Errors']  # Kontrollü Yürüyüş - Hatalar
    df['Most_Valuable_Shot'] = (df['AtBat'] / df['HmRun'])
    df['NEW_ValuableHit_Efficiency'] = (df['Most_Valuable_Shot'] * df['Successful_Shot_Number'])/df['Runs']
    df['NEW_NetContribution_Score'] = df['Errors'] - df['RBI'] + df['Walks']

    return df


df_1 = step_1_create_feature(df)

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df_1)
df_1 = one_hot_encoder(df, cat_cols, drop_first=True)

# Standart Scaler
df_1 = data_scaler(df_1)

# Modelling  *****************
y = df_1["Salary"]
X = df_1.drop(["Salary"], axis=1)

# Base Models
base_model_result(X, y)

# Results
# RMSE: 324610748078.6852 (LR)
# RMSE: 226.8166 (Ridge)
# RMSE: 224.6447 (Lasso)
# RMSE: 257.0942 (ElasticNet)
# RMSE: 257.0775 (KNN)
# RMSE: 278.989 (CART)
# RMSE: 207.3092 (RF)
# RMSE: 398.1319 (SVR)
# RMSE: 214.7398 (GBM)
# RMSE: 224.2335 (XGBoost)
# RMSE: 230.0838 (LightGBM)
# RMSE: 213.2132 (CatBoost)

# Random Forests
rf_final = rf_model_result(X, y)
# RMSE: 208.6642

# GBM Model
gbm_final = gbm_model_results(X, y)
# RMSE: 206.8134

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)





##########################################################################################################################################
### Step 2  #############################################################################################################################
# Düşük korelasyonlu değişkenlerin önemli olduğunu düşündüğümüz değişkenlerle birlikte anlamlı hale gelip gelmediğini” test edelim
def step_2_create_feature(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    numeric_data = pd.DataFrame()
    for feature in num_cols:
        numeric_data[feature] = df[feature]
    corr_data = numeric_data.corr(method='pearson')

    low_corr_features = corr_data[abs(corr_data['Salary']) < 0.5].index.tolist()
    imp_vals = ['Years', 'PutOuts', 'Assists', 'Errors']
    low_corr_features = [col for col in low_corr_features if col not in imp_vals]

    df = add_cross_combination_salary_features(df, imp_vals, low_corr_features, target='Salary')
    return df

df_2 = step_2_create_feature(df)

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df_2)
df_2 = one_hot_encoder(df_2, cat_cols, drop_first=True)

# Standart Scaler
df_2 = data_scaler(df_2)

# Modelling  *****************
y = df_2["Salary"]
X = df_2.drop(["Salary"], axis=1)

# Base Models
base_model_result(X, y)

# Results
# RMSE: 0.0393 (LR)
# RMSE: 0.2465 (Ridge)
# RMSE: 6.3809 (Lasso)
# RMSE: 0.6139 (ElasticNet)
# RMSE: 15.3115 (KNN)
# RMSE: 7.5877 (CART)
# RMSE: 5.4133 (RF)
# RMSE: 361.5344 (SVR)
# RMSE: 13.0256 (GBM)
# RMSE: 15.1069 (XGBoost)
# RMSE: 47.103 (LightGBM)
# RMSE: 8.2915 (CatBoost)

# Random Forests
rf_final = rf_model_result(X, y)
# RMSE: 4.3996

# GBM Model
gbm_final = gbm_model_results(X, y)


# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)




##########################################################################################################################################
### Step 3  ##############################################################################################################################
# Düşük korelasyonlu değişkenlerin önemli olduğunu düşündüğümüz değişkenlerle birlikte anlamlı hale gelip gelmediğini” test edelim.
# Ama bunu 'Salary - x == 0' olanları eleyerek bilgi taşımayan kolonları çıkararak deneyelim.( Bu sayede şişirme değişkenler modele girmez.)
# Yani 'Yeni Değişkenlerin Gerçek Katkısını' kontrol edelim.

def step_3_create_feature(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    numeric_data = pd.DataFrame()
    for feature in num_cols:
        numeric_data[feature] = df[feature]
    corr_data = numeric_data.corr(method='pearson')

    low_corr_features = corr_data[abs(corr_data['Salary']) < 0.5].index.tolist()
    imp_vals = ['Years', 'PutOuts', 'Assists', 'Errors']
    low_corr_features = [col for col in low_corr_features if col not in imp_vals]

    df = add_cross_combination_salary_features(df, imp_vals, low_corr_features, target='Salary')

    new_crt_vals = [col for col in df.columns if '_mean' in col]
    diff_df = df[new_crt_vals].apply(lambda x: df['Salary'] - x)
    del_cols = diff_df.columns[(diff_df == 0.000).all()]
    df = df.drop(columns=del_cols)
    return df

df_3 = step_3_create_feature(df)

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df_3)
df_3 = one_hot_encoder(df_3, cat_cols, drop_first=True)

# Standart Scaler
df_3 = data_scaler(df_3)

# Modelling  *****************
y = df_3["Salary"]
X = df_3.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)


##########################################################################################################################################
### Step 4  ##############################################################################################################################
# 0.45' den büyük olanlar i.in --> good_corr_features
def step_4_good_feature(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    numeric_data = pd.DataFrame()
    for feature in num_cols:
        numeric_data[feature] = df[feature]
    corr_data = numeric_data.corr(method='pearson')

    good_corr_features = corr_data[abs(corr_data['Salary']) > 0.45].index.tolist()
    good_corr_features = [col for col in good_corr_features if 'Salary' not in col]

    good_df, created_cols = add_salary_groupby_features(df, good_corr_features)

    return good_df, good_corr_features

good_df, good_corr_features = step_4_good_feature(df)

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(good_df)
good_df = one_hot_encoder(good_df, cat_cols, drop_first=True)

# Standart Scaler
good_df = data_scaler(good_df)

# Modelling  *****************
y = good_df["Salary"]
X = good_df.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)

##########################################################################################################################################
### Step 5  ##############################################################################################################################
# 0.4 - 0.45 arasında olanlar için --> midi_corr_features

def step_5_midi_feature(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    numeric_data = pd.DataFrame()
    for feature in num_cols:
        numeric_data[feature] = df[feature]
    corr_data = numeric_data.corr(method='pearson')

    midi_corr_features = corr_data[(abs(corr_data['Salary']) > 0.4) & (abs(corr_data['Salary']) < 0.45)].index.tolist()
    midi_df, created_cols = add_salary_groupby_features(df, midi_corr_features)

    return midi_df, midi_corr_features

midi_df, midi_corr_features = step_5_midi_feature(df)

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(midi_df)
midi_df = one_hot_encoder(midi_df, cat_cols, drop_first=True)

# Standart Scaler
midi_df = data_scaler(midi_df)

# Modelling  *****************
y = midi_df["Salary"]
X = midi_df.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)

##########################################################################################################################################
### Step 6  ##############################################################################################################################
# 0.4' den küçük olanlar için --> low_corr_features
def step_6_low_feature(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    numeric_data = pd.DataFrame()
    for feature in num_cols:
        numeric_data[feature] = df[feature]
    corr_data = numeric_data.corr(method='pearson')

    low_corr_features = corr_data[abs(corr_data['Salary']) < 0.4].index.tolist()
    low_df, created_cols = add_salary_groupby_features(df, low_corr_features)

    return low_df, low_corr_features


low_df, low_corr_features = step_6_low_feature(df)

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(low_df)
low_df = one_hot_encoder(low_df, cat_cols, drop_first=True)

# Standart Scaler
low_df = data_scaler(low_df)

# Modelling  *****************
y = low_df["Salary"]
X = low_df.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)


##########################################################################################################################################
### Step 7  ##############################################################################################################################
# Good - Midi Features Combinations
def add_cross_combination_salary_features(df, good_corr_features, midi_corr_features, target='Salary'):
    new_columns = []

    for imp in good_corr_features:
        for r in [1, 2]:  # 1'li ve 2'li kombinasyonlar
            for cols in combinations(midi_corr_features, r):
                comb_cols = [imp] + list(cols)
                new_col_name = '_'.join(comb_cols) + f'_{target}_mean'
                salary_mean = df.groupby(comb_cols)[target].transform('mean')
                new_columns.append(salary_mean.rename(new_col_name))

    df = pd.concat([df] + new_columns, axis=1)
    return df

good_midi_df = add_cross_combination_salary_features(df, good_corr_features, midi_corr_features, target='Salary')

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(good_midi_df)
good_midi_df = one_hot_encoder(good_midi_df, cat_cols, drop_first=True)

# Standart Scaler
good_midi_df = data_scaler(good_midi_df)

# Modelling  *****************
y = good_midi_df["Salary"]
X = good_midi_df.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)


##########################################################################################################################################
### Step 8  ##############################################################################################################################
# Good - Low Features Combinations
def add_cross_combination_salary_features(df, good_corr_features, low_corr_cols, target='Salary'):
    new_columns = []

    for imp in good_corr_features:
        for r in [1, 2]:  # 1'li ve 2'li kombinasyonlar
            for cols in combinations(low_corr_cols, r):
                comb_cols = [imp] + list(cols)
                new_col_name = '_'.join(comb_cols) + f'_{target}_mean'
                salary_mean = df.groupby(comb_cols)[target].transform('mean')
                new_columns.append(salary_mean.rename(new_col_name))

    df = pd.concat([df] + new_columns, axis=1)
    return df

good_low_df = add_cross_combination_salary_features(df, good_corr_features, low_corr_features, target='Salary')

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(good_low_df)
good_low_df = one_hot_encoder(good_low_df, cat_cols, drop_first=True)

# Standart Scaler
good_low_df = data_scaler(good_low_df)

# Modelling  *****************
y = good_low_df["Salary"]
X = good_low_df.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)

##########################################################################################################################################
### Step 9  ##############################################################################################################################
# Midi - Low Features Combinations
def add_cross_combination_salary_features(df, midi_corr_features, low_corr_cols, target='Salary'):
    new_columns = []

    for imp in midi_corr_features:
        for r in [2, 3]:  # 1'li ve 2'li kombinasyonlar
            for cols in combinations(low_corr_cols, r):
                comb_cols = [imp] + list(cols)
                new_col_name = '_'.join(comb_cols) + f'_{target}_mean'
                salary_mean = df.groupby(comb_cols)[target].transform('mean')
                new_columns.append(salary_mean.rename(new_col_name))

    df = pd.concat([df] + new_columns, axis=1)
    return df

midi_low_df = add_cross_combination_salary_features(df, midi_corr_features, low_corr_features, target='Salary')

# Encoding
cat_cols, num_cols, cat_but_car = grab_col_names(midi_low_df)
midi_low_df = one_hot_encoder(midi_low_df, cat_cols, drop_first=True)

# Standart Scaler
midi_low_df = data_scaler(midi_low_df)

# Modelling  *****************
y = midi_low_df["Salary"]
X = midi_low_df.drop(["Salary"], axis=1)


# Base Models
base_model_result(X, y)

# Random Forests
rf_final = rf_model_result(X, y)

# GBM Model
gbm_final = gbm_model_results(X, y)

# Feature Importance
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(rf_final, X, 50)
plot_importance(rf_final, X, 50)
