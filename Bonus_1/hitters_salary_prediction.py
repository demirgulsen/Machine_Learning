###################################################
# PROJECT: SALARY PREDICTİON WITH MACHINE LEARNING
###################################################

# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# : Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

############################################
# Gerekli Kütüphane ve Fonksiyonlar

############################################
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

#############################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
#############################################

# Genel Bakış
df = pd.read_csv('Github_Machine_Learning/Bonus_1/dataset/hitters.csv')

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T

# Numerik ve kategorik değişkenleri yakalayalım.
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

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Kategorik Değişken Analizi (Analysis of Categorical Variables)
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# Sayısal Değişken Analizi (Analysis of Numerical Variables)
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.figure(figsize=(8, 5))
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, True)


# Hedef Değişken Analizi (Analysis of Target Variable)
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"Salary", col)


# Bağımlı değişkenin incelenmesi
df["Salary"].hist(bins=100)

# Korelasyon Analizi (Analysis of Correlation)
numeric_data = pd.DataFrame()

for feature in num_cols:
    numeric_data[feature] = df[feature]

corr_data = numeric_data.corr(method='pearson')

plt.figure(figsize=(30, 30))
sns.heatmap(data= corr_data, cmap='coolwarm', annot=True, fmt='.2g')

#########################
numeric_data['Salary'] = df['Salary']

corr_data = numeric_data.corr(method='pearson')
corr_data = corr_data[['Salary']]

plt.figure(figsize=(7, 10))
sns.heatmap(data=corr_data, cmap='coolwarm', annot=True, fmt='.2g')

##############################################################
# Tüm değişkenlerin dağılımını kontrol edelim
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Sürekli Değişkenlerin Dağılımı", fontsize=14)
plt.show()

#######################################
# Outliers (Aykırı Değerler)

# Aykırı gözlemlerin incelenmesi.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, check_outlier(df, col))

############################
# Aykırı değerleri grafik üzerinde görelim.
for col in num_cols:
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df[num_cols])), df[col])
    plt.axhline(df[col].mean() + 3 * df[col].std(), color='r', linestyle='dashed', label="Üst Sınır")
    plt.axhline(df[col].mean() - 3 * df[col].std(), color='r', linestyle='dashed', label="Alt Sınır")
    plt.legend()
    plt.title(col+" Aykırı Değerlerin Scatter Plot ile Gösterimi")
    plt.show()


# Aykırı değerleri temizleyelim
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = int(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = int(up_limit)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Missing Values (Eksik Değerler)
def missing_values_table(dataframe, na_name=False, na_ratio=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

    if na_ratio:
        return ratio

missing_values_table(df)


df[df.isnull().any(axis=1)]

# Eksik değerleri silelim
df.dropna(inplace=True)

###########################
# Feature Engineering
###########################

# Feature Extraction (Özellik Çıkarımı)

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

##################
df['Avg_Salary_by_PutOuts_RBI_CWalks'] = df.groupby(['PutOuts', 'RBI','CWalks'])['Salary'].transform('mean')
df['Avg_Salary_by_Assists_AtBat_Runs'] = df.groupby(['Assists', 'AtBat', 'Runs'])['Salary'].transform('mean')
df['Avg_Salary_by_Years_Hits_Runs'] = df.groupby(['Years', 'Hits','Runs'])['Salary'].transform('mean')
df['Avg_Salary_by_PutOuts_Hits_Walks'] = df.groupby(['PutOuts', 'Hits','Walks'])['Salary'].transform('mean')

df['Avg_Salary_by_PutOuts_HmRun_RBI'] = df.groupby(['PutOuts', 'HmRun','RBI'])['Salary'].transform('mean')
df['Avg_Salary_by_Years_RBI'] = df.groupby(['Years', 'RBI'])['Salary'].transform('mean')
df['Avg_Salary_by_PutOuts_Assists_Years'] = df.groupby(['PutOuts', 'Assists', 'Years'])['Salary'].transform('mean')

df['Avg_Salary_by_League_Division_Years'] = df.groupby(['League', 'Division', 'Years'])['Salary'].transform('mean')
df['Avg_Salary_by_PutOuts_Assists'] = df.groupby(['PutOuts', 'Assists'])['Salary'].transform('mean')
df['Avg_Salary_by_PutOuts_Years'] = df.groupby(['PutOuts', 'Years'])['Salary'].transform('mean')
df['Avg_Salary_by_Assists_Years'] = df.groupby(['Assists', 'Years'])['Salary'].transform('mean')

####################################################################
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df = one_hot_encoder(df, cat_cols, drop_first=True)


# Feature Scaling (Özellik Ölçeklendirme)

# Standart Scaler
cat_cols, num_cols, cat_but_car = grab_col_names(df)

scaler = StandardScaler()
num_cols = [col for col in num_cols if col not in 'Salary']
X_scaled = scaler.fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)

df.head()

#############################################
# Base Models
#############################################

y = df["Salary"]
X = df.drop(["Salary"], axis=1)

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


# En düşük sonucu veren modelleri seçerek hiperparametre optimizasyonu yapalım ve rmse değerlerine tekrar bakalım
# Lasso, RF, GBM, XGBoost modelleri en düşük değerleri döndürdü ama biz ağaca dayalı modeller üzerinden gidelim

################################################
# Random Forests
################################################

rf_model = RandomForestRegressor(random_state=17)

rf_params = {"max_depth": [5,8,10,12],
             "max_features": [12,15,20],
             "min_samples_split": [2, 3, 5],
             "n_estimators": [100, 200, 300,400]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X, y, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)}")
# RMSE: 11.2935


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

################################################
# GBM Model
################################################

gbm_model = GradientBoostingRegressor(random_state=17)

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(gbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse

# 7.846640234574591
################################################
# LightGBM
################################################

lgbm_model = LGBMRegressor(random_state=17)

lgbm_params = {"learning_rate": [0.01, 0.1],
                "n_estimators": [300, 500],
                "colsample_bytree": [0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=-1).fit(X, y)
lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse
# 115.3843948674415

################################################
# BONUS : Sonuçları bir de train-test verileri ile görelim
################################################
y = df["Salary"]
X = df.drop(["Salary"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=46)

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


#########
rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_train, y_train)
rmse = np.mean(np.sqrt(-cross_val_score(rf_final, X_train, y_train, cv=10, scoring="neg_mean_squared_error")))
print(f"RMSE: {round(rmse, 4)}")

y_pred = rf_final.predict(X_test)
# RMSE: 22.7373

# ---------
new_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE: ', new_rmse)

# ---------
def prediction_comparison_table(y, y_pred):
    comparison_df = pd.DataFrame({
        'Gerçek Değer (y)': y,
        'Tahmin Edilen Değer (y_pred)': y_pred,
        'Hata (y - y_pred)': y - y_pred
    })
    return comparison_df.head(20)  # İlk 10 satırı göster


def plot_barplot_comparison(y, y_pred, n=20):
    df_bar = pd.DataFrame({
        'Gerçek Değer': y[:n].reset_index(drop=True),
        'Tahmin Değer': y_pred[:n] if isinstance(y_pred, pd.Series) else pd.Series(y_pred[:n])
    })

    df_bar = df_bar.reset_index().melt(id_vars='index', var_name='Tür', value_name='Değer')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='index', y='Değer', hue='Tür', data=df_bar, palette='Set2')
    plt.title(f'İlk {n} Gözlem İçin Gerçek vs Tahmin Değerleri')
    plt.xlabel('Gözlem')
    plt.ylabel('Değer')
    plt.grid(True)
    plt.show()


plot_barplot_comparison(y_test, y_pred, n=20)

# Tabloyu gör
prediction_comparison_table(y_test.reset_index(drop=True), pd.Series(y_pred))


plot_importance(rf_final, X,50)

################################################
# CatBoost
################################################

catboost_model = CatBoostRegressor(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_final, X, y, cv=10, scoring="neg_mean_squared_error")))
rmse


######################################################
#  Automated Hyperparameter Optimization
######################################################

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8],
              "n_estimators": [500, 1000],
              "subsample": [1, 0.5, 0.7]}


lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


regressors = [("RF", RandomForestRegressor(), rf_params),
              ('GBM', GradientBoostingRegressor(), gbm_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params),
              ("CatBoost", CatBoostRegressor(), catboost_params)]


best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model


################################################
# Feature Importance
################################################

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


plot_importance(rf_final, X)
plot_importance(gbm_final, X)
plot_importance(lgbm_final, X)
plot_importance(catboost_final, X)


################################################
# Analyzing Model Complexity with Learning Curves
# ################################################

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()


rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
                 ["max_features", [3, 5, 7, "auto"]],
                 ["min_samples_split", [2, 5, 8, 15, 20]],
                 ["n_estimators", [10, 50, 100, 200, 500]]]


rf_model = RandomForestRegressor(random_state=17)

for i in range(len(rf_val_params)):
    val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1],scoring="neg_mean_absolute_error")

rf_val_params[0][1]
