################################################################
# House Price Prediction Model - Ev Fiyat Tahmin Modeli
################################################################

###############
# İş Problemi
###############
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
# farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
# gerçekleştirilmek istenmektedir.

######################
# Veri Seti Hikayesi
######################
# Ames, Lowa’daki konut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor. Kaggle üzerinde bir yarışması
# da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz. Veri seti bir kaggle yarışmasına ait
# olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır. Test veri setinde ev fiyatları boş bırakılmış olup, bu
# değerleri sizin tahmin etmeniz beklenmektedir.
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

##############################
# Toplam Gözlem : 1460
# Sayısal Değişken : 38
# Kategorik Değişken : 43
##############################

# Görev
# Elimizdeki veri seti üzerinden minimum hata ile ev fiyatlarını tahmin eden bir makine öğrenmesi modeli geliştiriniz ve kaggle yarışmasına tahminlerinizi yükleyiniz.
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

##############################
# Görev 1: Keşifçi Veri Analizi
##############################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import itertools
from scipy.stats import skew

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Adım 1: Train ve Test veri setlerini okutup birleştirelim ve birleştirdiğimiz veri üzerinden ilerleyelim
train_df = pd.read_csv('Github_Machine_Learning/CaseStudy_1/datasets/train.csv')
test_df = pd.read_csv('Github_Machine_Learning/CaseStudy_1/datasets/test.csv')

df = pd.concat([train_df, test_df], ignore_index=False).reset_index(drop=True)

def dataframe_info(df):
    print("***** HEAD **********************************")
    print(df.head())
    print("***** TAIL **********************************")
    print(df.tail())
    print("****** SHAPE *********************************")
    print(df.shape)
    print("****** COLUMNS *********************************")
    print(df.columns)
    print("****** INFO *********************************")
    print(df.info())
    print("****** NA COLUMNS COUNT*********************************")
    print(df.isnull().sum())
    print("******* DESCRIBE ********************************")
    print(df.describe().T)

dataframe_info(df)

# Adım 2: Gerekli düzenlemeleri yapalım. (Tip hatası olan değişkenler gibi)
df["MSSubClass"] = df["MSSubClass"].astype(str)
df["YrSold"] = df["YrSold"].astype(str)
df["MoSold"] = df["MoSold"].astype(str)


# Adım 3: Numerik ve kategorik değişkenleri yakalayalım.
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
num_cols = [col for col in num_cols if col not in "Id"]


# Adım 4: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyelim

# Kategorik Değişken Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

############################
 # Sayısal Değişken Analizi
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


# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapalım.
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"SalePrice", col)


# Bağımlı değişkenin incelenmesi
df["SalePrice"].hist(bins=100)

# Adım 6: Aykırı gözlem var mı inceleyelim.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
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
    if col != "SalePrice":
        print(col, check_outlier(df, col))


# Aykırı değerleri grafik üzerinde görelim.
for col in num_cols:
    plt.figure(figsize=(8, 5))
    plt.scatter(range(len(df[num_cols])), df[col])
    plt.axhline(df[col].mean() + 3 * df[col].std(), color='r', linestyle='dashed', label="Üst Sınır")
    plt.axhline(df[col].mean() - 3 * df[col].std(), color='r', linestyle='dashed', label="Alt Sınır")
    plt.legend()
    plt.title(col+"Aykırı Değerlerin Scatter Plot ile Gösterimi")
    plt.show()


# Adım 7: Eksik gözlem var mı inceleyiniz.
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

# Eksik değerlerin bağımlı değişkenle ilişkisini inceleyelim.
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


na_cols = missing_values_table(df, True)
missing_vs_target(df, "SalePrice", na_cols)


# Kategorik değişkenlerin dağılımını inceleyelim.
num_plots = min(len(cat_cols), 9 * 5)  # Toplam 45 plot'tan fazlaysa sınır koy

fig, axes = plt.subplots(nrows=9, ncols=5, figsize=(30,30), constrained_layout=True)
axes = axes.flatten()  # 2D array yerine 1D array olarak kullan

for i, feature in enumerate(cat_cols[:num_plots]):  # Fazla kategorileri almamak için sınır koy
    sns.histplot(data=train_df, x=feature, ax=axes[i], color='dodgerblue')   # TRAIN DATA
    sns.histplot(data=test_df, x=feature, ax=axes[i], color='crimson') # TEST DATA


##################################
# Görev 2: Feature Engineering
##################################
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapalım.

# Boş değerleri YearBuilt değerleriyle dolduralım ve int tipine dönüştürelim.
df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt']).astype(int)


# Nümerik sütunların çarpıklık değerlerini hesaplayalım ve çarpıklık değeri 1'den büyük olanları silelim
skew_cols = df[num_cols].skew().sort_values(ascending=False)
skew_cols_over_10 = skew_cols[(skew_cols.values > 1.5) &
                              (skew_cols.index != 'SalePrice')]


# %50 üzerinde eksik değer içeren sütunları silelim.
na_cols = missing_values_table(df, False, True)
na_cols_over_10 = na_cols[(na_cols.values > 10) & (na_cols.index != 'SalePrice')]


# Baskın etikete sahip değişkenleri silelim
threshold = 0.95  # 85
dominant_ratio = df.apply(lambda col: col.value_counts(normalize=True).max())
dominant_cols = dominant_ratio[dominant_ratio > threshold].index.tolist()
print(dominant_ratio[dominant_ratio > threshold].sort_values(ascending=False))


drop_list = (set(na_cols_over_10.index.tolist()) |
             # set(skew_cols_over_10.index.tolist()) |
             set(dominant_cols))


len(df.columns)
df.drop(drop_list, axis=1, inplace=True)
len(df.columns)


# Bazı değişkenlerdeki null değerler değişkenlerin o özelliğe sahip olmadıkları anlamına gelir. O yüzden onları "No" ile dolduralım.
null_cols = ['GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1']

for col in null_cols:
    df[col] = df[col].fillna('No')


# Neighborhood" ve "YearBuilt" değişkenlerine göre ortalama satış fiyatlarını hesaplayarak yeni değişkenler oluşturalım.
df['AvgPrice_neighborhood'] = df.groupby("Neighborhood")['SalePrice'].transform('mean')
df['AvgPrice_YearBuilt'] = df.groupby("YearBuilt")['SalePrice'].transform('mean')

# 2ndFlrSF sütunundaki değeri 0 olan tüm satırları (0 baskın değer) NaN ile değiştirelim.
df.loc[df["2ndFlrSF"] == 0, "2ndFlrSF"] = np.nan


# Null değer içeren kategorik ve nümerik değişkenleri alalım ve kategorik değişkenleri mode ile nümerik değişkenleri mean ile dolduralım.
na_cols = missing_values_table(df, True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

na_and_cat_cols = [col for col in df[na_cols] if col in cat_cols]
na_and_num_cols = [col for col in df[na_cols] if col in num_cols and col != 'SalePrice']

for col in na_and_num_cols:
    df[col] = df[col].fillna(df[col].mean())


for col in na_and_cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Eksik değer içeren değişken kalmış mı kontrol edelim.(Sadece SalePrice kalması beklenir)
missing_values_table(df, True)


# Aykırı değerleri temizleyelim
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = int(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = int(up_limit)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))


# Adım 2: Rare Encoder uygulayalım.
# Rare Analizi yapalım
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

df = rare_encoder(df, 0.01)

# Rare kategoriler ile bağımlı değişken arasındaki ilişkiyi analiz edelim.
rare_analyser(df, "SalePrice", cat_cols)


##########  SİL #######################################################################################################################################################
#################################################################################################################################################################
######### DENEME ########## DENEME ####################################################################################################################
# feature_imp = pd.DataFrame({'Value': lgbm_final_model.feature_importances_, 'Feature': X.columns})
# added_features = feature_imp.sort_values(by="Value", ascending=False)[0:50]
# added_features = [col for col in added_features['Feature'].tolist() if col not in drop_list]
# best_features = list(added_features)
# best_features = [col for col in best_features if col in df.columns]

best_features = ['LotArea', 'BsmtUnfSF', 'YearRemodAdd', 'BsmtFinSF1', 'TotalBsmtSF',
                 'MasVnrArea', 'GarageArea', '1stFlrSF', 'OpenPorchSF', 'OverallCond',
                 'WoodDeckSF', 'GarageYrBlt', 'GrLivArea', 'YearBuilt', 'OverallQual',
                 '2ndFlrSF', 'EnclosedPorch', 'Fireplaces', 'BsmtFullBath', 'GarageCars']

cat_cols, num_cols, cat_but_car = grab_col_names(df)

best_cat_features = [bf for bf in best_features if bf in cat_cols]
best_num_features = [bf for bf in best_features if bf in num_cols]

from itertools import combinations

# 1
df[f"{'OverallQual'}_{'OverallCond'}_{'GarageCars'}_mean"] = df[["OverallQual", "OverallCond", "GarageCars"]].mean(axis=1)
df[f"{'GrLivArea'}_{'BsmtFinSF1'}_{'TotalBsmtSF'}_mean"] = df[['GrLivArea', 'BsmtFinSF1', 'TotalBsmtSF']].mean(axis=1)
df[f"{'GrLivArea'}_{'GarageArea'}_{'TotalBsmtSF'}_mean"] = df[['GrLivArea', 'GarageArea', 'TotalBsmtSF']].mean(axis=1)
df[f"{'GrLivArea'}_{'BsmtUnfSF'}_{'TotalBsmtSF'}_mean"] = df[['GrLivArea', 'BsmtUnfSF', 'TotalBsmtSF']].mean(axis=1)
df[f"{'YearBuilt'}_{'OpenPorchSF'}_{'EnclosedPorch'}_mean"] = df[['YearBuilt', 'OpenPorchSF', 'EnclosedPorch']].mean(axis=1)
df[f"{'1stFlrSF'}_{'GrLivArea'}_{'WoodDeckSF'}_mean"] = df[['1stFlrSF', 'GrLivArea', 'WoodDeckSF']].mean(axis=1)
df[f"{'LotArea'}_{'GrLivArea'}_{'BsmtUnfSF'}_mean"] = df[['LotArea', 'GrLivArea', 'BsmtUnfSF']].mean(axis=1)
df[f"{'GarageArea'}_{'OpenPorchSF'}_{'WoodDeckSF'}_mean"] = df[['GarageArea', 'OpenPorchSF', 'WoodDeckSF']].mean(axis=1)
df[f"{'OverallQual'}_{'Fireplaces'}_{'GarageCars'}_mean"] = df[['OverallQual', 'Fireplaces', 'GarageCars']].mean(axis=1)


df[f"{'GrLivArea'}_{'OverallQual'}_{'OverallCond'}_prod"] = df['GrLivArea'] * df['OverallQual'] * df['OverallCond']
df[f"{'1stFlrSF'}_{'GrLivArea'}_{'OverallCond'}_prod"] = df['1stFlrSF'] * df['GrLivArea'] * df['OverallCond']
df[f"{'GrLivArea'}_{'TotalBsmtSF'}_{'OverallQual'}_prod"] = df['GrLivArea'] * df['TotalBsmtSF'] * df['OverallQual']
df[f"{'YearBuilt'}_{'YearRemodAdd'}_{'OverallCond'}_prod"] = df['YearBuilt'] * df['YearRemodAdd'] * df['OverallCond']
df[f"{'OverallQual'}_{'OverallCond'}_{'2ndFlrSF'}_prod"] = df['OverallQual'] * df['OverallCond'] * df['2ndFlrSF']
df[f"{'GarageYrBlt'}_{'YearRemodAdd'}_{'OverallCond'}_prod"] = df['GarageYrBlt'] * df['YearRemodAdd'] * df['OverallCond']
df[f"{'GarageYrBlt'}_{'YearBuilt'}_{'YearRemodAdd'}_prod"] = df['GarageYrBlt'] * df['YearBuilt'] * df['YearRemodAdd']

df[f"{'1stFlrSF'}_times_{'OverallQual'}"] = df['1stFlrSF'] * df['OverallQual']
df[f"{'GrLivArea'}_times_{'OverallQual'}"] = df['GrLivArea'] * df['OverallQual']
df[f"{'GrLivArea'}_times_{'OverallQual'}_times_{'TotalBsmtSF'}"] = df['GrLivArea'] * df['OverallQual'] * df['TotalBsmtSF']
df[f"{'LotArea'}_times_{'OverallQual'}"] = df['LotArea'] * df['OverallQual']
df[f"{'GarageYrBlt'}_times_{'OverallQual'}"] = df['GarageYrBlt'] * df['OverallQual']
df[f"{'GarageYrBlt'}_times_{'YearRemodAdd'}_times_{'OverallQual'}"] = df['GarageYrBlt'] * df['YearRemodAdd']* df['OverallQual']
df[f"{'YearRemodAdd'}_times_{'OverallQual'}"] = df['YearRemodAdd'] * df['OverallQual']
df[f"{'LotArea'}_times_{'GarageArea'}"] = df['LotArea'] * df['GarageArea']
df[f"{'YearBuilt'}_times_{'OverallQual'}"] = df['YearBuilt'] * df['OverallQual']

df[f"{'GrLivArea'}_plus_{'GarageArea'}"] = df['GrLivArea'] + df['GarageArea']
df[f"{'1stFlrSF'}_plus_{'GarageArea'}"] = df['1stFlrSF'] + df['GarageArea']
df[f"{'GrLivArea'}_plus_{'BsmtFinSF1'}"] = df['GrLivArea'] + df['BsmtFinSF1']

df[f"{'GarageYrBlt'}_minus_{'OpenPorchSF'}"] = df['GarageYrBlt'] - df['OpenPorchSF']

df[f"{'GrLivArea'}_div_{'OverallQual'}"] = df['GrLivArea'] / (df['OverallQual'] + 1e-5)
df[f"{'TotalBsmtSF'}_div_{'OpenPorchSF'}"] = df['TotalBsmtSF'] / (df['OpenPorchSF'] + 1e-5)
df[f"{'GarageYrBlt'}_div_{'YearBuilt'}_div_{'YearRemodAdd'}"] = df['GarageYrBlt'] / (df['YearBuilt'] + 1e-5) / (df['YearRemodAdd'] + 1e-5)


#
# # DENENDİ
# ##############################################################################
# # Asıl numerik ve kategorik değişkneler
# num_features = [f for f in best_features if df[f].dtypes != 'O']
# def generate_pair_triple_combinations(df, cols, epsilon=1e-5, methods=["mean", "prod","div"]):
#     new_features = pd.DataFrame(index=df.index)
#
#     for col1, col2, col3 in combinations(cols, 3):
#         if "mean" in methods:
#             new_features[f"{col1}_{col2}_{col3}_mean"] = df[[col1, col2, col3]].mean(axis=1)
#         if "prod" in methods:
#             new_features[f"{col1}_{col2}_{col3}_prod"] = df[col1] * df[col2] * df[col3]
#         if "div" in methods:
#             new_features[f"{col1}_div_{col2}_div_{col3}"] = df[col1] / (df[col2] + epsilon) / (df[col3] + epsilon)
#             new_features[f"{col3}_div_{col2}_div_{col1}"] = df[col3] / (df[col2] + epsilon) / (df[col1] + epsilon)
#
#     return new_features
#
# new_features = generate_pair_triple_combinations(df, num_features)
# df_1 = pd.concat([df, new_features], axis=1)
#
#
# # 2 DENENDİ
# ####################################################################################################################
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# numeric_data = pd.DataFrame()
# for feature in num_cols:
#     numeric_data[feature] = df[feature]
# corr_data = numeric_data.corr(method='pearson')
#
# low_corr_features = corr_data[abs(corr_data['SalePrice']) < 0.5].index.tolist()
# low_corr_features = [col for col in low_corr_features if 'Id' not in col]
# midi_corr_features = corr_data[(abs(corr_data['SalePrice']) > 0.5) & (abs(corr_data['SalePrice']) < 0.6)].index.tolist()
# good_corr_features = corr_data[abs(corr_data['SalePrice']) > 0.6].index.tolist()
# good_corr_features = [col for col in good_corr_features if 'SalePrice' not in col]
#
# best_features = ['LotArea', 'BsmtUnfSF', 'YearRemodAdd', 'BsmtFinSF1', 'TotalBsmtSF',
#                  'MasVnrArea', 'GarageArea', '1stFlrSF', 'OpenPorchSF', 'OverallCond',
#                  'WoodDeckSF', 'GarageYrBlt', 'GrLivArea', 'YearBuilt', 'OverallQual',
#                  '2ndFlrSF', 'EnclosedPorch', 'Fireplaces', 'BsmtFullBath', 'GarageCars']
#
# cat_cols, num_cols, cat_but_car = grab_col_names(df)
#
# best_cat_features = [bf for bf in best_features if bf in cat_cols]
# best_num_features = [bf for bf in best_features if bf in num_cols]
#
# best_corr_features = [col for col in best_num_features if col not in good_corr_features]
#
# # def add_cross_combination_salary_features(df, imp_cols, low_corr_features, target='SalePrice'):
# #     new_columns = []
# #     imp_cols
# #     for imp in imp_cols:
# #         for r in [2, 3]:  # Sadece 2'li ve 3'lü kombinasyonlar
# #             for cols in combinations(low_corr_features, r):
# #                 comb_cols = [imp] + list(cols)
# #                 new_col_name = '_'.join(comb_cols) + f'_{target}_mean'
# #                 salary_mean = df.groupby(comb_cols)[target].transform('mean')
# #                 new_columns.append(salary_mean.rename(new_col_name))
# #
# #     df = pd.concat([df] + new_columns, axis=1)
# #     return df
#
# from itertools import combinations
#
# def add_cross_combination_salary_features(df, imp_cols, low_corr_features, target='SalePrice'):
#     new_columns = []
#
#     for imp in imp_cols:
#         for cols in combinations(low_corr_features, 2):  # low_corr'den 2 sütun seç
#             comb_cols = [imp] + list(cols)  # imp + 2 low_corr = 3'lü kombinasyon
#             new_col_name = '_'.join(comb_cols) + f'_{target}_mean'
#             salary_mean = df.groupby(comb_cols)[target].transform('mean')
#             new_columns.append(salary_mean.rename(new_col_name))
#
#     df = pd.concat([df] + new_columns, axis=1)
#     return df
#
#
# df_2 = add_cross_combination_salary_features(df, best_corr_features, good_corr_features,  target='SalePrice')
#
# df_2 = add_cross_combination_salary_features(df, good_corr_features, midi_corr_features, target='SalePrice')
#
# df_2 = add_cross_combination_salary_features(df, good_corr_features, low_corr_features, target='SalePrice')
#
# df_2 = add_cross_combination_salary_features(df, midi_corr_features, low_corr_features, target='SalePrice')
#
#
# # 3 - DENENDİ
# ###############################################################################
# from itertools import combinations
#
# def generate_numeric_combinations_3way(df, cols, methods=["sub", "mul", "div"], epsilon=1e-5):
#     """
#     Sayısal değişkenler için 3'lü kombinasyonlarla yeni matematiksel özellikler üretir.
#
#     Parameters:
#     df (pd.DataFrame): Orijinal veri seti
#     cols (list): Sayısal değişken adları
#     methods (list): Uygulanacak işlemler: "add", "sub", "mul", "div"
#     epsilon (float): Bölme işlemlerinde sıfıra bölmeyi engellemek için
#
#     Returns:
#     pd.DataFrame: Yeni oluşturulmuş kombinasyonları içeren DataFrame
#     """
#     combo_df = pd.DataFrame(index=df.index)
#
#     for col1, col2, col3 in combinations(cols, 3):
#
#         if "sub" in methods:
#             combo_df[f"{col1}_minus_{col2}_minus_{col3}"] = df[col1] - df[col2] - df[col3]
#             combo_df[f"{col3}_minus_{col2}_minus_{col1}"] = df[col3] - df[col2] - df[col1]
#         if "mul" in methods:
#             combo_df[f"{col1}_times_{col2}_times_{col3}"] = df[col1] * df[col2] * df[col3]
#         if "div" in methods:
#             combo_df[f"{col1}_div_{col2}_div_{col3}"] = df[col1] / (df[col2] + epsilon) / (df[col3] + epsilon)
#             combo_df[f"{col3}_div_{col2}_div_{col1}"] = df[col3] / (df[col2] + epsilon) / (df[col1] + epsilon)
#
#     return combo_df
#
# new_numeric_features_3way = generate_numeric_combinations_3way(df, best_num_features)
#
# df_3 = pd.concat([df, new_numeric_features_3way], axis=1)
#
#
# # 4  - DENENDİ
# #############################################################################
# # Nümerik Değişkenler
# def generate_numeric_combinations(df, cols, methods=["add", "sub", "mul", "div"], epsilon=1e-5):
#     """
#     Sayısal değişkenler için çeşitli matematiksel kombinasyonlarla yeni değişkenler üretir.
#
#     Parameters:
#     df (pd.DataFrame): Orijinal veri seti
#     cols (list): Sayısal değişken adları
#     methods (list): Uygulanacak işlemler: "add", "sub", "mul", "div"
#     epsilon (float): Bölme işlemlerinde sıfıra bölmeyi engellemek için
#
#     Returns:
#     pd.DataFrame: Yeni oluşturulmuş kombinasyonları içeren DataFrame
#     """
#     combo_df = pd.DataFrame(index=df.index)
#
#     for col1, col2 in combinations(cols, 2):
#         if "add" in methods:
#             combo_df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]
#         if "sub" in methods:
#             combo_df[f"{col1}_minus_{col2}"] = df[col1] - df[col2]
#             combo_df[f"{col2}_minus_{col1}"] = df[col2] - df[col1]
#         if "mul" in methods:
#             combo_df[f"{col1}_times_{col2}"] = df[col1] * df[col2]
#         if "div" in methods:
#             combo_df[f"{col1}_div_{col2}"] = df[col1] / (df[col2] + epsilon)
#             combo_df[f"{col2}_div_{col1}"] = df[col2] / (df[col1] + epsilon)
#
#     return combo_df
#
#
# new_numeric_features = generate_numeric_combinations(df, best_num_features)
# df_4 = pd.concat([df, new_numeric_features], axis=1)

##########################################################################################################################################
##########################################################################################################################################################
########################################################################################################################################################################

# Adım 3: Yeni değişkenler oluşturalım.
df['GarageEfficiency'] = df['GarageArea'] / (df['GarageCars'] + 1)   # +1 TO AVOID DIVISION BY ZERO
df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]
# ****
df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF # 35

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])
df['AvgPrice_neighborhood'] = df.groupby("Neighborhood")['SalePrice'].transform('mean')
df['AvgPrice_YearBuilt'] = df.groupby("YearBuilt")['SalePrice'].transform('mean')

# Dene
df["Bathrooms_Total"] = df["FullBath"] + df["HalfBath"] + df["BsmtFullBath"]
df["Bed_Bath_Ratio"] = df["BedroomAbvGr"] / (df["Bathrooms_Total"] + 1e-3)
df["Garage_Per_Bedroom"] = df["GarageCars"] / (df["BedroomAbvGr"] + 1e-3)
df["OverallCond_CentralAir"] = df["OverallCond"].astype(str) + "_" + df["CentralAir"]
df["Comfort_Score"] = df["Fireplaces"] + df["GarageCars"] + df["CentralAir"].apply(lambda x: 1 if x == 'Y' else 0)

############################################################
# Bu değişkenler de oluşturulabilir fakat model başarısına çok fazla etkisi olmadı.
# df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
# df["TotalBath"] = df["FullBath"] + (0.5 * df["HalfBath"]) + df["BsmtFullBath"] + (0.5 * df["BsmtHalfBath"])
# df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt # 48
# df["NEW_HouseAge"] = df.YrSold - df.YearBuilt # 73
# # Total Floor
# df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"] # 32
# # Total House Area
# df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF # 156
# df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"] # 61
# df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd # 40
# df["NEW_TotalQual"] = df[["OverallQual", "OverallCond"]].sum(axis=1) # 42
# df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd) # 30
# df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt # 31


############################################################
# Değişkenler arasındaki korelasyonu inceleyelim
cat_cols, num_cols, cat_but_car = grab_col_names(df)

numeric_data = pd.DataFrame()

for feature in num_cols:
    numeric_data[feature] = df[feature]

corr_data = numeric_data.corr(method='pearson')

plt.figure(figsize=(30, 30))
sns.heatmap(data= corr_data, cmap='coolwarm', annot=True, fmt='.2g')

#############################################################
numeric_data['SalePrice'] = df['SalePrice']

corr_data = numeric_data.corr(method='pearson')
corr_data = corr_data[['SalePrice']]

plt.figure(figsize=(7, 10))
sns.heatmap(data=corr_data, cmap='coolwarm', annot=True, fmt='.2g')

##############################################################
# Tüm değişkenlerin dağılımını kontrol edelim
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Sürekli Değişkenlerin Dağılımı", fontsize=14)
plt.show()

##############################################################
# Kategorik değişkenlerin dağılımını inceleyelim
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df, x=col, hue=col, legend=False, palette="viridis")
    plt.xticks(rotation=45)
    plt.title(f"{col} Değişkeninin Dağılımı")
    plt.show()




##############################################################
#  0.1' den daha düşük korelasyona sahip değişkenleri silelim.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

numeric_data = pd.DataFrame()
for feature in num_cols:
    numeric_data[feature] = df[feature]

corr_data = numeric_data.corr(method='pearson')
numeric_data['SalePrice'] = df['SalePrice']  # ADD SALEPRICE COLUMN
corr_data = corr_data[['SalePrice']]          # ONLY SHOWS CORRELATION FOR SALEPRICE FEATURE

low_corr_features = corr_data[abs(corr_data['SalePrice']) < 0.1].index.tolist()
low_corr_features = [col for col in low_corr_features if 'Id' not in col]
print(low_corr_features)

df.drop(low_corr_features, axis=1, inplace=True)
# df.drop(columns='ScreenPorch',axis=1, inplace=True)

# Adım 4: Encoding işlemlerini gerçekleştirelim.
# Label Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One-Hot Encoding
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

ohe_cols = [col for col in df.columns if df[col].dtype == 'O']
df = one_hot_encoder(df, ohe_cols, drop_first=True)

# ohe_cols = [col for col in new_df.columns if (10 >= new_df[col].nunique() > 2) and (new_df[col].dtype == 'O')]
# one_hot_encoder(new_df, ohe_cols)
cat_cols, num_cols, cat_but_car = grab_col_names(df)

####################################

# Standart Scaler
scaler = StandardScaler()
num_cols = [col for col in num_cols if col not in ["Id", "SalePrice"]]
X_scaled = scaler.fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)


###################################
# Görev 3: Model Kurma
###################################
# Adım 1: Train ve Test verisini ayıralım. (SalePrice değişkeni boş olan değerler test verisidir.)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

#
# y = np.log1p(train_df['SalePrice'])
# X = train_df.drop(["Id", "SalePrice"], axis=1)
#
# X_to_predict = test_df.drop(["Id", "SalePrice"], axis=1)
#
# # Eğitim
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=17)
# lgbm_model = LGBMRegressor(random_state=46, verbose=-1).fit(X_train, y_train)
#
# # RMSE hesapla
# rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
# print("RMSE:", rmse)

# 1
# 0.131684   --mean
# 0.132029   -- *
# 2
# 0.0447732   , 0.044513,  RMSE: 0.0419
# 3
# 0.135085
# 4
# 0.1327740
# En son
# 0.1298

# ########
# df_1  : 0.057
# df_2  : 0.048
# df_3  : 0.137
# df_4  : 0.133
# final : 0.0442
########

# # Gerçek test verisinde tahmin
# test_pred_log = lgbm_model.predict(X_to_predict)
# test_pred = np.expm1(test_pred_log)
#
# # Submission dosyası
# df_submission = pd.DataFrame({"Id": test_df['Id'], "SalePrice": test_pred})
# df_submission.to_csv("housePricePredictions_submission_new_features.csv", index=False)

# def plot_importance(model, features, num=len(X), save=False):
#
#     feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
#     plt.figure(figsize=(10, 10))
#     sns.set(font_scale=1)
#     sns.barplot(x="Value",
#                 y="Feature",
#                 data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
#                 palette= sns.color_palette("viridis", num))
#     plt.title("Features")
#     plt.tight_layout()
#     plt.show()
#
#     if save:
#         plt.savefig("importances.png")
#
# # En önemli olan ilk 50 değişkeni görüntüleyelim
# plot_importance(lgbm_model, X, 50)


# Gereksiz değişkenleri silelim
# feature_imp = pd.DataFrame({'Value': lgbm_model.feature_importances_, 'Feature': X.columns})
# feature_imp = feature_imp[feature_imp['Feature'] != 'position_id']
# # deleted_features = feature_imp.sort_values(by="Value", ascending=True)[:50]
# deleted_features = feature_imp[feature_imp['Value'] == 0]
# deleted_features = deleted_features["Feature"].tolist()
# df.drop(deleted_features, axis=1, inplace=True)

##################################################################
# Adım 2: Train verisi ile model kurup, model başarısını değerlendirelim.
def base_model(train_df):
    X = train_df.drop(['Id', 'SalePrice'], axis=1)
    y = train_df['SalePrice']

    models = [('LR', LinearRegression()),
              ("Ridge", Ridge(alpha=1.0, max_iter=5000)),
              ("Lasso", Lasso(alpha=0.01, max_iter=5000)),
              ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.2)),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor(verbose=-1)),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")


base_model(train_df)

"""
RMSE: 3468026753.5811 (LR) 
RMSE: 30728.8878 (Ridge) 
RMSE: 31553.0787 (Lasso) 
RMSE: 29438.7946 (ElasticNet) 
RMSE: 31926.122 (KNN) 
RMSE: 37668.7387 (CART) 
RMSE: 27954.1397 (RF) 
RMSE: 24269.6138 (GBM) 
RMSE: 27820.6724 (XGBoost) 
RMSE: 26783.2776 (LightGBM)  
RMSE: 23823.1671 (CatBoost)

"""

# Bonus: Hedef değişkene log dönüşümü yaparak model kuralım ve rmse sonuçlarını gözlemleyelim.( Not: Log'un tersini (inverse) almayı unutmayınız.)
def log_transformed_basic_model(train_df):
    X = train_df.drop(["Id", "SalePrice"], axis=1)
    y = np.log1p(train_df['SalePrice'])

    models = [('LR', LinearRegression()),
              ("Ridge", Ridge(max_iter=5000)),
              ("Lasso", Lasso(alpha=0.001, max_iter=5000)),
              ("ElasticNet", ElasticNet(alpha=0.01, l1_ratio=0.2)),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor(verbose=-1)),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name}) ")

log_transformed_basic_model(train_df)


# RMSE: 597487997.7884 (LR)
# RMSE: 0.1278 (Ridge)
# RMSE: 0.1227 (Lasso)
# RMSE: 0.1293 (ElasticNet)
# RMSE: 0.1511 (KNN)
# RMSE: 0.1985 (CART)
# RMSE: 0.1383 (RF)
# RMSE: 0.1253 (GBM)
# RMSE: 0.1357 (XGBoost)
# RMSE: 0.1271 (LightGBM)
# RMSE: 0.1173 (CatBoost)

#######################################################################
# Log dönüşümünün gerçekleştirilmesi

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

# Verinin eğitim ve tet verisi olarak bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=46)


# lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

y_pred
# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y = np.expm1(y_pred)
new_y
new_y_test = np.expm1(y_test)
new_y_test

np.sqrt(mean_squared_error(new_y_test, new_y))

# RMSE : 23448.08812951357
#######################################################################
# Adım 3: Hiperparemetre optimizasyonu gerçekleştirelim.

def lgbm_model_with_hiperparameter_opt(train_df):
    y = np.log1p(train_df['SalePrice'])
    X = train_df.drop(["Id", "SalePrice"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

    lgbm_model = LGBMRegressor(random_state=46, verbose=-1)

    rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print("Before Hiperparametre Optimization...")
    print('RMSE: ', rmse)
    # rms : 0.1304040453986512

    lgbm_params = {"learning_rate": [0.001, 0.003, 0.004],
                   "n_estimators": [2000, 2500, 3000,3500],  # 1500
                   "colsample_bytree": [0.1, 0.3, 0.4, 0.5]}

    lgbm_gs_best = GridSearchCV(lgbm_model,
                                lgbm_params,
                                cv=5,
                                n_jobs=-1,
                                verbose=True).fit(X_train, y_train)


    final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

    lgbm_gs_best.best_params_

    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
    print("After Hyperparameter Optimization...")
    print('RMSE: ', rmse)

    # lgbm_pred = final_model.predict(X_test)
    # lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    # print("LGBM RMSE: ", lgbm_rmse)
    # # 0.048

    return final_model, X


lgbm_final_model, X = lgbm_model_with_hiperparameter_opt(train_df)

# # Before RMSE:  0.1325
# # Afeter RMSE:  0.1202

#################################################################
# Birden fazla model ile hiperparametre optimizasyonu gerçekleştirelim ve sonuçları inceleyelim.

ridge_params = {'alpha': np.logspace(-5, 5, 1000),
                 'max_iter':[500, 1000,5000]}

elastic_params = {"alpha": [0.001, 0.003, 0.005, 0.01, 0.03],
                  "l1_ratio": [0.001, 0.005, 0.01, 0.03, 0.05, 0.1],
                  'max_iter': [500, 1000, 5000]}  #(1000, 5000,10000)

gbm_params = {"learning_rate": [0.01, 0.03, 0.05,0.1],
              "n_estimators": [500, 1000, 1500,2000],
              "max_depth": [3, 5, 7]}

# LightGBM
# lgbm_params = {"learning_rate": [0.001, 0.003, 0.005, 0.01, 0.03],
#                 "n_estimators": [500, 1000, 1500, 2000, 2500],
#                 "colsample_bytree": [0.1, 0.3, 0.5, 0.7, 1.0]}

lgbm_params = {"learning_rate": [0.001, 0.003, 0.004],
               "n_estimators": [2000, 2500, 3000, 3500],  # 1500
               "colsample_bytree": [0.1, 0.3, 0.4, 0.5]}
# XGBoost Modeli
xgb_params = {"learning_rate": [0.005,0.01, 0.03,0.04, 0.05],
              "n_estimators": [500,1000, 1500, 2000,2500],
              "max_depth": [3, 4, 5]}

# CatBoost Modeli
cat_params = {"learning_rate": [0.03, 0.05, 0.1],
              "iterations": [500, 1000, 1500],
              "depth": [4, 6, 8]}

models = [("Ridge", Ridge(), ridge_params),
          ("ElasticNet", ElasticNet(), elastic_params),
          ("GBM", GradientBoostingRegressor(random_state=46), gbm_params),
          ("LGBM", LGBMRegressor(random_state=46, verbose=-1), lgbm_params),
          ("XGBoost", XGBRegressor(random_state=46, objective="reg:squarederror"), xgb_params),
          ("CatBoost", CatBoostRegressor(random_state=46, verbose=False), cat_params)]


y = np.log1p(train_df['SalePrice'])  # Log dönüşümü
X = train_df.drop(["Id", "SalePrice"], axis=1)


def best_model_with_hyperparameter_opt(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=46)

    best_rmse_values = []
    for name, model, params in models:
        print(f"##### {name} MODEL ####")
        model = type(model)()
        model_gs = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=False).fit(X_train, y_train)
        best_model = model.set_params(**model_gs.best_params_).fit(X,y)
        # best_model = model_gs.best_estimator_
        model_pred = best_model.predict(X_test)
        best_rmse = np.sqrt(mean_squared_error(y_test, model_pred))

        best_rmse_values.append((name, best_rmse, model_gs.best_params_, best_model))

    for name, best_rmse, best_params, model in best_rmse_values:
        print(f"{name} -  Best RMSE: {best_rmse:.4f}")
        print(f"En iyi parametreler: {best_params}")
        print("_____------______-------_____------______-------_____------______")

best_model_with_hyperparameter_opt(X,y)

###################################################################
# En iyi sonucu veren modeller ile hiperparametre optimizasyonu gerçekleştirip ağırlıklandırılmış bir tahmin verisi elde edelim
def bbm_model_with_hyperparameter_opt(train_df):
    y = np.log1p(train_df['SalePrice'])
    X = train_df.drop(["Id", "SalePrice"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)


    # GBM
    # gbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1],
    #               "n_estimators": [500, 1000, 1500, 2000],
    #               "max_depth": [3, 5, 7]}
    #
    # gbm_model = GradientBoostingRegressor(random_state=46)
    # gbm_gs = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    # gbm_best = gbm_model.set_params(**gbm_gs.best_params_).fit(X, y)
    #
    # gbm_pred = gbm_best.predict(X_test)
    # gbm_rmse = np.sqrt(mean_squared_error(y_test, gbm_pred))
    # print("GBM RMSE: ", gbm_rmse)
    # 0.0745


    # LightGBM Modeli
    lgbm_params = {"learning_rate": [0.001, 0.003, 0.005],
                   "n_estimators": [1500, 2000, 2500],  # 1500
                   "colsample_bytree": [0.3, 0.5, 0.7]}

    lgbm_model = LGBMRegressor(random_state=46)
    lgbm_gs = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    lgbm_best = lgbm_model.set_params(**lgbm_gs.best_params_).fit(X, y)

    lgbm_pred = lgbm_best.predict(X_test)
    lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    print("LGBM RMSE: ", lgbm_rmse)
    # RMSE: 0.5014

    # XGBoost Modeli
    # xgb_params = {"learning_rate": [0.01, 0.03, 0.05],
    #               "n_estimators": [1000, 1500, 2000],  # 2000
    #               "max_depth": [2, 3, 5]}  # 2
    xgb_params = {"learning_rate": [0.01, 0.03, 0.04, 0.05],
                  "n_estimators": [500, 1000, 1500, 2000],  # 2000
                  "max_depth": [3, 4, 5]}
    xgb_model = XGBRegressor(random_state=46, objective="reg:squarederror")
    xgb_gs = GridSearchCV(xgb_model, xgb_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    xgb_best = xgb_model.set_params(**xgb_gs.best_params_).fit(X, y)

    xgb_pred = xgb_best.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    print("XGBM RMSE: ", xgb_rmse)
    # 0.0773

    # CatBoost Modeli
    cat_params = {"learning_rate": [0.01, 0.03, 0.05],
                  "iterations": [500, 1000, 1500],
                  "depth": [4, 6, 8]}

    cat_model = CatBoostRegressor(random_state=46, verbose=False)
    cat_gs = GridSearchCV(cat_model, cat_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    cat_best = cat_model.set_params(**cat_gs.best_params_).fit(X, y)

    cat_pred = cat_best.predict(X_test)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
    print("CAT RMSE: ", cat_rmse)

    # Model Tahminleri (Blend Etme)
    lgbm_pred = lgbm_best.predict(X)
    xgb_pred = xgb_best.predict(X)
    cat_pred = cat_best.predict(X)

    final_pred = (0.28 * lgbm_pred) + (0.12 * xgb_pred) + (0.6 * cat_pred)  # Ağırlıklı ortalama

    # RMSE Hesaplama
    rmse = np.sqrt(mean_squared_error(y, final_pred))
    print("After Hyperparameter Optimization (BBM)...")
    print('RMSE: ', rmse)

    return lgbm_best, xgb_best, cat_best, final_pred, X, y

lgbm_best, xgb_best, cat_best, final_pred, X, y = bbm_model_with_hyperparameter_opt(train_df)
# LGBM RMSE:  0.0462
# XGBM RMSE:  0.0563
# CAT RMSE:  0.03081
# Final RMSE:  0.0388

# LGBM RMSE:  0.03363694285493568
# XGBM RMSE:  0.07508018806085527
# RMSE:  0.03378057187402545

#LGBM RMSE:  0.0148011

# En iyi ağırlıkları bulalım.
def best_weights_bbm(lgbm_pred, xgb_pred, cat_pred, y):
    # Test edilecek ağırlık kombinasyonları
    weight_combinations = list(itertools.product([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], repeat=3))
    best_score = float("inf")
    best_weights = None

    for w in weight_combinations:
        if sum(w) == 1.0:  # Ağırlıkların toplamı 1 olmalı
            final_pred_best_we = w[0] * lgbm_pred + w[1] * cat_pred + w[2] * xgb_pred
            score = np.sqrt(mean_squared_error(y, final_pred_best_we))  # RMSE

            if score < best_score:
                best_score = score
                best_weights = w

    print(f"Best Weights: {best_weights} (Best Score: {best_score})")
    print(f"LGBM Best Weights:{best_weights[0]}")
    print(f"CAT Best Weights:{best_weights[1]}")
    print(f"XGBosst Best Weights:{best_weights[2]}")
    return best_weights[0], best_weights[1], best_weights[2]

# En son tahmin sonuçlarını inceleyelim ve RMSE değerlerini karşılaştıralım.
def final_prediction(lgbm_best, xgb_best, cat_best, X, y):
    lgbm_pred = lgbm_best.predict(X)
    xgb_pred = xgb_best.predict(X)
    cat_pred = cat_best.predict(X)

    final_pred = (0.28 * lgbm_pred) + (0.12 * xgb_pred) + (0.6 * cat_pred)  # Ağırlıklı ortalama
    rmse = np.sqrt(mean_squared_error(y, final_pred))
    print('Final RMSE: ', rmse)

    lgbm_best_weight, cat_best_weight, xgbm_best_weight = best_weights_bbm(lgbm_pred, xgb_pred, cat_pred, y)

    # Herbir model için en iyi ağırlıkları hesaplayalım
    weighted_final_pred = (lgbm_best_weight * lgbm_pred) + (cat_best_weight * cat_pred) + (
            xgbm_best_weight * xgb_pred)  # Ağırlıklı ortalama

    w_rmse = np.sqrt(mean_squared_error(y, weighted_final_pred))
    print('Weighted Final RMSE: ', w_rmse)
    # Weighted Final RMSE:  0.002695

final_prediction(lgbm_best, xgb_best, cat_best, X, y)

# First Final RMSE:  0.0604619065306031
# Weighted Final RMSE:  0.060084452406837484

from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score

# Taban modeller (önceden en iyi params ile eğitilmiş modelleri veriyoruz)
base_learners = [
    ('lgbm', lgbm_best),
    ('xgb', xgb_best),
    ('cat', cat_best)
]

# Meta model: Ridge genellikle iyi sonuç verir
stack_model = StackingRegressor(estimators=base_learners, final_estimator=RidgeCV(), cv=5)

# Eğitimi tüm veride yap
stack_model.fit(X, y)

# RMSE değerlendirmesi (cross_val_score ile)
stack_rmse = np.mean(np.sqrt(-cross_val_score(stack_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print("STACKING RMSE:", stack_rmse)

###################################################################
# Adım 4: Değişken önem düzeyini inceleyelim.

def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value",
                y="Feature",
                data=feature_imp.sort_values(by="Value", ascending=False)[0:num],
                palette= sns.color_palette("viridis", num))
    plt.title("Features")
    plt.tight_layout()
    plt.show()

    if save:
        plt.savefig("importances.png")


# En önemli olan ilk 50 değişkeni görüntüleyelim
plot_importance(lgbm_final_model, X, 50)


# Başka modellerle de deneyebiliriz
# model = LGBMRegressor()
# model.fit(X, y)
# plot_importance(model, X, 50)

######################################################
# Bonus: Test verisinde boş olan salePrice değişkenlerini tahminleyelim ve Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturup sonucumuzu yükleyelim.

test_X = test_df.drop(["Id", "SalePrice"], axis=1, errors='ignore')

# Test setinde tahmin yap
lgbm_pred_test = lgbm_best.predict(test_X)
xgb_pred_test = xgb_best.predict(test_X)
cat_pred_test = cat_best.predict(test_X)

best_w_lgbm, best_w_cat, best_w_xgb = best_weights_bbm(lgbm_best.predict(X),
                                                       xgb_best.predict(X),
                                                       cat_best.predict(X), y)

# Ağırlıklı ortalama ile test tahminleri
weighted_final_pred_test = (best_w_lgbm * lgbm_pred_test) + (best_w_cat * cat_pred_test) + (best_w_xgb * xgb_pred_test)
final_pred_test = np.expm1(weighted_final_pred_test)  # Log dönüşümünü tersine çevir

# X_test_final = test_df.drop(["Id", "SalePrice"], axis=1)
# final_preds_test = np.expm1(stack_model.predict(X_test_final))

stack_pred_test = stack_model.predict(test_X)
# blending (ağırlıklı) tahminin zaten var:
# weighted_final_pred_test = (w1 * lgbm) + (w2 * xgb) + (w3 * cat)

final_ensemble_log = (0.4 * stack_pred_test) + (0.6 * weighted_final_pred_test)
final_pred_test = np.expm1(final_ensemble_log)



df_submission = pd.DataFrame({"Id": test_df['Id'], "SalePrice": final_pred_test})
df_submission.to_csv("housePricePredictions_submission_final_test2.csv", index=False)

# predictions = lgbm_final_model.predict(test_df.drop(["Id","SalePrice"], axis=1))
# dictionary = {"Id": test_df['Id'], "SalePrice":predictions}
# dfSubmission = pd.DataFrame(dictionary)
# dfSubmission.to_csv("housePricePredictions_lgbm.csv", index=False)
