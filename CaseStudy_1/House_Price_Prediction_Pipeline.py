########################################################
# Ev Fiyat Tahmin Modeli | House Price Prediction Model
########################################################

########################################################
# Keşifçi Veri Analizi | Exploratory Data Analysis
########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
import itertools

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.10)
    quartile3 = dataframe[variable].quantile(0.90)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

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

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


def bbm_model_with_hyperparameter_opt(train_df):
    y = np.log1p(train_df['SalePrice'])
    X = train_df.drop(["Id", "SalePrice"], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

    # LightGBM
    lgbm_params = {"learning_rate": [0.001, 0.003, 0.004],
                   "n_estimators": [2000, 2500,3000,3500],
                   "colsample_bytree": [0.1, 0.3, 0.4, 0.5]}

    lgbm_model = LGBMRegressor(random_state=46)
    lgbm_gs = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    lgbm_best = lgbm_model.set_params(**lgbm_gs.best_params_).fit(X, y)
    # lgbm_best = lgbm_gs.best_estimator_

    lgbm_pred = lgbm_best.predict(X_test)
    lgbm_rmse = np.sqrt(mean_squared_error(y_test, lgbm_pred))
    print("LGBM RMSE: ", lgbm_rmse)
    # RMSE:  0.0547  {'colsample_bytree': 0.3, 'learning_rate': 0.003, 'n_estimators': 3000}

    # XGBoost
    xgb_params = {"learning_rate": [0.01,0.02, 0.03,0.04, 0.05],
                  "n_estimators": [500,1000, 1500, 2000,2500],
                  "max_depth": [3,4, 5]}

    xgb_model = XGBRegressor(random_state=46, objective="reg:squarederror")
    xgb_gs = GridSearchCV(xgb_model, xgb_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    xgb_best = xgb_model.set_params(**xgb_gs.best_params_).fit(X, y)
    # xgb_best = xgb_gs.best_estimator_

    xgb_pred = xgb_best.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    print("XGBM RMSE: ", xgb_rmse)
    # XGBM RMSE:  0.0669    {'learning_rate': 0.03, 'max_depth': 3, 'n_estimators': 1000}

    # CatBoost
    cat_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1],
                  "iterations": [500, 1000, 1500],
                  "depth": [4, 6, 8]}

    cat_model = CatBoostRegressor(random_state=46, verbose=False)
    cat_gs = GridSearchCV(cat_model, cat_params, cv=5, n_jobs=-1, verbose=True).fit(X_train, y_train)
    cat_best = cat_model.set_params(**cat_gs.best_params_).fit(X, y)
    # cat_best = cat_gs.best_estimator_

    cat_pred = cat_best.predict(X_test)
    cat_rmse = np.sqrt(mean_squared_error(y_test, cat_pred))
    print("CAT RMSE: ", cat_rmse)
    # CAT RMSE:  0.0616


    return lgbm_best, xgb_best, cat_best, X, y

###################################################################
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


def final_prediction(lgbm_best, xgb_best, cat_best, X, y):
    lgbm_pred = lgbm_best.predict(X)
    xgb_pred = xgb_best.predict(X)
    cat_pred = cat_best.predict(X)

    final_pred = (0.28 * lgbm_pred) + (0.12 * xgb_pred) + (0.6 * cat_pred)  # Ağırlıklı ortalama
    rmse = np.sqrt(mean_squared_error(y, final_pred))
    print('First RMSE: ', rmse)

    lgbm_best_weight, cat_best_weight, xgbm_best_weight = best_weights_bbm(lgbm_pred, xgb_pred, cat_pred, y)
    # Herbir model için en iyi ağırlıkları hesaplayalım
    weighted_final_pred = (lgbm_best_weight * lgbm_pred) + (cat_best_weight * cat_pred) + (
            xgbm_best_weight * xgb_pred)  # Ağırlıklı ortalama

    rmse = np.sqrt(mean_squared_error(y, weighted_final_pred))
    print('Weighted Final RMSE: ', rmse)
    # Before RMSE:  0.0388
    # After RMSE: 0.0349

    # Modeli eğitelim ve eğittiğimiz modeli test_prediction fonksiyonuna gönderelim
    # pred_stac_xgb, stacking_model = stacking_with_xgbm(lgbm_pred, cat_pred, xgb_pred, weighted_final_pred, y)
    # final_pred = (0.8 * pred_stac_xgb) + (0.2 * weighted_final_pred)
    # final_rmse = np.sqrt(mean_squared_error(y, final_pred))
    # print("Final RMSE with Ensemble Stacking:", final_rmse)
    # return final_pred, stacking_model

def final_test_predictions(test_df, lgbm_best, xgb_best, cat_best,X, y):
    test_X = test_df.drop(["Id", "SalePrice"], axis=1, errors='ignore')

    # Test setinde tahmin yap
    lgbm_pred_test = lgbm_best.predict(test_X)
    xgb_pred_test = xgb_best.predict(test_X)
    cat_pred_test = cat_best.predict(test_X)

    ####################################
    # En iyi ağırlıkları belirle
    best_w_lgbm, best_w_cat, best_w_xgb = best_weights_bbm(lgbm_best.predict(X),
                                                           xgb_best.predict(X),
                                                           cat_best.predict(X), y)

    # Ağırlıklı ortalama ile test tahminleri
    weighted_final_pred_test = (best_w_lgbm * lgbm_pred_test) + (best_w_cat * cat_pred_test) + (best_w_xgb * xgb_pred_test)
    final_pred_test = np.expm1(weighted_final_pred_test)  # Log dönüşümünü tersine çevir

    return final_pred_test

def create_submission_file(final_pred_test, test_df):
    df_submission = pd.DataFrame({"Id": test_df['Id'], "SalePrice": final_pred_test})
    df_submission.to_csv("housePricePredictions_2222.csv", index=False)
    print("Submission dosyası oluşturuldu: housePricePredictions5.csv")


#####################################################################
# Veri Hazırlama | Data Preparation
def data_prep(df):
    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    num_cols = [col for col in num_cols if col not in "Id"]

    # missing_values_table(df)

    # %10 üzerinde eksik değer içeren sütunları silelim
    na_cols = missing_values_table(df, False, True)
    na_cols_over_50 = na_cols[(na_cols.values > 50) & (na_cols.index != 'SalePrice')]

    # Nümerik sütunların çarpıklık değerlerini hesaplayalım
    skew_cols = df[num_cols].skew().sort_values(ascending=False)
    skew_cols_over_10 = skew_cols_over_10 = skew_cols[(skew_cols.values > 1.5) &
                              (skew_cols.index != 'SalePrice')]

    #  Baskın Etikete Sahip Değişkenler
    # RoofMatl, Street, GarageCond, Condition2, Utilities, Heating
    threshold = 0.95
    dominant_ratio = df.apply(lambda col: col.value_counts(normalize=True).max())
    dominant_cols = dominant_ratio[dominant_ratio > threshold].index.tolist()
    # print(dominant_ratio[dominant_ratio > threshold].sort_values(ascending=False))

    drop_list = (set(na_cols_over_50.index.tolist())
                 | set(skew_cols_over_10.index.tolist())
                 | set(dominant_cols))

    df.drop(drop_list, axis=1, inplace=True)

    # null_cols = ['GarageFinish','GarageQual','GarageType','BsmtExposure','BsmtCond','BsmtQual','BsmtFinType2','BsmtFinType1']
    null_cols = ['GarageFinish', 'GarageType', 'BsmtExposure', 'BsmtQual', 'BsmtFinType1']
    for col in null_cols:
        df[col] = df[col].fillna('No')


    df['AvgPrice_neighborhood'] = df.groupby("Neighborhood")['SalePrice'].transform('mean')
    df['AvgPrice_YearBuilt'] = df.groupby("YearBuilt")['SalePrice'].transform('mean')

    df.loc[df["2ndFlrSF"] == 0, "2ndFlrSF"] = np.nan

    # GarageYrBlt -> boş değerleri YearBuilt değerleriyle dolduralım + int tipine dönüştürelim
    df['GarageYrBlt'] = df['GarageYrBlt'].fillna(df['YearBuilt']).astype(int)

    na_cols = missing_values_table(df, True)
    na_and_cat_cols = [col for col in df[na_cols] if col in cat_cols]
    na_and_num_cols = [col for col in df[na_cols] if col in num_cols and col != 'SalePrice']

    for col in na_and_num_cols:
        df[col] = df[col].fillna(df[col].mean())
        # df[col] = df[col].fillna(0)

    for col in na_and_cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Aykırı değerleri temizleyelim
    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    for col in num_cols:
        if col != "SalePrice":
            replace_with_thresholds(df, col)

    for col in num_cols:
        if col != "SalePrice":
            print(col, check_outlier(df, col))

    # Rare Encoder uygulayalım
    df = rare_encoder(df, 0.01)

    # Yeni değişkenler oluşturunuz.
    df['TotalArea'] = df['GrLivArea'] + df['TotalBsmtSF']
    df['GarageEfficiency'] = df['GarageArea'] / (df['GarageCars'] + 1)  # +1 TO AVOID DIVISION BY ZERO
    df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]
    df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    numeric_data = pd.DataFrame()
    for feature in num_cols:
        numeric_data[feature] = df[feature]

    corr_data = numeric_data.corr(method='pearson')
    numeric_data['SalePrice'] = df['SalePrice']  # ADD SALEPRICE COLUMN
    corr_data = corr_data[['SalePrice']]  # ONLY SHOWS CORRELATION FOR SALEPRICE FEATURE

    low_corr_features = corr_data[abs(corr_data['SalePrice']) < 0.1].index.tolist()
    low_corr_features = [col for col in low_corr_features if 'Id' not in col]
    print(low_corr_features)

    df.drop(low_corr_features, axis=1, inplace=True)
    # SİLEREK MODEL PERFORMANSINI DENE
    # drop_list = ["LandContour", "LandSlope", "Neighborhood", "MSSubClass", "MoSold"]
    # # drop_list'teki değişkenlerin düşürülmesi
    # df.drop(drop_list, axis=1, inplace=True)

    # Encoding işlemlerini gerçekleştiriniz.
    binary_cols = [col for col in df.columns if df[col].dtype not in ['int64', 'float64']
                   and df[col].nunique() == 2]

    for col in binary_cols:
        label_encoder(df, col)

    # ohe_cols = [col for col in new_df.columns if (10 >= new_df[col].nunique() > 2) and (new_df[col].dtype == 'O')]
    # one_hot_encoder(new_df, ohe_cols)
    ohe_cols = [col for col in df.columns if df[col].dtype == 'O']
    df = one_hot_encoder(df, ohe_cols, drop_first=True)

    cat_cols, num_cols, cat_but_car = grab_col_names(df)
    scaler = StandardScaler()
    num_cols = [col for col in num_cols if col not in ["Id", "SalePrice"]]
    X_scaled = scaler.fit_transform(df[num_cols])
    df[num_cols] = pd.DataFrame(X_scaled, columns=num_cols, index=df.index)

    return df

def main():
    train_df = pd.read_csv('Github_Machine_Learning/CaseStudy_1/datasets/train.csv')
    test_df = pd.read_csv('Github_Machine_Learning/CaseStudy_1/datasets/test.csv')

    df_ = pd.concat([train_df, test_df], ignore_index=False).reset_index(drop=True)

    df = data_prep(df_)

    train_df = df[df['SalePrice'].notnull()]
    test_df = df[df['SalePrice'].isnull()]

    lgbm_best, xgb_best, cat_best, X, y = bbm_model_with_hyperparameter_opt(train_df)
    # LGBM RMSE:  0.043967
    # XGBM RMSE:  0.064556
    # CAT RMSE:  0.032337

    final_prediction(lgbm_best, xgb_best, cat_best, X, y)
    # 0.0368727

    final_pred_test = final_test_predictions(test_df, lgbm_best, xgb_best, cat_best, X, y)
    # Tahmin sonuçlarını kaydet
    create_submission_file(final_pred_test, test_df)
