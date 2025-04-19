#######################################################
# Makine Öğrenmesi ile Yetenek Avcılığı Sınıflandırma
#######################################################

# İş Problemi:
# Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf (average, highlighted) oyuncu olduğunu tahminleme.

#######################################################
# Veri Seti Hikayesi
# Veri seti Scoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
# içerisinde puanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.

#######################################################
# scoutium_attributes.csv
# 8 Değişken 10.730 Gözlem

# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id : Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# position_id : İlgili oyuncunun o maçta oynadığı pozisyonun id’si
            # 1: Kaleci
            # 2: Stoper
            # 3: Sağ bek
            # 4: Sol bek
            # 5: Defansif orta saha
            # 6: Merkez orta saha
            # 7: Sağ kanat
            # 8: Sol kanat
            # 9: Ofansif orta saha
            # 10: Forvet

# analysis_id : Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
# attribute_id : Oyuncuların değerlendirildiği her bir özelliğin id'si
# attribute_value : Bir scoutun bir oyuncunun bir özelliğine verdiği değer(puan)


# scoutium_potential_labels.csv
# 5 Değişken 322 Gözlem

# task_response_id : Bir scoutun bir maçta bir takımın kadrosundaki tüm oyunculara dair değerlendirmelerinin kümesi
# match_id : İlgili maçın id'si
# evaluator_id : Değerlendiricinin(scout'un) id'si
# player_id : İlgili oyuncunun id'si
# potential_label : Bir scoutun bir maçta bir oyuncuyla ilgili nihai kararını belirten etiket. (hedef değişken)


#############################
# Keşifçi Veri Analizi
#############################

import numpy as np
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import itertools

import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz.

sc_attr = pd.read_csv('Github_Machine_Learning/Bonus_2/datasets/scoutium_attributes.csv', sep=';')
sc_pot_labels = pd.read_csv('Github_Machine_Learning/Bonus_2/datasets/scoutium_potential_labels.csv', sep=';')

def check_info(dataframe):
    print("########## HEAD ##########)")
    print(dataframe.head())
    print("########## SHAPE ##########)")
    print(dataframe.shape)
    print("########## INFO ##########)")
    print(dataframe.info())
    print("########## NULL COUNT ##########)")
    print(dataframe.isnull().sum())
    print("########## DESCRIBE ##########)")
    print(dataframe.describe().T)

check_info(sc_attr)

check_info(sc_pot_labels)

# Okutmuş olduğumuz csv dosyalarını merge fonksiyonunu kullanarak birleştiriniz.
# ("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
df = sc_attr.merge(sc_pot_labels, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="outer")

df.head()

# position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.

df = df[df['position_id'] != 1]

# potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.
# ( below_average sınıfı tüm verisetinin %1'ini oluşturur)

df['potential_label'].value_counts()

df = df[df['potential_label'] != 'below_average']

check_info(df)

# Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. Bu pivot table'da her satırda bir oyuncu
# olacak şekilde manipülasyon yapınız.

# İndekste “player_id”,“position_id” ve “potential_label”, sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
# “attribute_value” olacak şekilde pivot table’ı oluşturunuz.

#                                               4322     4323     4324     4325     4326     4327 ..
# player_id   position_id   potential_label                          ..
# 1355710     7             average             50.500   50.500   34.000   50.500   45.000   45.000 ..
# 1356362     9             average             67.000   67.000   67.000   67.000   67.000   67.000 ..
# 1356375     3             average             67.000   67.000   67.000   67.000   67.000   67.000 ..
#             4             average             67.000   78.000   67.000   67.000   67.000   78.000 ..
# 1356411     9             average             67.000   67.000   78.000   78.000   67.000   67.000 ..


pivot_df = df.pivot_table(index=["player_id","position_id","potential_label"], columns="attribute_id", values="attribute_value")
         # pd.pivot_table(df, values="attribute_value", columns="attribute_id", index=["player_id","position_id","potential_label"])


# “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz.
pivot_df = pivot_df.reset_index()

pivot_df.columns = pivot_df.columns.astype(str)
#         [str(col) for col in pt.columns]

pivot_df.head()
pivot_df.info()


# Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
num_cols = [col for col in pivot_df.columns if pivot_df[col].dtypes != 'O']
# num_cols = pivot_df.columns[3:]

# Numerik ve kategorik değişkenleri inceleyelim.

# Kategorik Değişkenlerin Analizi
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in ["position_id","potential_label"]:
    cat_summary(pivot_df, col)


# Numerik Değişkenlerin Analizi
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(pivot_df, col, plot=True)


# Numerik değişkenler ile hedef değişken incelemesini yapalım.
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(pivot_df, "potential_label", col)

# pt.groupby(["position_id", "potential_label"])["4423"].mean()

##################################
# Korelasyona bakınız.
##################################
pivot_df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(pivot_df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#################
# position_id ve potential_label arasındaki ilişkiyi inceleyelim
cross_tab = pd.crosstab(df['position_id'], df['potential_label'], normalize='index')
plt.subplot(2, 2, 1)
sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='.2f', cbar_kws={'label': 'Oran'})
plt.title('Position ID - Potential Label İlişkisi (Normalleştirilmiş)')
plt.xlabel('Potential Label')
plt.ylabel('Position ID')

########################
from statsmodels.graphics.mosaicplot import mosaic
plt.subplot(2, 2, 3)
mosaic(df, ['position_id', 'potential_label'], title='Mozaik Grafiği')

#################
# Buraya kadar olan süreci fonksiyonlaştıralım
def data_prep(df):
    df = df[df['position_id'] != 1]
    df = df[df['potential_label'] != 'below_average']
    pivot_df = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id",
                              values="attribute_value")

    pivot_df = pivot_df.reset_index()
    pivot_df.columns = pivot_df.columns.astype(str)

    return pivot_df

df = sc_attr.merge(sc_pot_labels, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="outer")
pivot_df = data_prep(df)


# Adım 6: Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

pivot_df = label_encoder(pivot_df, 'potential_label')


# Adım 8: Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
num_cols = [col for col in pivot_df.columns if pivot_df[col].dtypes != 'O']
num_cols = num_cols[3:]

scaler = StandardScaler()
pivot_df[num_cols] = scaler.fit_transform(pivot_df[num_cols])


pivot_df.head()

# Adım 9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

y = pivot_df["potential_label"]
X = pivot_df.drop(["player_id","potential_label"], axis=1)


models = [('LR', LogisticRegression(max_iter=1000, random_state=17)),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier(random_state=17)),
          ('RF', RandomForestClassifier(random_state=17)),
          # ('SVM', SVC(gamma='auto', random_state=17)),
          # ("SVC", SVC(random_state=17)),
          ('Adaboost', AdaBoostClassifier(algorithm="SAMME",random_state=17)),
          ('XGB', XGBClassifier(random_state=17)),
          ("LightGBM", LGBMClassifier(verbose=-1, random_state=17)),
          ("CatBoost", CatBoostClassifier(verbose=False, random_state=17))]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

########## LR ##########
# Accuracy: 0.8562
# Auc: 0.8461
# Recall: 0.5067
# Precision: 0.7738
# F1: 0.5818
# ########## KNN ##########
# Accuracy: 0.845
# Auc: 0.7257
# Recall: 0.31
# Precision: 0.775
# F1: 0.4279
# ########## CART ##########
# Accuracy: 0.834
# Auc: 0.7577
# Recall: 0.6267
# Precision: 0.6146
# F1: 0.5934
# ########## RF ##########
# Accuracy: 0.8708
# Auc: 0.8988
# Recall: 0.4367
# Precision: 0.905
# F1: 0.5668
# ########## Adaboost ##########
# Accuracy: 0.8709
# Auc: 0.8923
# Recall: 0.5067
# Precision: 0.8433
# F1: 0.5902
# ########## XGB ##########
# Accuracy: 0.8747
# Auc: 0.8762
# Recall: 0.5933
# Precision: 0.8187
# F1: 0.6523
# ########## LightGBM ##########
# Accuracy: 0.8817
# Auc: 0.8982
# Recall: 0.5933
# Precision: 0.8071
# F1: 0.6633
# ########## CatBoost ##########
# Accuracy: 0.8819
# Auc: 0.9001
# Recall: 0.4533
# Precision: 0.95
# F1: 0.5893


################################################################
# Görev 9: Hiperparametre Optimizasyonu yapınız.
################################################################

lgbm_model = LGBMClassifier(random_state=17)
first_cvs = cross_val_score(lgbm_model, X, y, scoring="accuracy", cv=10).mean()

lgbm_params = {"colsample_bytree": [0.1, 0.3, 0.5, 0.7, 1],
               "learning_rate": [0.001, 0.003, 0.005, 0.01, 0.05],
               "max_depth": [3, 5, 7, 8, 10],
               "n_estimators": [500, 700, 800, 1000, 1200]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=False).fit(X, y)


lgbm_gs_best.best_params_
final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

final_cvs = pd.DataFrame(cross_validate(final_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"]))
print(f"Accuracy: {round(final_cvs['test_accuracy'].mean(), 4)}")
print(f"Auc: {round(final_cvs['test_roc_auc'].mean(), 4)}")
print(f"Recall: {round(final_cvs['test_recall'].mean(), 4)}")
print(f"Precision: {round(final_cvs['test_precision'].mean(), 4)}")
print(f"F1: {round(final_cvs['test_f1'].mean(), 4)}")

# final_cvs = cross_val_score(final_model, X, y, scoring="accuracy", cv=10).mean()
# print("İlk CVS:", first_cvs)
# print("Final CVS:", final_cvs)

# LGBM
# Accuracy: 0.8894
# Auc: 0.8768
# Recall: 0.5864
# Precision: 0.8662
# F1: 0.6782

# Adım10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.
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
plot_importance(final_model, X, 100)

# 4332,4325, 4426, 4353, 4407,4349, 4344, 4326, 4350,4355, 4354,4357,4341,4423,4329,4338,4322
feature_imp = pd.DataFrame({'Value': final_model.feature_importances_, 'Feature': X.columns})
feature_imp = feature_imp[feature_imp['Feature'] != 'position_id']
added_features = feature_imp.sort_values(by="Value", ascending=False)[0:20]
best_features = added_features["Feature"].tolist()

# Bu değişkenkenlerin önem düzeyleri yüksek görünmekte, o yüzden bu değişkenleri kullanarak kombinasyonları ile yeni değişkenler üretip model başarısını gözlemleyelim
###################################
# Feature Extraction uygulayarak başarı metriklerini tekrar değerlendirelim.
##################################
def data_prep(df):
    df = df[df['position_id'] != 1]
    df = df[df['potential_label'] != 'below_average']
    pivot_df = df.pivot_table(index=["player_id", "position_id", "potential_label"], columns="attribute_id",
                              values="attribute_value")

    pivot_df = pivot_df.reset_index()
    pivot_df.columns = pivot_df.columns.astype(str)

    return pivot_df

df = sc_attr.merge(sc_pot_labels, on=["task_response_id", "match_id", "evaluator_id", "player_id"], how="outer")
pivot_df = data_prep(df)


# 1- Feature Extraction
num_cols = [col for col in pivot_df.columns if pivot_df[col].dtypes != 'O']
num_cols = num_cols[2:]

numeric_data = pd.DataFrame()
for feature in num_cols:
    numeric_data[feature] = pivot_df[feature]
corr_data = numeric_data.corr().abs()

# Korelasyon matrisini üst üçgene dönüştür
upper_corr = corr_data.where(np.triu(np.ones(corr_data.shape), k=1).astype(bool))

min_corr = 0.5
max_corr = 0.7
unique_features = set()
for col in upper_corr.columns:
    for idx, value in upper_corr[col].items():
        if min_corr <= value <= max_corr:
            unique_features.add(idx)
            unique_features.add(col)

unique_features_list = list(unique_features)

pivot_df['high_corr_ort'] = pivot_df[unique_features_list].mean(axis=1)
pivot_df['high_corr_std'] = pivot_df[unique_features_list].std(axis=1)


# 2 - Feature Extraction   ***
best_pairs = list(combinations(best_features, 2))
print("Etkileşimli Değişken Adayları:", best_pairs)

new_features = {}
for col1, col2 in best_pairs:
    new_features[f"BEST_{col1}_x_{col2}"] = pivot_df[col1] * pivot_df[col2]
    new_features[f"BEST_{col1}_plus_{col2}"] = pivot_df[col1] + pivot_df[col2]
    new_features[f"BEST_{col1}_minus_{col2}"] = pivot_df[col1] - pivot_df[col2]

pivot_df = pd.concat([pivot_df, pd.DataFrame(new_features)], axis=1)

# 3 - Feature Extraction
num_cols = [col for col in pivot_df.columns if pivot_df[col].dtypes != 'O']
pivot_df["min"] = pivot_df[num_cols].min(axis=1)
pivot_df["max"] = pivot_df[num_cols].max(axis=1)  # *
pivot_df["sum"] = pivot_df[num_cols].sum(axis=1)
pivot_df["mean"] = pivot_df[num_cols].mean(axis=1)
pivot_df["median"] = pivot_df[num_cols].median(axis=1)
pivot_df["mentality"] = pivot_df["position_id"].apply(lambda x: "defender" if (x == 2) | (x == 3) | (x == 4) | (x == 5) else "attacker")
pivot_df["position_group"] = pd.qcut(pivot_df["position_id"], q=3, labels=["low_pos", "mid_pos", "high_pos"])
pivot_df["position_label_combo"] = pivot_df["position_id"].astype(str) + "_" + pivot_df["potential_label"]


# 4- Feature Extraction
two_combinations = list(itertools.combinations(best_features, 2))

pairwise_features = []
for col1, col2 in two_combinations:
    new_col_name = f"{col1}_{col2}_diff"
    diff_series = (pivot_df[col1] - pivot_df[col2]).rename(new_col_name)
    pairwise_features.append(diff_series)

pairwise_df = pd.concat(pairwise_features, axis=1)

pivot_df = pd.concat([pivot_df, pairwise_df], axis=1)

# 3'lü kombinasyonlar için (örnek: ortalama)
three_list = [4325, 4329, 4332, 4333, 4335, 4338, 4343, 4344, 4349, 4350, 4353, 4354, 4407, 4423, 4426]
three_list = list(map(str, three_list))
three_combinations = list(itertools.combinations(three_list, 3))

triple_features = []
for col1, col2, col3 in three_combinations:
    new_col_name = f"{col1}_{col2}_{col3}_mean"
    mean_series = pivot_df[[col1, col2, col3]].mean(axis=1)
    triple_features.append(mean_series)

pairwise_df = pd.concat(triple_features, axis=1)
pivot_df = pd.concat([pivot_df, pairwise_df], axis=1)


# 5- Feature Extraction
high_corr_pairs = list(combinations(best_features, 2))
print("Etkileşimli Değişken Adayları:", high_corr_pairs)

new_features = {}
for col1, col2 in high_corr_pairs:
    new_features[f"{col1}_x_{col2}"] = pivot_df[col1] * pivot_df[col2]
    new_features[f"{col1}_plus_{col2}"] = pivot_df[col1] + pivot_df[col2]
    new_features[f"{col1}_minus_{col2}"] = pivot_df[col1] - pivot_df[col2]

# Hepsini tek seferde ekleyelim:
pivot_df = pd.concat([pivot_df, pd.DataFrame(new_features)], axis=1)


# Label Encoder
# labelEncoderCols = ["potential_label", "mentality","position_group","position_label_combo"]
# labelEncoderCols = ["potential_label", "mentality"]
#
# for col in labelEncoderCols:
#     pivot_df = label_encoder(pivot_df, col)
#
le_group = LabelEncoder()
le_combo = LabelEncoder()

# Fit ve transform işlemleri
pivot_df['position_group_encoded'] = le_group.fit_transform(pivot_df['position_group'])
pivot_df['position_label_combo_encoded'] = le_combo.fit_transform(pivot_df['position_label_combo'])

# Scaler
num_cols = [col for col in pivot_df.columns if pivot_df[col].dtypes != 'O']
num_cols = num_cols[3:]

scaler = StandardScaler()
pivot_df[num_cols] = scaler.fit_transform(pivot_df[num_cols])


#######################################################################
y = pivot_df["potential_label"]
X = pivot_df.drop(["player_id", "potential_label"], axis=1)


for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")



########## LR ##########
# Accuracy: 0.882
# Auc: 0.8543
# Recall: 0.64
# Precision: 0.7642
# F1: 0.6846
# ########## KNN ##########
# Accuracy: 0.8522
# Auc: 0.8127
# Recall: 0.3567
# Precision: 0.85
# F1: 0.4915
# ########## CART ##########
# Accuracy: 0.8636
# Auc: 0.8207
# Recall: 0.7433
# Precision: 0.6576
# F1: 0.6831
# ########## RF ##########
# Accuracy: 0.8817
# Auc: 0.9309
# Recall: 0.5433
# Precision: 0.8467
# F1: 0.6335
# ########## Adaboost ##########
# Accuracy: 0.9001
# Auc: 0.9216
# Recall: 0.7
# Precision: 0.8
# F1: 0.737
# ########## XGB ##########
# Accuracy: 0.8819
# Auc: 0.9145
# Recall: 0.67
# Precision: 0.7348
# F1: 0.6908
# ########## LightGBM ##########
# Accuracy: 0.8856
# Auc: 0.9343
# Recall: 0.6867
# Precision: 0.7719
# F1: 0.7109
# ########## CatBoost ##########
# Accuracy: 0.8816
# Auc: 0.9313
# Recall: 0.5033
# Precision: 0.875
# F1: 0.6177


#######################################################################
def lgbm_model(X, y):
    lgbm_model = LGBMClassifier(random_state=46)

    first_cvs = cross_val_score(lgbm_model, X, y, scoring="accuracy", cv=10).mean()

    lgbm_params = {"learning_rate": [0.005, 0.01, 0.1],
                   "n_estimators": [200, 500, 1500, 2000],
                   "colsample_bytree": [0.3, 0.5, 1],
                   "max_depth": [5, 7, 8]
                 }

    lgbm_gs_best = GridSearchCV(lgbm_model,
                                lgbm_params,
                                cv=5,
                                n_jobs=-1,
                                verbose=False).fit(X, y)

    # normal y cv süresi: 16.2s
    # scale edilmiş y ile: 13.8s

    lgbm_gs_best.best_params_
    final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

    final_cvs = pd.DataFrame(cross_validate(final_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"]))
    print(f"Accuracy: {round(final_cvs['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(final_cvs['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(final_cvs['test_recall'].mean(), 4)}")
    print(f"Precision: {round(final_cvs['test_precision'].mean(), 4)}")
    print(f"F1: {round(final_cvs['test_f1'].mean(), 4)}")

    # final_cvs = cross_val_score(final_model, X, y, scoring="accuracy", cv=10).mean()
    # print("İlk CVS:", first_cvs)
    # print("Final CVS:", final_cvs)
    return final_model

final_model = lgbm_model(X, y)
# final_model = lgbm_model(X_selected, y)

# Accuracy: 0.904
# Auc: 0.9341
# Recall: 0.6576
# Precision: 0.8469
# F1: 0.7298

#######################################################################
def rf_model(X_selected, y):
    model = RandomForestClassifier(random_state=42)
    cv_results = cross_validate(model, X_selected, y, cv=5,
                                scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

    print("Accuracy:", cv_results["test_accuracy"].mean())
    print("F1 Score:", cv_results["test_f1"].mean())
    print("ROC AUC:", cv_results["test_roc_auc"].mean())
    print("precision:", cv_results["test_precision"].mean())
    print("recall:", cv_results["test_recall"].mean())

rf_model(X, y)
# rf_model(X_selected, y)



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

plot_importance(final_model, X, 50)

# plot_importance(final_model, X_selected, 50)