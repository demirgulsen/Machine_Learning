import datetime

###############################################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu (Customer Segmentation with Unsupervised Learning)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################

# Unsupervised Learning yöntemleriyle (Kmeans, Hierarchical Clustering )  müşteriler kümelere ayrılıp ve davranışları gözlemlenmek istenmektedir.

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline) olarak yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

# 20.000 gözlem, 13 değişken

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type : 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A,B şeklinde yazılmıştır.


###############################################################
# GÖREVLER
###############################################################

# GÖREV 1: Veriyi Hazırlama
# 1. flo_data_20K.csv.csv verisini okuyalım ve bir dataframe' e atalım.

import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

df = pd.read_csv('Github_Machine_Learning/CaseStudy_3/dataset/flo_data_20k.csv')

df.head()
df.shape
df.info()
df.isnull().sum()
df.describe().T

# Önce tarih içeren değişkenlerin tip dönüşümünü yapalım
date_cols = [col for col in df.columns if 'date' in col]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors='coerce')

df.info()

# 2. Müşterileri segmentlerken kullanacağımız değişkenleri seçelim. Tenure(Müşterinin yaşı), Recency (en son kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.

# veri seti eski olduğu için analiz tarihini biz belirleyelim
analysis_date = df['last_order_date'].max() + pd.Timedelta(days=2)
df['recency'] = df['last_order_date'].apply(lambda x: analysis_date - x)
df['tenure'] = df['last_order_date'] - df['first_order_date']
df['frequency'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['monetary'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']

df.head()

# Adım 5: Aykırı gözlem analizi yapınız.

numerical_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]

def outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    quartile1 = dataframe[variable].quantile(q1)
    quartile3 = dataframe[variable].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    if not outliers.empty:  # Eğer outlier'lar varsa
        return True
    else:
        return False

for col in numerical_cols:
    print(col, check_outlier(df, col))

# Aykırı değer bulunan sütunları döndüren fonksiyon
def get_outliers(dataframe, num_cols):
    outlier_cols = []
    for col in numerical_cols:
        if check_outlier(dataframe, col):
            outlier_cols.append(col)
    return outlier_cols


# cat_cols, num_cols, cat_but_car = grab_col_names(df)
outlier_columns = get_outliers(df, numerical_cols)

# Aykırı değerleri temizleyelim
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = int(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = int(up_limit)


for col in outlier_columns:
    replace_with_thresholds(df, col)

for col in numerical_cols:
    print(col, check_outlier(df, col))


# num_cols_without_date = [col for col in num_cols if 'date' not in col]
new_num_cols = numerical_cols
new_num_cols.extend(['recency', 'tenure'])

model_df = df[new_num_cols]
model_df.head()

################################################################################
# GÖREV 2: K-Means ile Müşteri Segmentasyonu

# 1. Değişkenleri standartlaştıralım.
# Değişkenlerdeki çarpıklıkları kontrol edelim
def check_skew(df_skew, column):
    """
    Verilen sütunlar için dağılım grafiklerini ve çarpıklık değerlerini görselleştirir.
    """
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.histplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return


convert_val = ['recency', 'tenure']
for col in convert_val:
    # Ensure the conversion to total seconds as float is done correctly
    model_df.loc[:, col] = model_df[col].dt.total_seconds().astype('float64')


model_df[['recency', 'tenure']].info()

plt.figure(figsize=(9, 9))  # Grafik boyutunu ayarlıyoruz

# Her bir sütun için bir subplot oluşturuyoruz
for i, col in enumerate(numerical_cols, start=1):
    plt.subplot(len(numerical_cols), 1, i)  # Alt grafik sayısını değişken sayısına göre ayarlıyoruz
    sns.histplot(model_df[col], kde=True, color="g", bins=20)  # Her bir sütun için histogram ve KDE ekliyoruz
    plt.title(f'Distribution of {col}')  # Başlık olarak sütun adını yazıyoruz
    plt.xlabel('Value')  # X ekseni etiketi
    plt.ylabel('Count')  # Y ekseni etiketi

plt.tight_layout()
plt.show()

# Normal dağılımın sağlanması için değişkenlere Log transformation uygulayalım
for col in model_df.columns:
    model_df.loc[:, col] = np.log1p(model_df[col])


# Değişkenleri standartlaştıralım
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()

# 2. Optimum küme sayısını belirleyelim
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

elbow.elbow_value_


# 3. Model oluşturalım ve müşterileri segmentlere ayıralım.
# cluster sayısını optimum noktaya göre ayarlayalım
kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(model_df)
segments = kmeans.labels_

model_df['segment'] = segments

# 4. Herbir segmenti istatistiksel olarak inceleyeniz.
model_df.groupby("segment").agg(["mean", "min", "max"])

################################################################################
# GÖREV 3: Hierarchical Clustering ile Müşteri Segmentasyonu
# 1. Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

# linkage : birleştirici clustiring yöntemi - öklid uzaklık yöntemini kulanarak kümeleme yapar
# Yani; kümeleme sırasında iki küme arasındaki mesafenin nasıl hesaplanacağını belirler.
hc_average = linkage(model_df, "complete")

# hc_average = linkage(model_df, "average")  # bu da kullanılabilir

# complete linkage (tam bağlantı):
# İki küme arasındaki en uzak noktalar arasındaki mesafeyi kullanır.
# Daha kompakt ve sıkı kümeler üretir.
# Outlier'lara duyarlıdır çünkü en uçtaki mesafe esas alınır.

# average linkage (ortalama bağlantı):
# İki küme arasındaki tüm nokta çiftlerinin ortalama mesafesini alır.
# Daha dengeli ve yumuşak geçişli kümeler oluşturur.
# Outlier'lara karşı daha az hassastır.


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=1.2, color='r', linestyle='--')
plt.show()


# 2. Modelinizi oluşturunuz ve müşterileriniz segmentleyiniz.
cluster = AgglomerativeClustering(n_clusters=5)
clusters = cluster.fit_predict(model_df)
model_df['hc_segments'] = clusters

model_df.head()

# 3. Herbir segmenti istatistiksel olarak inceleyeniz.
model_df.groupby("hc_segments").agg(["mean", "min", "max"])

