#!/usr/bin/env python
# coding: utf-8

# # 1. EDA

# In[6]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from ydata_profiling import ProfileReport
from scipy import stats
from scipy import special

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 500)

df_train = pd.read_csv("train_test_submission/train.csv")
df_train_Id = df_train["Id"]
df_train = df_train.drop("Id", axis=1)

df_test = pd.read_csv("train_test_submission/test.csv")
df_test_Id = df_test["Id"]
df_test = df_test.drop("Id", axis=1)

print(f"df_train.shape: {df_train.shape}")
display(df_train.head(5))
print(f"df_train.shape: {df_test.shape}")
display(df_test.head(5))


# In[7]:


print("-" * 10, "df_train.info()", "-" * 10)
print(df_train.info())

print("-" * 10, "df_test.info()", "-" * 10)
print(df_test.info())


# In[8]:


# # ydata_profilingを使う場合。時間かかるので注意

# if not os.path.exists("ydata_profiling"):
#     os.makedirs("ydata_profiling")

# profile = ProfileReport(df_train, minimal=True)
# profile.to_file("ydata_profiling/kaggle_houseprices_minimal.html")

# # profile = ProfileReport(df_train, minimal=False)
# # profile.to_file("ydata_profiling/kaggle_houseprices.html")


# In[9]:


print("-" * 10, 'df_train["SalePrice"].describe()', "-" * 10)
print(df_train["SalePrice"].describe())

# SalePriceの分布
plt.hist(x=df_train["SalePrice"], bins=50)
plt.xlabel("SalePrice")
plt.suptitle("SalePriceの分布")
plt.show()


# In[10]:


corr_matrix = df_train.corr(numeric_only=True)
"""
    訓練データdf_trainの相関係数行列
    corr_matrix = df_train.corr(numeric_only=True)
"""

plt.figure(figsize=(12, 10))
sns.heatmap(abs(corr_matrix), annot=True, fmt=".1f", annot_kws={"fontsize": 6})

plt.suptitle("訓練データの相関係数(絶対値)行列")
plt.show()


# In[11]:


threshold = 0.6
high_corr_cols = (
    corr_matrix["SalePrice"][abs(corr_matrix["SalePrice"]) >= threshold]
    .sort_values(ascending=False)
    .index
).drop("SalePrice")

# これらについて、SalePriceに対する散布図を描写したいね
# matplotlibのsubplotsの取り扱いに習熟したい

# プロットのサイズを指定 (行数と列数は自由に調整可能)
num_cols = len(high_corr_cols)
fig, axes = plt.subplots(nrows=(num_cols // 3 + 1), ncols=3, figsize=(15, 5 * (num_cols // 3 + 1)))

# high_corr_colsにある特徴量ごとに散布図を描く
for ax, col in zip(axes.flatten(), high_corr_cols):
    sns.scatterplot(x=df_train[col], y=df_train["SalePrice"],alpha=0.3 , ax=ax)
    ax.set_title(f"{col} vs SalePrice. 相関係数: {corr_matrix["SalePrice"][col]:.3f}")

# グラフのレイアウトを自動調整
plt.suptitle(f"SalePriceとの相関係数の絶対値が{threshold}以上の特徴量についての散布図")
plt.tight_layout()
plt.show()


# # 2. 前処理(とりあえずlightGBMで回すために)

# In[12]:


# lightGBMに突っ込むためには数値型(またはbool型)である必要があるので、object型のデータをlabel encodingで処理する
# https://qiita.com/Hyperion13fleet/items/afa49a84bd5db65ffc31　こっちのほうが便利？

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor
from sklearn.metrics import root_mean_squared_error as rmse

# object型のデータが入っている列を抽出
object_columns = df_train.select_dtypes(include="object").columns
df_train_pre_encoding = df_train.copy()
df_test_pre_encoding = df_test.copy()

# ラベルエンコーディング
oe = OrdinalEncoder()
df_train[object_columns] = oe.fit_transform(df_train[object_columns])
df_test[object_columns] = oe.fit_transform(df_test[object_columns])

# display(df_train_pre_encoding.head(10))
# display(df_train.head(10))


# In[13]:


X = df_train.drop(["SalePrice"], axis=1)
y = df_train["SalePrice"]

# クロスバリデーション
kf = KFold(n_splits=4, shuffle=True, random_state=42)

scores = []

for tr_idx, va_idx in kf.split(X):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = LGBMRegressor(max_depth=-1)
    # GBDTのパラメータについて。https://knknkn.hatenablog.com/entry/2021/06/29/125226
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)
    score = rmse(np.log10(y_pred), np.log10(y_va))
    scores.append(score)

print(f"\n\nThe score is {np.mean(scores)}.")

# 次にやること：max_depthのチューニングってどうやればいいだろう？てか一回さっさと提出してみないか？
# メモ：[LightGBM] [Warning] No further splits with positive gain, best gain: -infについて
# これは「決定木の作成中、これ以上分岐を作っても予測誤差が下がらなかったのでこれ以上分岐をさせなかった」ことを意味するらしい


# In[14]:


# 一度このまま提出用のデータを出力してしまおう
model = LGBMRegressor(max_depth=-1)
model.fit(X, y)
sub_pred = model.predict(df_test)
submission = pd.DataFrame({"Id": df_test_Id, "SalePrice": sub_pred})
submission.to_csv("train_test_submission\submission.csv", index=False)


# In[ ]:




