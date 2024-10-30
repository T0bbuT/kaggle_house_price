#!/usr/bin/env python
# coding: utf-8

# # 1. EDA

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
from ydata_profiling import ProfileReport
from scipy import stats
from scipy import special

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

df_train = pd.read_csv("train_test_submission/train.csv")
df_train_Id = df_train["Id"]
df_train = df_train.drop("Id", axis=1)

df_test = pd.read_csv("train_test_submission/test.csv")
df_test_Id = df_test["Id"]
df_test = df_test.drop("Id", axis=1)

df_all_data = pd.concat([df_train, df_test])

print(f"{df_train.shape=}")
display(df_train.head(5))
print(f"{df_test.shape=}")
display(df_test.head(5))

print("-" * 10, "df_train.info()", "-" * 10)
print(df_train.info())
print("\n", "-" * 10, "df_test.info()", "-" * 10)
print(df_test.info())



# # ydata_profilingを使う場合。時間かかるので注意

# if not os.path.exists("ydata_profiling"):
#     os.makedirs("ydata_profiling")

# profile = ProfileReport(df_all_data, minimal=True)
# profile.to_file("ydata_profiling/kaggle_houseprices_minimal.html")

# # profile = ProfileReport(df_all_data, minimal=False)
# # profile.to_file("ydata_profiling/kaggle_houseprices.html")


# In[2]:


print("-" * 10, "df_train.columns", "-" * 10)
print(df_train.columns)


# In[3]:


SalePrice = df_train["SalePrice"]

skewness_SalePrice = SalePrice.skew()
kurtosis_SalePrice = SalePrice.kurtosis()

print("-" * 10, 'df_train["SalePrice"].describe()', "-" * 10)
print(df_train["SalePrice"].describe())

print(f"{skewness_SalePrice=}")
print(f"{kurtosis_SalePrice=}")

# plotlyではkdeを描写するのが面倒っぽいのでseabornで描写
fig, ax = plt.subplots(1, 2,figsize=(10, 4))

sns.histplot(SalePrice, stat="density", kde=True, ax=ax[0])
ax[0].set_title("ヒストグラムと正規分布(黒線)")
ax[0].tick_params(axis="x", labelsize=8, rotation=20)

xmin, xmax = ax[0].get_xlim()
x = np.linspace(xmin, xmax, 100)
y_norm = stats.norm.pdf(x, np.mean(SalePrice), np.std(SalePrice))
ax[0].plot(x, y_norm, "k", linewidth=1)

stats.probplot(SalePrice, plot=ax[1])
ax[1].set_title("正規確率プロット")

plt.tight_layout()
plt.show()


# In[4]:


numeric_columns = df_train.select_dtypes(include="number").columns

df_skew_kurt_train = pd.DataFrame({
    "Feature": numeric_columns,
    "Skewness": [stats.skew(df_train[col], nan_policy="omit") for col in numeric_columns],
    "Kurtosis": [stats.kurtosis(df_train[col], nan_policy="omit") for col in numeric_columns]
})

display(df_skew_kurt_train.sort_values(by="Skewness", ascending=False))


# In[5]:


corr_matrix = df_train.corr(numeric_only=True)
"""
    訓練データdf_trainの相関係数行列
    corr_matrix = df_train.corr(numeric_only=True)
"""

plt.figure(figsize=(12, 10))
sns.heatmap(abs(corr_matrix), annot=True, fmt=".1f", annot_kws={"fontsize": 6})

plt.suptitle("訓練データの相関係数(絶対値)行列_カテゴリ変数を除く")
plt.show()


# In[6]:


# plotly版。インデックス番号が一目で確認できる

import plotly.express as px
import plotly.subplots as sp

threshold = 0.6
high_corr_cols = (
    corr_matrix["SalePrice"][abs(corr_matrix["SalePrice"]) >= threshold]
    .sort_values(ascending=False)
    .index
).drop("SalePrice")

# プロットのサイズを指定
plot_size = len(high_corr_cols)
rows = plot_size // 3 + 1  # 行数
cols = 3  # 列数

# サブプロットの作成
fig = sp.make_subplots(
    rows=rows, 
    cols=cols, 
    subplot_titles=[f"{col} vs SalePrice （相関係数{corr_matrix["SalePrice"][col]:.3f}）" for col in high_corr_cols],
    horizontal_spacing=0.05,
    vertical_spacing=0.1,
    )

# high_corr_colsにある特徴量ごとに散布図を描く
for i, col in enumerate(high_corr_cols):
    row = i // cols + 1
    col_num = i % cols + 1
    scatter = px.scatter(df_train, x=col, y="SalePrice", opacity=0.3, hover_data=[df_train.index])
    for trace in scatter.data:
        fig.add_trace(trace, row=row, col=col_num)
    fig.update_annotations()

# グラフのタイトルを設定
fig.update_layout(
    title_text=f"SalePriceとの相関係数の絶対値が{threshold}以上の特徴量についての散布図",
    showlegend=False,
    height=400 * rows,
    width=1200,
)

# グラフの表示
fig.show()

# レイアウト調節 https://data-analytics.fun/2021/06/19/plotly-subplots/


# # 2. 前処理

# ## 外れ値処理

# In[7]:


# 外れ値処理(訓練データ)
# 外れ値のインデックス番号は、plotlyで描いたグラフから得た
df_train_befdrop = df_train
df_train = df_train.drop(df_train.index[[523, 1298]])

fig = px.scatter(
    df_train, x="GrLivArea", y="SalePrice",
    opacity=0.3,
    hover_data=[df_train.index]
)

fig.update_layout(
    title_text="SalePrice vs GrLivArea. 外れ値処理後",
    showlegend=False,
    height=500,
    width=600
)

# グラフの表示
fig.show()


# ## 欠損値補完・列削除

# In[8]:


# 欠損値処理(訓練データ、テストデータ)
df_all_data = pd.concat([df_train, df_test])

df_missing_values_count = df_all_data.isna().sum()
df_missing_values_table = pd.DataFrame(
    {
        "Missing_count": df_missing_values_count,
        "Percent (%)": round(df_missing_values_count / len(df_all_data) * 100, 2)
    }
).sort_values("Missing_count", ascending=False)

# chatGPTに作ってもらった各特徴量の説明をまとめたcsvを読み込み、欠損値に関する表と結合
df_data_description = pd.read_csv("data_description/data_descripsion_simple_jp.csv", index_col=0)
df_missing_value_description = pd.concat([df_missing_values_table, df_data_description], axis=1)

# csvに出力。これとydata_profilingのレポートを眺めながら各欠損値をどう処理するか考える。
if not os.path.exists("missing_value"):
    os.makedirs("missing_value")
df_missing_value_description.to_csv(
    "missing_value/missing_value_processing.csv", encoding="utf-8_sig"
)

display(df_missing_value_description.head(15))


# In[9]:


# LotFrontageの欠損割合が多いが、何で補完するかが難しい。どれかのカテゴリ変数に対する傾向がないか調べてみる

# object型のデータが入っている列を抽出
object_columns = df_all_data.select_dtypes(include="object").columns

# プロットのサイズを指定
plot_size = len(object_columns)
rows = plot_size // 6 + 1  # 行数
cols = 6  # 列数

# サブプロットの作成
fig = sp.make_subplots(
    rows=rows, 
    cols=cols, 
    subplot_titles=[f"{col} vs LotFrontage" for col in object_columns],
    )

# object_columnsにある特徴量ごとに箱ひげ図を描く
for i, col in enumerate(object_columns):
    row = i // cols + 1
    col_num = i % cols + 1
    box = px.box(df_all_data, x=col, y="LotFrontage")
    for trace in box.data:
        fig.add_trace(trace, row=row, col=col_num)
    fig.update_annotations()

# グラフのタイトルを設定
fig.update_layout(
    title_text=f"各カテゴリ変数に対するLotFrontageの箱ひげ図",
    showlegend=False,
    height=400 * rows,
    width=1600,
)

# グラフの表示
fig.show()


# In[10]:


# x="Neighborhood", y="LotFrontage"が傾向を捉えていそう。詳しく確認する

fig = px.box(df_all_data, x="Neighborhood", y="LotFrontage")

fig.update_layout(
    # title_text=" ",
    showlegend=False,
    height=500,
    width=1000
)

# グラフの表示
fig.show()


# In[11]:


# 各地域"Neighborhood"の"LotFrontage"の中央値で欠損値を補完する

df_medLot_groupby_Neighborhood = df_all_data.groupby(by="Neighborhood")["LotFrontage"].agg("median")

def fillnaLot(row):
    """
    ある1つの住宅データについて、"LotFrontage"列の値が欠損している場合はそのデータの地域（"Neighborhood"）の"LotFrontage"の中央値を返す。
    欠損していない場合、元の値をそのまま返す。

    Args:
        row (pd.Series): "LotFrontage"列の欠損値処理をしたいデータ

    Return
    -------
        "LotFrontage"列が…
            欠損の場合: df_group_LotFrontage[row["Neighborhood"]]
            欠損でない場合: row["LotFrontage"]
    """
    if pd.isna(row["LotFrontage"]):
        return df_medLot_groupby_Neighborhood[row["Neighborhood"]]
    else:
        return row["LotFrontage"]


# In[12]:


# LotFrontageの補完
df_all_data["LotFrontage"] = df_all_data.apply(fillnaLot, axis=1)

# "None"で補完
cols_fillNone = [
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "GarageType",
    "BsmtCond",
    "BsmtExposure",
    "BsmtQual",
    "BsmtFinType2",
    "BsmtFinType1"    
]
# 0で補完
cols_fill0 = [
    "GarageYrBlt",
    "MasVnrArea",
    "BsmtHalfBath",
    "BsmtFullBath",
    "GarageArea",
    "GarageCars",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF"
]
# 最頻値で補完
cols_fillmode = [
    "MSZoning",
    "Functional",
    "Exterior2nd",
    "Exterior1st",
    "SaleType",
    "KitchenQual",
    "Electrical"
]
# 列削除：PoolQC(99.7%が欠損)、Utilities(99.6%が"allpub")、PoolArea(99.6%が0)
cols_drop = [
    "PoolQC",
    "Utilities",
    "PoolArea"
]

for col in cols_fillNone:
    df_all_data[col] = df_all_data[col].fillna("None")
for col in cols_fill0:
    df_all_data[col] = df_all_data[col].fillna(0)
for col in cols_fillmode:
    df_all_data[col] = df_all_data[col].fillna(df_all_data[col].mode()[0])
df_all_data = df_all_data.drop(columns=cols_drop)


# ## 新たな特徴量の作成(訓練データ、テストデータ)

# In[13]:


# 新しい特徴量の作成
# 'YrBltAndRemod': 'YearBuilt' + 'YearRemodAdd'

df_all_data["TotalSF"] = (
    df_all_data["TotalBsmtSF"]
    + df_all_data["1stFlrSF"] 
    + df_all_data["2ndFlrSF"]
)
df_all_data["TotalFinSF"] = (
    df_all_data["BsmtFinSF1"]
    + df_all_data["BsmtFinSF2"]
    + df_all_data["1stFlrSF"]
    + df_all_data["2ndFlrSF"]
)
df_all_data["TotalBathrooms"] = (
    df_all_data["BsmtFullBath"]
    + 0.5 * df_all_data["BsmtHalfBath"]
    + df_all_data["FullBath"]
    + 0.5 * df_all_data["HalfBath"]
)
df_all_data["TotalPorchSF"] = (
    df_all_data["3SsnPorch"]
    + df_all_data["EnclosedPorch"]
    + df_all_data["OpenPorchSF"]
    + df_all_data["ScreenPorch"]
)

df_all_data["has2ndfloor"] = df_all_data["2ndFlrSF"] > 0
df_all_data["hasGarage"] = df_all_data["GarageArea"] > 0
df_all_data["hasBsmt"] = df_all_data["TotalBsmtSF"] > 0
df_all_data["hasFireplace"] = df_all_data["Fireplaces"] > 0

df_all_data[[
    "TotalSF",
    "TotalFinSF",
    "TotalBathrooms",
    "TotalPorchSF",
    "has2ndfloor",
    "hasGarage",
    "hasBsmt",
    "hasFireplace"    
]].head(5)


# ## カテゴリ変数のエンコーディング

# In[14]:


# カテゴリ変数のエンコーディング

# lightGBMに突っ込むためには数値型(またはbool型)である必要があるので、object型のデータをlabel encodingで処理する
# https://qiita.com/Hyperion13fleet/items/afa49a84bd5db65ffc31　こっちのほうが便利？

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import PowerTransformer

# object型のデータが入っている列を抽出
object_columns = df_all_data.select_dtypes(include="object").columns
# エンコード前に退避
df_all_data_pre_encoding = df_all_data.copy()

# one-hot encoding
df_all_data = pd.get_dummies(df_all_data).reset_index(drop=True)

print(f"{df_all_data_pre_encoding.shape=}")
display(df_all_data_pre_encoding.head(3))
print(f"{df_all_data.shape=}")
display(df_all_data.head(3))


# ## 数値変換

# In[15]:


# 実務上の運用を想定し、数値変換の方針は訓練データのみを使って得る

# まず、df_all_dataをdf_trainとdf_testに分割し直す
ntrain = len(df_train)

df_train = df_all_data[:ntrain]
df_test = df_all_data[ntrain:].drop(["SalePrice"], axis=1)

SalePrice = df_train["SalePrice"]

# 目的変数について
SalePrice_aft_boxcox, lambda_SalePrice = stats.boxcox(SalePrice)
print(f"{lambda_SalePrice=}")
print(f"Skewness of SalePrice after boxcox: {stats.skew(SalePrice_aft_boxcox)}")
print(f"Kurtosis of SalePrice after boxcox: {stats.kurtosis(SalePrice_aft_boxcox)}")


# In[26]:


# 特徴量の数値変換は、bool型を除いた数値型の特徴量についてのみ行う
numeric_columns = df_train.select_dtypes(include="number").columns

df_skew_kurt_train = pd.DataFrame({
    "Feature": numeric_columns,
    "Skewness": [stats.skew(df_train[col], nan_policy="omit") for col in numeric_columns],
    "Kurtosis": [stats.kurtosis(df_train[col], nan_policy="omit") for col in numeric_columns]
})

# display(df_skew_kurt_train.sort_values(by="Skewness", ascending=False))

# ここで、skewnessが高いもののみ抽出する！

# from sklearn.preprocessing import PowerTransformer を使うのが良さそう！！
pt = PowerTransformer(method="yeo-johnson")
pt.fit(df_train[numeric_columns])
df_lambdas = pd.DataFrame({
    "Feature": numeric_columns,
    "lambda": pt.lambdas_
})

display(df_lambdas)


# # 3. モデル構築

# In[36]:


# モデル構築

X = df_train.drop(["SalePrice"], axis=1)
y = df_train["SalePrice"]

# クロスバリデーション
kf = KFold(n_splits=5, shuffle=True, random_state=42)

params_lgbm = {}
params_ridge = {}
params_lasso = {}
# params_lgbm = {"max_depth": 19, "learning_rate": 0.1}
# パラメータチューニングにはoptunaというのを使うと良いらしい
# https://qiita.com/tetsuro731/items/a19a85fd296d4b87c367
# https://qiita.com/tetsuro731/items/76434194bab336a97172
# GBDTのパラメータについて。https://knknkn.hatenablog.com/entry/2021/06/29/125226

models = [
    LGBMRegressor(**params_lgbm),
    Ridge(**params_ridge),
    Lasso(**params_lasso)
]

for model in models:
    model_name = model.__class__.__name__
    print("-" * 10, f"{model_name=}", "-" * 10)
    scores = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
        print(f"分割 {fold_idx + 1} / {kf.n_splits}")

        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        
        # 予測値に負の値が含まれているかをチェック
        flag_neg = np.any(y_pred <= 0)
        if flag_neg:
            print("予測値に0以下の値が含まれています。スコア算出のためこれらの予測値は十分小さい正の値1e-6に変換されます。")
            y_pred  = np.maximum(y_pred, 1e-6)    

        score = rmse(np.log(y_pred), np.log(y_va))
        mape_ = mape(y_pred, y_va) * 100
        scores.append(score)
        print(f"Score: {score}")
        print(f"MAPE(平均絶対誤差率): {mape_:.2f}%")

    print(f"\n分割した計{fold_idx + 1}個のモデルのスコアの平均値: {np.mean(scores)}\n")   

# メモ：[LightGBM] [Warning] No further splits with positive gain, best gain: -infについて
    # これは「決定木の作成中、これ以上分岐を作っても予測誤差が下がらなかったのでこれ以上分岐をさせなかった」ことを意味するらしい
# メモ：Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation.について
    # これは「回帰モデルの目的関数が収束せず、推定結果が安定していない可能性」を示唆している。最大反復回数の増加、データのスケーリング、正則化パラメータの調整を試すと良い。


# # 4. 提出

# In[15]:


# 提出用のデータを出力
model = LGBMRegressor(max_depth=-1)
model.fit(X, y)
sub_pred = model.predict(df_test)
flag_neg = np.any(sub_pred < 0)
if flag_neg:
    print("予測値に0以下の値が含まれています。スコア算出のためこれらの予測値は十分小さい正の値1e-6に変換されます。")
    sub_pred  = np.maximum(sub_pred, 0)  
submission = pd.DataFrame({"Id": df_test_Id, "SalePrice": sub_pred})
submission.to_csv("train_test_submission/submission.csv", index=False)


# In[16]:


# 学習結果の図示
tree_idx = 0
print(f"{tree_idx + 1}番目の木の様子は以下の通り")

plot_tree(model, tree_index=tree_idx, figsize=(20, 10))

# 特徴量重要度
df_feature_importances = pd.DataFrame(
    {"feature_name": model.feature_name_, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

# 重要度が一定以上のものだけ抽出
threshold = 5.0
df_feature_importances_filterd = df_feature_importances[df_feature_importances["importance"] >= threshold]

plt.figure(figsize=(16, 8))
sns.barplot(data=df_feature_importances_filterd, x="feature_name", y="importance")
plt.suptitle(f"特徴量重要度(≧{threshold}のものを抽出)")
plt.xticks(rotation=90)
plt.show()

