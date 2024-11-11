#!/usr/bin/env python
# coding: utf-8

# # 1. EDA

# In[3]:


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
pd.set_option("display.max_rows", 100)

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



# ydata_profilingを使う場合。時間かかるので注意
# minimal=Falseにすると更に時間がかかり、出力されるhtmlも非常に重くなるなので注意

if not os.path.exists("ydata_profiling"):
    os.makedirs("ydata_profiling")

profile = ProfileReport(df_all_data, minimal=True)
profile_path = "ydata_profiling/kaggle_houseprices_minimal.html"
profile.to_file(profile_path)

print(f"{profile_path}にレポートが出力されました。")


# In[ ]:


SalePrice = df_train["SalePrice"]

skewness_SalePrice = SalePrice.skew()
kurtosis_SalePrice = SalePrice.kurtosis()

print("-" * 10, 'df_train["SalePrice"].describe()', "-" * 10)
print(df_train["SalePrice"].describe())

print(f"{skewness_SalePrice=}")
print(f"{kurtosis_SalePrice=}")

# plotlyではkdeを描写するのが面倒なのでseabornで描写
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


# In[5]:


df_all_data_features = df_all_data.drop(["SalePrice"], axis=1)
numeric_features = df_all_data_features.select_dtypes(include="number").columns


df_skew_kurt = pd.DataFrame({
    "Feature": numeric_features,
    "Skewness": [stats.skew(df_all_data_features[col], nan_policy="omit") for col in numeric_features],
    "Kurtosis": [stats.kurtosis(df_all_data_features[col], nan_policy="omit") for col in numeric_features]
})

display(df_skew_kurt.sort_values(by="Skewness", ascending=False).head(10))


# In[6]:


corr_matrix = df_train.corr(numeric_only=True)
"""
    訓練データdf_trainの相関係数行列
    corr_matrix = df_train.corr(numeric_only=True)
"""

plt.figure(figsize=(12, 10))
sns.heatmap(abs(corr_matrix), cmap="viridis", annot=True, fmt=".1f", annot_kws={"fontsize": 6})

plt.suptitle("訓練データの相関係数(絶対値)行列_カテゴリ変数を除く")
plt.show()


# In[7]:


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

# In[8]:


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

# In[9]:


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


# In[10]:


# LotFrontageの欠損割合が多いが、何で補完するかが難しい。どれかのカテゴリ変数に対する傾向がないか調べてみる

# object型のデータが入っている列を抽出
object_cols = df_all_data.select_dtypes(include="object").columns

# プロットのサイズを指定
plot_size = len(object_cols)
rows = plot_size // 6 + 1  # 行数
cols = 6  # 列数

# サブプロットの作成
fig = sp.make_subplots(
    rows=rows, 
    cols=cols, 
    subplot_titles=[f"{col} vs LotFrontage" for col in object_cols],
    )

# object_colsにある特徴量ごとに箱ひげ図を描く
for i, col in enumerate(object_cols):
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


# In[11]:


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


# In[12]:


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


# In[13]:


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

# In[14]:


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

# In[15]:


from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.preprocessing import StandardScaler, PowerTransformer

# 1. カテゴリごとのユニークな値を取得
category_mappings = {col: set(df_all_data[col].dropna().unique()) for col in df_all_data.select_dtypes(include=['object', 'category']).columns}

# 2. 同じカテゴリーリストを持つ変数をグループ化
from collections import defaultdict

grouped_categories = defaultdict(list)
for col, categories in category_mappings.items():
    grouped_categories[frozenset(categories)].append(col)

# 結果の表示
for categories, columns in grouped_categories.items():
    print("カテゴリーリスト:", categories)
    print("同じマッピングを持つ変数:", columns)
    print()


# In[16]:


# 変換前
display(df_all_data.head(5))

# 順序を定義する
mapping = {
    'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],
    'KitchenQual': ['Fa', 'TA', 'Gd', 'Ex'],
    'ExterCond': ['Po', 'Fa', 'TA', 'Gd'],
    'HeatingQC': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtQual': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
    'BsmtCond': ['None', 'Po', 'Fa', 'TA', 'Gd'],
    'FireplaceQu': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageQual': ['None', 'Fa', 'TA', 'Gd', 'Ex'],
    'GarageCond': ['None', 'Po', 'TA', 'Gd', 'Ex'],
    'BsmtExposure': ['None', 'No', 'Mn', 'Av', 'Gd'],
    'BsmtFinType1': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'BsmtFinType2': ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'],
    'GarageFinish': ['None', 'Unf', 'RFn', 'Fin']
}

# 各変数を順序付きでエンコード
for col, order in mapping.items():
    df_all_data[col] = df_all_data[col].astype(pd.CategoricalDtype(categories=order, ordered=True))

# Ordinal encodingに変換
df_all_data = df_all_data.apply(lambda x: x.cat.codes if x.dtype.name == 'category' else x)

# 結果を表示
display(df_all_data.head(5))


# In[17]:


# 残りのカテゴリ変数をone-hot encodingする

# object型のデータが入っている列を抽出
object_cols = df_all_data.select_dtypes(include="object").columns

# one-hot encoding
df_all_data = pd.get_dummies(df_all_data).reset_index(drop=True)

display(df_all_data)


# ## 数値変換(目的変数、特徴量)

# In[ ]:


# 目的変数SalePriceの数値変換(box-cox)

# まず、ここまで使ってきたdf_all_dataをdf_trainとdf_testに分割し直す
ntrain = len(df_train)

df_train = df_all_data[:ntrain]
df_test = df_all_data[ntrain:].drop(["SalePrice"], axis=1)

# 全データ、訓練データを特徴量と目的変数に分ける
df_all_data_features = df_all_data.drop(["SalePrice"], axis=1)
df_train_features = df_train.drop(["SalePrice"], axis=1)
SalePrice = df_train["SalePrice"]

print("boxcox前")
print(f"{stats.skew(SalePrice)=}")
print(f"{stats.kurtosis(SalePrice)=}")

# SalePriceに対してBox-Cox変換の実行
SalePrice_boxcox, lambda_SalePrice_boxcox = stats.boxcox(SalePrice)

# 変換後のSalePriceを新しいDataFrameに保存し、元のインデックスを保持
df_SalePrice_boxcox = pd.DataFrame(SalePrice_boxcox, index=SalePrice.index, columns=["SalePrice_boxcox"])

print("boxcox後")
print(f"{stats.skew(SalePrice_boxcox)=}")
print(f"{stats.kurtosis(SalePrice_boxcox)=}")
print("Lambda value used for transformation:", lambda_SalePrice_boxcox)

fig, ax = plt.subplots(1, 2,figsize=(10, 4))
fig.suptitle("SalePriceの様子_boxcox後")

sns.histplot(SalePrice_boxcox, stat="density", kde=True, ax=ax[0])
ax[0].set_title("ヒストグラムと正規分布(黒線)")
ax[0].tick_params(axis="x", labelsize=8, rotation=20)

xmin, xmax = ax[0].get_xlim()
x = np.linspace(xmin, xmax, 100)
y_norm = stats.norm.pdf(x, np.mean(SalePrice_boxcox), np.std(SalePrice_boxcox))
ax[0].plot(x, y_norm, "k", linewidth=1)

stats.probplot(SalePrice_boxcox, plot=ax[1])
ax[1].set_title("正規確率プロット")

plt.tight_layout()
plt.show()


# In[19]:


# 特徴量の数値変換(yeo-johnson)

# 特徴量の数値変換は、bool型を除いた数値型の特徴量についてのみ行う
numeric_features = df_all_data_features.select_dtypes(include="number").columns

skewness = df_all_data_features[numeric_features].skew()
high_skew_features = skewness[skewness > 0.75].index

# yeo-johnson変換器を作成。学習
pt = PowerTransformer(method="yeo-johnson")
pt.fit(df_all_data_features[high_skew_features])
# 訓練データにyeo-johnson変換を実行
df_train_features[high_skew_features] = pt.transform(df_train_features[high_skew_features])
# テストデータにyeo-johnson変換を実行
df_test[high_skew_features] = pt.transform(df_test[high_skew_features])

# 訓練データにboxcoxしたSalePriceを結合
df_train = pd.concat([df_train_features, df_SalePrice_boxcox], axis=1)


# # 3. モデル構築

# ## モデルのパラメータチューニング

# In[20]:


import optuna
from sklearn.linear_model import Ridge, Lasso
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import numpy as np
from scipy import special

# データの準備
X = df_train.drop(["SalePrice_boxcox"], axis=1)
y = df_train["SalePrice_boxcox"]

# クロスバリデーション
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def objective(trial, model_name):
    scores = []
    
    # モデルごとのパラメータ範囲を設定
    if model_name == 'LGBMRegressor':
        params = {
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),  # 対数スケールで探索
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)  # 連続範囲で探索
        }
        model = LGBMRegressor(**params, verbose=-1)
    elif model_name == 'Ridge':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True)  # 対数スケールで探索
        }
        model = Ridge(**params)
    elif model_name == 'Lasso':
        params = {
            'alpha': trial.suggest_float('alpha', 1e-5, 1.0, log=True)  # 対数スケールで探索
        }
        model = Lasso(**params, max_iter=100000)
    
    # クロスバリデーションで評価
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        
        # 負の値を回避
        y_pred = np.maximum(y_pred, 1e-6)

        # 逆Box-Cox変換
        y_pred_inv_boxcox = special.inv_boxcox(y_pred, lambda_SalePrice_boxcox)
        y_va_inv_boxcox = special.inv_boxcox(y_va, lambda_SalePrice_boxcox)
        
        # 評価スコアをRMSEで計算
        score = rmse(np.log(y_pred_inv_boxcox), np.log(y_va_inv_boxcox))
        scores.append(score)
    
    return np.mean(scores)

# 各モデルの最適パラメータを保存する辞書
best_params_dict = {}

# モデルごとにOptunaでパラメータチューニング
for model_name in ['LGBMRegressor', 'Ridge', 'Lasso']:
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_name), n_trials=50)
    
    # 最適パラメータとスコアを表示
    print(f"\n{model_name}の最適パラメータ: {study.best_params}")
    print(f"最良スコア: {study.best_value}\n")
    
    # 最適パラメータを辞書に保存
    best_params_dict[model_name] = study.best_params

# 各モデルの最適パラメータが辞書に保存されていることを確認
print("各モデルの最適パラメータ一覧:")
for model_name, params in best_params_dict.items():
    print(f"{model_name}: {params}")


# # 4. 提出

# In[21]:


if not os.path.exists("train_test_submission"):
    os.makedirs("train_test_submission")

for model_name in ['LGBMRegressor', 'Ridge', 'Lasso']:
    params = best_params_dict[model_name]
    model = None

    if model_name == "LGBMRegressor":
        model = LGBMRegressor(**params, verbose=-1)
    elif model_name == "Ridge":
        model = Ridge(**params)
    elif model_name == "Lasso":
        model = Lasso(**params)

    # 学習・予測
    model.fit(X, y)
    pred = model.predict(df_test)
    sub_pred = special.inv_boxcox(pred, lambda_SalePrice_boxcox)
    sub_pred = np.maximum(sub_pred, 1e-6)

    # 提出データ作成
    submission = pd.DataFrame({"Id": df_test_Id, "SalePrice": sub_pred})
    submission_path = f"train_test_submission/submission_{model_name}.csv"
    submission.to_csv(submission_path, index=False)
    print(f"{model_name}の提出データが{submission_path}に出力されました。")


# In[ ]:




