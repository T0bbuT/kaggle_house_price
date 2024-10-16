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


# In[2]:


print("-" * 10, "df_train.info()", "-" * 10)
print(df_train.info())

print("-" * 10, "df_test.info()", "-" * 10)
print(df_test.info())


# In[3]:


# # ydata_profilingを使う場合。時間かかるので注意

# if not os.path.exists("ydata_profiling"):
#     os.makedirs("ydata_profiling")

# profile = ProfileReport(df_train, minimal=True)
# profile.to_file("ydata_profiling/kaggle_houseprices_minimal.html")

# # profile = ProfileReport(df_train, minimal=False)
# # profile.to_file("ydata_profiling/kaggle_houseprices.html")


# In[4]:


print("-" * 10, 'df_train["SalePrice"].describe()', "-" * 10)
print(df_train["SalePrice"].describe())

# SalePriceの分布
sns.histplot(df_train["SalePrice"], kde=True)
plt.suptitle("SalePriceの分布")
plt.show()


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


threshold = 0.6
high_corr_cols = (
    corr_matrix["SalePrice"][abs(corr_matrix["SalePrice"]) >= threshold]
    .sort_values(ascending=False)
    .index
).drop("SalePrice")

# プロットのサイズを指定 (行数と列数は自由に調整可能)
num_cols = len(high_corr_cols)
fig, axes = plt.subplots(
    nrows=(num_cols // 3 + 1), ncols=3, figsize=(15, 5 * (num_cols // 3 + 1))
)

# high_corr_colsにある特徴量ごとに散布図を描く
for ax, col in zip(axes.flatten(), high_corr_cols):
    sns.scatterplot(x=df_train[col], y=df_train["SalePrice"], alpha=0.3, ax=ax)
    ax.set_title(f"{col} vs SalePrice. 相関係数: {corr_matrix["SalePrice"][col]:.3f}")

# グラフのレイアウトを自動調整
plt.suptitle(
    f"SalePriceとの相関係数の絶対値が{threshold}以上の特徴量についての散布図\n"
)
plt.tight_layout()
plt.show()

# 外れ値が同じデータを指しているのかどうかをパパッと確認したいが…このままだと出来ない
# plotlyとかいうインタラクティブにグラフを描けるライブラリを使うと良いかも？
# とりあえず、もう一度スターター？見るかあ


# # 2. 前処理

# In[7]:


# 特徴量エンジニアリング
# 新しい特徴量（'TotalSF'：'TotalBsmtSF'、'1stFlrSF'、'2ndFlrSF'を合計したもの。）の作成

datasets = [df_train, df_test]
for i in range(len(datasets)):
    datasets[i]["TotalSF"] = (
        datasets[i]["TotalBsmtSF"] + datasets[i]["1stFlrSF"] + datasets[i]["2ndFlrSF"]
    )


# In[8]:


# lightGBMに突っ込むためには数値型(またはbool型)である必要があるので、object型のデータをlabel encodingで処理する
# https://qiita.com/Hyperion13fleet/items/afa49a84bd5db65ffc31　こっちのほうが便利？

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from lightgbm import LGBMRegressor, plot_tree
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape

# object型のデータが入っている列を抽出
object_columns = df_train.select_dtypes(include="object").columns
df_train_pre_encoding = df_train.copy()
df_test_pre_encoding = df_test.copy()

# ラベルエンコーディング
oe = OrdinalEncoder()
df_train[object_columns] = oe.fit_transform(df_train[object_columns])
df_test[object_columns] = oe.fit_transform(df_test[object_columns])

print("df_train_pre_encoding")
display(df_train_pre_encoding.head(3))
print("df_train")
display(df_train.head(3))


# In[9]:


# # ラベルエンコーディング後に改めて相関係数行列を表示してみる
# corr_matrix = df_train.corr(numeric_only=True)

# plt.figure(figsize=(24, 20))
# sns.heatmap(abs(corr_matrix), annot=True, fmt=".1f", annot_kws={"fontsize": 6})

# カテゴリ変数を含めて相関をみたいのなら、カテゴリ変数の順位関係を考慮したラベル付けをしておかねばなるまい
# しかし現状はそうはなっていない…
# plt.suptitle("訓練データの相関係数(絶対値)行列_ラベルエンコーディング後")
# plt.show()


# In[10]:


X = df_train.drop(["SalePrice"], axis=1)
y = df_train["SalePrice"]

# クロスバリデーション
kf = KFold(n_splits=4, shuffle=True, random_state=42)

scores = []
# params = {}
params = {"max_depth": 19, "learning_rate": 0.1}
# パラメータチューニングにはoptunaというのを使うと良いらしい
# https://qiita.com/tetsuro731/items/a19a85fd296d4b87c367
# https://qiita.com/tetsuro731/items/76434194bab336a97172

for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X)):
    print(f"分割 {fold_idx + 1} / {kf.n_splits}")

    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = LGBMRegressor(**params)
    # GBDTのパラメータについて。https://knknkn.hatenablog.com/entry/2021/06/29/125226
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_va)

    score = rmse(np.log10(y_pred), np.log10(y_va))
    print(f"スコア(rmse(np.log10(y_pred), np.log10(y_va)): {score}")
    mape_ = mape(y_pred, y_va) * 100
    print(f"MAPE (平均絶対誤差率): {mape_:.2f}%")
    rmspe = np.sqrt(np.mean(np.square((y_va - y_pred) / y_va))) * 100
    print(f"RMSPE (平均平方二乗誤差率): {rmspe:.2f}%")
    print("\n")

    scores.append(score)

print(f"{fold_idx + 1}個のモデルのスコアの平均値: {np.mean(scores)}.")

# メモ：[LightGBM] [Warning] No further splits with positive gain, best gain: -infについて
# これは「決定木の作成中、これ以上分岐を作っても予測誤差が下がらなかったのでこれ以上分岐をさせなかった」ことを意味するらしい


# MAPE（平均絶対誤差率）とRMSPE（平均平方二乗誤差率）は、どちらもモデルの予測精度をパーセンテージで表現する指標ですが、それぞれの特徴と捉え方に違いがあります。
# 
# ### 特徴の違い
# 
# 1. **MAPE（Mean Absolute Percentage Error）**:
#    - MAPEは各データポイントの絶対誤差の割合を平均したものです。
#    - 計算式：
#      \[
#      \text{MAPE} = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100
#      \]
#    - **直感的なイメージ**: MAPEは、予測値が実際の値に対してどれだけ外れているかを「平均的に」示します。外れ値の影響を受けにくいため、安定した誤差の評価に適しています。
#    - **メリット**: 解釈が簡単で、「平均して予測が実際の値から○○%ずれている」と直感的に理解しやすいです。
#    - **デメリット**: 実際の値がゼロに近い場合、誤差が無限大になるため、ゼロやゼロに近い値が含まれているデータには不向きです。
# 
# 2. **RMSPE（Root Mean Square Percentage Error）**:
#    - RMSPEは各データポイントの誤差率を平方して平均し、その平方根を取ったものです。
#    - 計算式：
#      \[
#      \text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2} \times 100
#      \]
#    - **直感的なイメージ**: RMSPEは、予測誤差が大きいデータポイントに対して、より強いペナルティを課します。つまり、大きな誤差に対して敏感であり、外れ値の影響を大きく受けます。
#    - **メリット**: 大きな誤差をより重視するため、予測精度を厳しく評価できます。外れ値や大きな誤差が特に重要な場合に適しています。
#    - **デメリット**: 外れ値の影響を強く受けるため、データにノイズが多い場合は過度に悪い評価が出ることがあります。
# 
# ### 直感的なイメージの違い
# 
# - **MAPE** は、「平均的なずれ」を強調します。どのデータポイントにおいても均等に誤差を見ているため、大きな誤差よりも「全体的な傾向」に重きを置きます。外れ値が少なく、すべてのデータが均等に重要な場合に有用です。
#   
# - **RMSPE** は、「大きなずれ」を強調します。誤差を平方しているため、外れ値（極端にずれた値）の影響が大きくなります。これは、誤差が非常に大きい場合にはその影響を強調したい場合に有効です。
# 
# ### 適切な使い分けの例
# 
# - **MAPE** は、予測が多少ずれても問題ない場合、または均等に重要なデータセットを扱うときに適しています。例えば、売上予測のように、大きな外れ値があっても全体の傾向を把握したい場合に使うことが多いです。
# 
# - **RMSPE** は、外れ値を重視したい場合や、極端な誤差が許容できない状況で使用するのが良いです。例えば、医療データや金融データのように、大きな誤差が許されないケースに適しています。
# 
# このように、MAPEとRMSPEはそれぞれの用途やデータの特性に応じて使い分けるべき指標です。どちらを使用するかは、あなたのデータや分析目的によって決めると良いでしょう。

# In[11]:


# 学習結果の図示(ここで表示しているのはクロスバリデーションの最後の分割時のモデルについて)
tree_idx = 0
print(f"{tree_idx + 1}番目の木の様子は以下の通り")


plot_tree(model, tree_index=tree_idx, figsize=(20, 10))

# 特徴量重要度
df_feature_importances = pd.DataFrame(
    {"feature_name": model.feature_name_, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)

plt.figure(figsize=(16, 8))
sns.barplot(data=df_feature_importances, x="feature_name", y="importance")
plt.xticks(rotation=90)
plt.show()


# In[12]:


# 一度このまま提出用のデータを出力
model = LGBMRegressor(max_depth=-1)
model.fit(X, y)
sub_pred = model.predict(df_test)
submission = pd.DataFrame({"Id": df_test_Id, "SalePrice": sub_pred})
submission.to_csv(r"train_test_submission\submission.csv", index=False)


# In[ ]:




