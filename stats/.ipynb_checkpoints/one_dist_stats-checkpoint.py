from typing import Optional, List, Union
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# calculate skewness
def calc_skew(data: Union[np.ndarray, pd.Series]) -> (float, str):
    """
    データの歪度を計算する．

    Parameters
    ----------
    data : pd.Series, np.ndarray etc.
        歪度を計算したいデータ
    """

    if data.dtype not in [int, float]:
        raise Exception("Data variable type must be numeric type as int, float etc.")

    if type(data) == pd.Series:
        data = data.values
    elif type(data) not in [np.ndarray, pd.Series]:
        raise Exception("Data type must be np.ndarray or pd.Series.")

    size = len(data)
    std = data.std()
    residual = data - data.mean()
    value = (size) / ((size-1)*(size-2)) * ((residual / std) ** 3).sum()

    if value > 0:
        skew = "left"
    elif value < 0:
        skew = "right"
    else:
        skew = "center"

    return value, skew


# calculate kurtosis
def calc_kurto(data: Union[np.ndarray, pd.Series]) -> float:
    """
    データの尖度を計算する．

    Parameters
    ----------
    data : pd.Series, np.ndarray etc.
        尖度を計算したいデータ
    """

    if data.dtype not in [int, float]:
        raise Exception("Data variable type must be numeric type as int, float etc.")

    if type(data) == pd.Series:
        data = data.values
    elif type(data) not in [np.ndarray, pd.Series]:
        raise Exception("Data type must be np.ndarray or pd.Series.")

    size = len(data)
    std = data.std()
    residual = data - data.mean()

    first = (size * (size + 1)) / ((size - 1) * (size - 2) * (size - 3))
    second = ((residual / std) ** 4).sum()
    third = (3 * (size - 1) ** 2) / ((size - 2) * (size - 3))

    return first * second - third


def all_stats(data: Union[np.ndarray, pd.Series],
              is_return: bool = False) -> Optional[List]:
    """
    データの統計量（平均，標準偏差，歪度，尖度）を計算する．

    Parameters
    ----------
    data : pd.Series, np.ndarray etc.
        統計量を計算したいデータ．
    is_return : bool
        計算した統計量を返すかどうか決めるフラグ(default=False)
    """

    if data.dtype not in [int, float]:
        raise Exception("Data variable type must be numeric type as int, float etc.")

    if type(data) == pd.Series:
        data = data.values
    elif type(data) not in [np.ndarray, pd.Series]:
        raise Exception("Data type must be np.ndarray or pd.Series.")

    mean = data.mean()
    std = data.std()
    skew_value, skewness = calc_skew(data)
    kurtosis = calc_kurto(data)

    print(f"data mean: {mean}")
    print(f"data std: {std}")
    print(f"data skewness: {skew_value}, {skewness}")
    print(f"data kurtosis: {kurtosis}")

    if is_return:
        return (mean, std, skew_value, skewness, kurtosis)


def dist_categorical(data: Union[pd.DataFrame, pd.Series],
                     feature_name: str) -> None:
    """
    カテゴリ変数のグラフの可視化（円グラフ，棒グラフ）

    Parameters
    ----------
    data : pd.DataFrame, pd.Series
        可視化したいデータ．
    feature_name : str
        可視化したいデータの名前．
    """

    if type(data) not in [pd.DataFrame, pd.Series]:
        raise Exception("Data type must be pd.DataFrame or pd.Series")

    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    data[feature_name].value_counts().plot.pie(autopct='%1.1f%%',
                                               ax=ax[0], shadow=True)
    ax[0].set_title(feature_name)
    sns.countplot(feature_name, data=data, ax=ax[1])
    ax[1].set_title(feature_name)
    plt.show()


def sturges_rule(n: int) -> int:
    """
    ヒストグラムを作成する際のビンの数をスタージェスの公式を用いて計算する．

    Parameters
    ----------
    n : int
        データ数．

    Returns
    -------
    value : int
        スタージェスの公式に基づいた最適なビンの数．
    """

    return math.ceil(1 + np.log2(n))


def dist_numerical(data: Union[pd.DataFrame, pd.Series],
                   feature_name: str) -> None:
    """
    数値変数のグラフの可視化（ヒストグラム，箱ひげ図）

    Parameters
    ----------
    data : pd.DataFrame, pd.Series
        可視化したいデータ．
    feature_name : str
        可視化したいデータの名前．
    """

    if type(data) not in [pd.DataFrame, pd.Series]:
        raise Exception("Data type must be pd.DataFrame or pd.Series")

    bins = sturges_rule(len(data))
    f, ax = plt.subplots(1, 2, figsize=(18, 8))

    if type(data) == pd.DataFrame:
        ax[0].hist(data[feature_name], bins=bins)
        ax[1].boxplot(data[feature_name], labels=[feature_name])
    elif type(data) == pd.Series:
        ax[0].hist(data, bins=bins)
        ax[1].boxplot(data, labels=[feature_name])

    ax[0].set_title(feature_name + "histogram")
    ax[1].set_title(feature_name + "box plot")
    plt.show()