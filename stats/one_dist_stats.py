from typing import Optional, List, Union
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


# calculate skewness
def calc_skew(data: Union[np.ndarray, pd.Series]) -> (float, str):
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
    return math.ceil(1 + np.log2(n))


def dist_numerical(data: Union[pd.DataFrame, pd.Series],
                   feature_name: str) -> None:
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