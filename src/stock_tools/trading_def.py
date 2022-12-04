# ライブラリー
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import calendar
import locale
import warnings
import datetime
import jpholiday
from logging import getLogger

logger = getLogger(__name__)

warnings.simplefilter("ignore")
# PL評価


def eval_pl(pl: float, Print: bool = True):
    sr = np.mean(pl) / np.std(pl) * np.sqrt(252)
    sr = round(sr, 2)
    ev = np.mean(pl)
    ev = round(ev * 10000, 2)
    wp = np.mean(pl > 0)
    wp = round(wp * 100, 2)
    sr252 = np.mean(pl[-252:]) / np.std(pl[-252:]) * np.sqrt(252)
    sr252 = round(sr252, 2)
    ev252 = np.mean(pl[-252:])
    ev252 = round(ev252 * 10000, 2)
    wp252 = np.mean(pl[-252:] > 0)
    wp252 = round(wp252 * 100, 2)
    if Print:
        return print(
            f"SR : {sr} , EV : {ev} wp : {wp} % \nSR252 : {sr252} , EV252 : {ev252} wp252 : {wp252} %"
        )
    else:
        return sr, ev, wp, sr252, ev252, wp252


# 対数騰落率(二日以上保有して累積騰落率出す場合)
def logPL(CC_Change, price):
    # priceは率だけ測るりたい場合-1
    if price == -1:
        logPL = np.exp(np.log(CC_Change + 1).cumsum()) * 1 - 1
    else:
        logPL = np.exp(np.log(CC_Change + 1).cumsum()) * price
    return logPL


# 日にちファクター
def dateFacter(cdf):
    # 月初判定
    monthFirst0 = np.unique(
        pd.DatetimeIndex(
            np.unique(
                pd.DatetimeIndex(cdf.index).year.astype(str)
                + "/"
                + pd.DatetimeIndex(cdf.index).month.astype(str)
            )
        ).date
    )
    monthFirst = [cdf.index[cdf.index >= date][0] for date in monthFirst0[1:]]
    cdf["BeingOfTheMonth"] = 0
    cdf.loc[monthFirst, "BeingOfTheMonth"] = 1
    # 月列を作成
    cdf["Month"] = pd.DatetimeIndex(cdf.index).month
    # 月末判定
    EndOfMonthArr = []
    for n in range(len(monthFirst0)):
        EndOfMonth = calendar.monthrange(monthFirst0[n].year, monthFirst0[n].month)[1]
        EndOfMonth = monthFirst0[n].replace(day=EndOfMonth)
        EndOfMonthArr.append(EndOfMonth)
    EndOfMonthList = [cdf.index[cdf.index <= date][-1] for date in EndOfMonthArr]
    # 最終月は排除
    EndOfMonthList = EndOfMonthList[:-1]
    cdf["EndOfMonthList"] = 0
    cdf.loc[EndOfMonthList, "EndOfMonthList"] = 1
    # 曜日判定
    locale.setlocale(locale.LC_TIME, "ja_JP.UTF-8")
    # 0→月 4→金
    weekdayList = [date.weekday() for date in cdf.index]
    cdf["WeekDay"] = weekdayList


def DateMach(cdf, idx_df):
    # 日付合わせ
    if len(cdf) - len(idx_df) > 0:
        cdf = cdf.reindex(idx_df.index)
    else:
        idx_df = idx_df.reindex(cdf.index)
    return cdf, idx_df


def ic_plot(x, returns, normalize=True):
    """
    :param np.ndarray x: 指標
    :param np.ndarray returns: リターン
    :param bool normalize: x をスケーリングするかどうか
    """

    def _steps(x, y):
        int_x = np.round(x)
        ret_x = np.unique(int_x)
        ret_y = []
        for xa in ret_x:
            ret_y.append(np.average(y[int_x == xa]))
        return ret_x, np.array(ret_y)

    assert len(x) == len(returns)
    # 正規化
    x = (x - x.mean()) / x.std() if normalize else x
    x = x.astype(float)
    returns = (returns - returns.mean()) / returns.std() if normalize else returns
    returns = returns.astype(float)
    # 散布図
    plt.plot(x, returns, "x")
    # 回帰直線
    reg = np.polyfit(x, returns, 1)
    plt.plot(x, np.poly1d(reg)(x), color="c", linewidth=2)
    # 区間平均値
    plt.plot(*_steps(x, returns), drawstyle="steps-mid", color="r", linewidth=2)

    # 相関係数（情報係数）
    ic = np.corrcoef(x, returns)[0, 1]
    plt.title(f"IC={ic:.3f}, y={reg[0]:.3f}x{reg[1]:+.3f}")
    plt.grid()
    plt.show()


def coef_intercept(x, y, scaler: str = None):
    """
    xとyの傾きと切片を算出
    xをNoneにすると連続整数がxに入る
    param : scaler str (None,ss,mm,mmss,ssmm) スケール方法を選択
    return : dict["minmax","standardscaler","coef","intercept"]
    """
    # データ前処理
    if x is None:
        x = np.arange(len(y)) + 1
        x = x.reshape(-1, 1)
    else:
        x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # 傾きと切片算出
    dict_return = {
        "minmax": None,
        "standardscaler": None,
        "coef": None,
        "intercept": None,
    }
    ss = StandardScaler()
    mm = MinMaxScaler()
    lr = LinearRegression()
    if scaler is None:
        lr.fit(x, y)
    elif scaler == "ss":
        x, y = ss.fit_transform(x), ss.fit_transform(y)
        lr.fit(x, y)
    elif scaler == "mm":
        x, y = mm.fit_transform(x), mm.fit_transform(y)
        lr.fit(x, y)
    elif scaler == "mmss":
        x, y = mm.fit_transform(x), mm.fit_transform(y)
        x, y = ss.fit_transform(x), ss.fit_transform(y)
        lr.fit(x, y)
    elif scaler == "ssmm":
        x, y = ss.fit_transform(x), ss.fit_transform(y)
        x, y = mm.fit_transform(x), mm.fit_transform(y)
        lr.fit(x, y)

    # dictに格納
    for val, name in zip((ss, mm, lr), ("standardscaler", "minmax", "lr")):
        if "lr" == name:
            dict_return["coef"] = lr.coef_[0][0]
            dict_return["intercept"] = lr.intercept_[0]
        else:
            dict_return[name] = val
    return dict_return


def is_sq(date):
    year, month = date.year, date.month
    sq_date = get_day_of_nth_dow(year, month, 2, 4)
    sq_date = datetime.date(year, month, sq_date)
    while True:
        if jpholiday.is_holiday(sq_date) or (sq_date.weekday() > 5):
            sq_date = sq_date - datetime.timedelta(days=1)
        else:
            break
    return sq_date == date


def get_day_of_nth_dow(year, month, nth, dow):
    '''dow: Monday(0) - Sunday(6)'''
    if nth < 1 or dow < 0 or dow > 6:
        return None

    first_dow, n = calendar.monthrange(year, month)
    day = 7 * (nth - 1) + (dow - first_dow) % 7 + 1

    return day if day <= n else None


def trand_score(val, return_arr=False):
    try:
        val = val.values
    except AttributeError:
        pass
    is_not_var0 = int(np.var(val) != 0)

    x = np.arange(len(val)) + 1
    y = val

    lr = LinearRegression()
    lr.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    coef = lr.coef_[0][0] * 10000
    pred = lr.predict(x.reshape(-1, 1)).ravel()
    val_r2 = r2_score(y, pred) * is_not_var0
    score = val_r2 * coef
    if return_arr:
        return score
    else:
        return pd.DataFrame([[coef, val_r2, score]], columns=["coef", "r2", "score"])
