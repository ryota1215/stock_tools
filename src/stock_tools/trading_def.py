# ライブラリー
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
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

    coef = np.polyfit(x, y, 1)[0]
    pred = np.polyval(np.polyfit(x, y, 1), x)
    val_r2 = r2_score(y, pred) * is_not_var0
    score = val_r2 * coef
    if return_arr:
        return score
    else:
        return pd.DataFrame([[coef, val_r2, score]], columns=["coef", "r2", "score"])


def trand_score(val, return_arr=False):
    try:
        val = val.values
    except AttributeError:
        pass
    is_not_var0 = int(np.var(val) != 0)

    x = np.arange(len(val)) + 1
    y = val

    coef = np.polyfit(x, y, 1)[0]
    pred = np.polyval(np.polyfit(x, y, 1), x)
    val_r2 = r2_score(y, pred) * is_not_var0
    score = val_r2 * coef
    if return_arr:
        return score
    else:
        return pd.DataFrame([[coef, val_r2, score]], columns=["coef", "r2", "score"])


def make_sector_index(dfs_code, dict_sector_codes, lst_sector):
    """
    jpxの株価データからセクターインデックスを作成する
    param : dfs_code 各銘柄の入ったdict
    param : dict_sector_codes セクター毎の銘柄コード取得
    param : lst_sector セクター名の入ったのリスト
    return : セクターインデックスのoc,cc,gapが入ったdf
    """
    df_sector_indexs = pd.DataFrame([])
    max_len = len(dfs_code["72030"])
    for target_sector in lst_sector:
        # その他はetf
        if target_sector == "(その他)":
            continue
        # セクター内の各銘柄の時価総額を取得
        dict_market_cap = {
            code: dfs_code[code].set_index("Date")["market_cap"].rename(code)
            for code in dict_sector_codes[target_sector]
        }
        df_market_cap = pd.concat(dict_market_cap, axis=1)
        df_market_cap = df_market_cap.fillna(0.0000000000000000000000000000000000000000000000001)
        # セクター内の各銘柄の騰落率を取得
        dict_oc_change = {
            code: dfs_code[code].set_index("Date")["AdjustmentClose"].rename(code)
            / dfs_code[code]["AdjustmentOpen"].values - 1
            for code in dict_sector_codes[target_sector]
        }
        dict_cc_change = {
            code: dfs_code[code].set_index("Date")["AdjustmentClose"].rename(code).pct_change()
            for code in dict_sector_codes[target_sector]
        }
        dict_gap = {
            code: dfs_code[code].set_index("Date")["AdjustmentOpen"].rename(code) /
            dfs_code[code]["AdjustmentClose"].shift().values - 1
            for code in dict_sector_codes[target_sector]
        }

        df_oc_change = pd.concat(dict_oc_change, axis=1)
        df_cc_change = pd.concat(dict_cc_change, axis=1)
        df_gap = pd.concat(dict_gap, axis=1)
        df_oc_change = df_oc_change.fillna(0)
        df_cc_change = df_cc_change.fillna(0)
        df_gap = df_gap.fillna(0)
        # 時価総額加重平均算出
        df_sector_oc = pd.DataFrame(
            np.average(df_oc_change, axis=1, weights=df_market_cap),
            index=pd.to_datetime(df_oc_change.index),
            columns=["sector_oc_change"],
        )
        df_sector_cc = pd.DataFrame(
            np.average(df_cc_change, axis=1, weights=df_market_cap),
            index=pd.to_datetime(df_oc_change.index),
            columns=["sector_cc_change"],
        )
        df_sector_gap = pd.DataFrame(
            np.average(df_gap, axis=1, weights=df_market_cap),
            index=pd.to_datetime(df_oc_change.index),
            columns=["sector_gap"],
        )
        lst_term_oc = []
        for term in [3, 5, 10, 20, 60, 120]:
            lst_term_oc.append(pd.DataFrame(
                np.average(
                    pd.DataFrame(
                        [
                            calc_term_oc(code, dfs_code, term, max_len)
                            for code in dict_sector_codes[target_sector]
                        ]
                    ).T,
                    axis=1,
                    weights=df_market_cap,
                ),
                index=pd.to_datetime(df_oc_change.index),
                columns=[f"sector_term{term}_oc"],
            ))
        df_terms_oc = pd.concat(lst_term_oc, axis=1)

        df_sector_index = pd.merge(df_sector_oc, df_sector_cc, left_index=True, right_index=True)
        df_sector_index = pd.merge(df_sector_index, df_sector_gap,
                                   left_index=True, right_index=True)
        df_sector_index = pd.merge(df_sector_index, df_terms_oc, left_index=True, right_index=True)
        df_sector_index.columns = [c.replace("sector", target_sector)
                                   for c in df_sector_index.columns]
        df_sector_indexs = pd.concat([df_sector_indexs, df_sector_index], axis=1)
    return df_sector_indexs


def calc_term_oc(code, dfs_code, term, max_len):
    try:
        term_close = sliding_window_view(dfs_code[code]["AdjustmentClose"].values, term)[:, -1]
    except ValueError:
        return np.arange(max_len) * 0
    term_open = sliding_window_view(dfs_code[code]["AdjustmentOpen"].values, term)[:, 0]
    val = term_close / term_open - 1
    if len(val) != max_len:
        return np.hstack((np.arange(max_len - len(val)) * 0, val))
    else:
        return val


def preprocess_jpxdata(df):
    """
    jpx前処理
    param : df jpxのDataflame
    return : df
    """
    df = df.copy()
    if len(df) == 0:
        return df
    df["Date"] = pd.to_datetime(df["Date"])
    df["oc_change"] = df["AdjustmentClose"] / df["AdjustmentOpen"] - 1
    df["gap"] = df["AdjustmentOpen"] / df["AdjustmentClose"].shift().values - 1
    df["cc_change"] = df["AdjustmentClose"].pct_change()
    df = add_market_cap(df)
    return df


def preprocess_kabudata(df):
    """
    kabu+前処理
    param : df kabu+のDataflame
    return : df
    """
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df


def add_market_cap(df):
    """
    jpxデータの時価総額修正
    param : df jpxのDataflame
    return : df
    """
    df_code = df.set_index("Date")
    df_adjustmentfactor = df_code["AdjustmentFactor"][df_code["AdjustmentFactor"] != 1]
    df_shares = df_code["NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock"]
    df_shares_pct = df_shares.pct_change().dropna()
    df_shares_pct_select = df_shares_pct[(df_shares_pct >= 0.03)
                                         | (df_shares_pct <= -0.03)].dropna()
    for date, val in df_adjustmentfactor.iteritems():
        try:
            target = df_shares_pct_select.loc[
                date - datetime.timedelta(days=120):date + datetime.timedelta(days=120)
            ]
            target_date = target.index[np.argmin(abs(target - val))]
        except Exception as e:
            print(e)
            print(df_code["Code"].unique(), date, val)

            adjust = 1 / val - 1
            df_shares_pct.loc[date] = adjust
            continue
        shares = df_shares_pct_select.loc[target_date]
        assert shares != 0, f"shares Error {date} {shares}"
        adjust = 1 / val - 1
        df_shares_pct.loc[date] = adjust
        df_shares_pct.loc[target_date] = (1 / round(1 + df_shares_pct.loc[date], 3)) - 1

    df_code["StockShares_fixed"] = df_shares * np.cumprod(df_shares_pct + 1)
    df_code["market_cap"] = df_code["StockShares_fixed"] * df_code["Close"]
    df_code = df_code.reset_index()
    return df_code
