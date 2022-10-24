import numpy as np
import pandas as pd
import ta
import talib

# すべての警告を削除
import warnings

warnings.simplefilter("ignore")


class add_technical:
    def __init__(self, df, is_jpx: bool = False):
        """
        dfにテクニカル指標を追加する
        :param df
        :param is_jpx dfがjpxのdataか否か
        :return ファクターを追加したdf
        """
        df_c = df.copy()

        if is_jpx:
            open = df_c["AdjustmentOpen"]
            high = df_c["AdjustmentHigh"]
            low = df_c["AdjustmentLow"]
            close = df_c["AdjustmentClose"]
            volume = df_c["AdjustmentVolume"]
        else:
            # 修正株価
            bukatu_rate = np.cumprod(1 / df_c["adjustmentfactor"])
            bukatu_rate = bukatu_rate / bukatu_rate.iloc[-1]
            bukatu_rate = bukatu_rate.shift().fillna(method="bfill")
            df_c["fix_open"] = df_c["open"] * bukatu_rate
            df_c["fix_high"] = df_c["high"] * bukatu_rate
            df_c["fix_low"] = df_c["low"] * bukatu_rate
            df_c["fix_close"] = df_c["close"] * bukatu_rate
            df_c["fix_volume"] = df_c["volume"] / bukatu_rate
            open = df_c["fix_open"]
            high = df_c["fix_high"]
            low = df_c["fix_low"]
            close = df_c["fix_close"]

        # 直近の騰落率ファクター追加
        df_c = self.pricechange(df_c, open, close)

        # 各ファクター追加
        df_c = self.momentums(df, open, high, low, close, volume)
        df_c = self.trends(df, open, high, low, close, volume)
        df_c = self.volatilities(df, open, high, low, close, volume)
        df_c = self.volumes(df, open, high, low, close, volume)
        df_c = self.others(df, open, high, low, close, volume)

        return df_c

    def pricechange(self, df, open, close):
        """
        直近の騰落率を算出
        :param df 株価のdataframe
        :param serries_column ラグ特徴量を生成したいcolumnのSerries
        :param lag_span ラグ数のarray 初期 1~10
        :return ファクターを追加したdf
        """
        df_c = df.copy()
        df_c["oc_change"] = close / open - 1
        df_c["cc_change"] = close.pct_change()
        df_c["cc_change"].iloc[0] = 0
        df_c["gap"] = open / close.shift() - 1
        return df

    def lag_make(self, df, serries_column, lag_span=np.arange(1, 11)):
        """
        ラグ特徴量作成関数 lag_spanでラグ数設定
        :param df 株価のdataframe
        :param serries_column ラグ特徴量を生成したいcolumnのSerries
        :param lag_span ラグ数のarray 初期 1~10
        :return ファクターを追加したdf
        """
        for span in lag_span:
            df[f"{serries_column.name}_lag{span}"] = serries_column.shift(span)
        return df

    def momentums(self, df, open, high, low, close, volume):
        df_c = df.copy()
        old_columns = df_c.columns

        # RSIIndicator
        momentum_rsi_span = [5, 10, 25, 50, 75]
        for span in momentum_rsi_span:
            df_c[f"momentum_rsi{span}"] = ta.momentum.RSIIndicator(
                close, window=span, fillna=False
            ).rsi()
            self.lag_make(df_c, df_c[f"momentum_rsi{span}"])

        # StochRSIIndicator
        momentum_stoch_rsi_span = [5, 10, 14, 25, 50, 75]
        for span in momentum_stoch_rsi_span:
            df_c[f"momentum_stoch_rsi{span}"] = ta.momentum.StochRSIIndicator(
                close, window=span, smooth1=3, smooth2=3, fillna=False
            ).stochrsi()
            self.lag_make(df_c, df_c[f"momentum_stoch_rsi{span}"])
            df_c[f"momentum_stoch_rsi_k{span}"] = ta.momentum.StochRSIIndicator(
                close, window=span, smooth1=3, smooth2=3, fillna=False
            ).stochrsi_k()
            self.lag_make(df_c, df_c[f"momentum_stoch_rsi_k{span}"])
            df_c[f"momentum_stoch_rsi_d{span}"] = ta.momentum.StochRSIIndicator(
                close, window=span, smooth1=3, smooth2=3, fillna=False
            ).stochrsi_d()
            self.lag_make(df_c, df_c[f"momentum_stoch_rsi_d{span}"])

        # TSIIndicator
        """
        一般的に使われる期間[s=25, f=13]
        """
        momentum_tsi_span = [[13, 7], [25, 13], [40, 20]]
        for s, f in momentum_tsi_span:
            df_c[f"momentum_tsi{s}"] = ta.momentum.TSIIndicator(
                close, window_slow=s, window_fast=f, fillna=False
            ).tsi()
            self.lag_make(df_c, df_c[f"momentum_tsi{s}"])

        # UltimateOscillator
        """
        一般的に使われる期間は7日
        """
        momentum_uo_span = [5, 7, 10, 14, 25]
        for span in momentum_uo_span:
            df_c[f"momentum_uo{span}"] = ta.momentum.UltimateOscillator(
                high,
                low,
                close,
                window1=span,
                window2=span * 2,
                window3=span * 4,
                weight1=4.0,
                weight2=2.0,
                weight3=1.0,
                fillna=False,
            ).ultimate_oscillator()
            self.lag_make(df_c, df_c[f"momentum_uo{span}"])

        # StochasticOscillator
        momentum_stoch_span = [5, 10, 14, 25, 50, 75]
        for span in momentum_stoch_span:
            df_c[f"momentum_stoch{span}"] = ta.momentum.StochasticOscillator(
                high, low, close, window=span, smooth_window=3, fillna=False
            ).stoch()
            self.lag_make(df_c, df_c[f"momentum_stoch{span}"])
            momentum_stoch_signal = ta.momentum.StochasticOscillator(
                high, low, close, window=span, smooth_window=3, fillna=False
            ).stoch_signal()
            df_c[f"momentum_stoch_signal{span}"] = momentum_stoch_signal
            self.lag_make(df_c, df_c[f"momentum_stoch_signal{span}"])

        # WilliamsRIndicator
        momentum_wr_span = [5, 10, 14, 25, 50, 75]
        for span in momentum_wr_span:
            df_c[f"momentum_wr{span}"] = ta.momentum.WilliamsRIndicator(
                high, low, close, lbp=span, fillna=False
            ).williams_r()
            self.lag_make(df_c, df_c[f"momentum_wr{span}"])

        # awesome_oscillator
        """
        一般的に使われる期間[w1=5, w2=34]
        他はmacdのパラメータを参考にする。
        """
        trend_macd_span = [[5, 34], [12, 26], [19, 39]]
        for w1, w2 in trend_macd_span:
            df_c[f"momentum_ao{w1}"] = ta.momentum.awesome_oscillator(
                high, low, window1=w1, window2=w2, fillna=False
            )
            self.lag_make(df_c, df_c[f"momentum_ao{w1}"])

        # ROCIndicator
        """
        一般的に使われる期間は12日
        """
        momentum_roc_span = [5, 10, 12, 25, 50, 75]
        for span in momentum_roc_span:
            df_c[f"momentum_roc{span}"] = ta.momentum.ROCIndicator(
                close, window=span, fillna=False
            ).roc()
            self.lag_make(df_c, df_c[f"momentum_roc{span}"])

        # PercentagePriceOscillator
        """
        期間は一般的に使われるもののみ使用する。
        """
        df_c["momentum_ppo"] = ta.momentum.PercentagePriceOscillator(
            close, window_slow=26, window_fast=12, window_sign=9, fillna=False
        ).ppo()
        self.lag_make(df_c, df_c["momentum_ppo"])
        df_c["momentum_ppo_signal"] = ta.momentum.PercentagePriceOscillator(
            close, window_slow=26, window_fast=12, window_sign=9, fillna=False
        ).ppo_signal()
        self.lag_make(df_c, df_c["momentum_ppo_signal"])
        df_c["momentum_ppo_hist"] = ta.momentum.PercentagePriceOscillator(
            close, window_slow=26, window_fast=12, window_sign=9, fillna=False
        ).ppo_hist()
        self.lag_make(df_c, df_c["momentum_ppo_hist"])

        # PercentageVolumeOscillator
        """
        期間は一般的に使われるもののみ使用する。
        """
        df_c["momentum_pvo"] = ta.momentum.PercentageVolumeOscillator(
            volume, window_slow=26, window_fast=12, window_sign=9, fillna=False
        ).pvo()
        self.lag_make(df_c, df_c["momentum_pvo"])
        df_c["momentum_pvo_signal"] = ta.momentum.PercentageVolumeOscillator(
            volume, window_slow=26, window_fast=12, window_sign=9, fillna=False
        ).pvo_signal()
        self.lag_make(df_c, df_c["momentum_pvo_signal"])
        df_c["momentum_pvo_hist"] = ta.momentum.PercentageVolumeOscillator(
            volume, window_slow=26, window_fast=12, window_sign=9, fillna=False
        ).pvo_hist()
        self.lag_make(df_c, df_c["momentum_pvo_hist"])

        # kama
        """
        一般的に使われる期間は10日
        """
        momentum_kama_span = [5, 10, 25, 50, 75]
        for span in momentum_kama_span:
            df_c[f"momentum_kama{span}"] = ta.momentum.kama(
                close, window=span, pow1=2, pow2=30, fillna=False
            )
            self.lag_make(df_c, df_c[f"momentum_kama{span}"])
            df_c[f"momentum_kama_st{span}"] = df_c[f"momentum_kama{span}"] / close
            self.lag_make(df_c, df_c[f"momentum_kama_st{span}"])

        # sonar
        """
        一般的に使われる期間は10,25日
        window1はn日指数移動平均、window2は過去n日前。
        http://exceltechnical.web.fc2.com/sonar.html

        """
        momentum_sonar_span = [[10, 25], [20, 50], [50, 150]]
        for window1, window2 in momentum_sonar_span:
            ema = talib.EMA(close, timeperiod=window1)
            ema_shift = ema.shift(window2)
            ema_result = ema - ema_shift
            df_c[f"momentum_sonar{window1}"] = ema_result
            self.lag_make(df_c, df_c[f"momentum_sonar{window1}"])

        # AB Ratio
        """
        windowは値動きの合計期間
        https://search.sbisec.co.jp/v2/popwin/tools/hyper/RM_08.pdf
        """
        momentum_abratio_span = [5, 10, 25, 50, 75]
        for window in momentum_abratio_span:
            a_ratio1 = high - open
            a_ratio1 = a_ratio1.rolling(window).sum()
            a_ratio2 = open - low
            a_ratio2 = a_ratio2.rolling(window).sum()
            df_c[f"momentum_aratio{window}"] = a_ratio1 / a_ratio2 * 100
            self.lag_make(df_c, df_c[f"momentum_aratio{window}"])
            b_ratio1 = high - close.shift(1)
            b_ratio1 = b_ratio1.rolling(window).sum()
            b_ratio2 = close.shift(1) - low
            b_ratio2 = b_ratio2.rolling(window).sum()
            df_c[f"momentum_bratio{window}"] = b_ratio1 / b_ratio2 * 100
            self.lag_make(df_c, df_c[f"momentum_bratio{window}"])

        # RCI
        """
        windowはcloseの順位付けする期間
        https://search.sbisec.co.jp/v2/popwin/tools/hyper/RM_08.pdf
        """

        def _rci(x):
            num = np.array(range(window))
            num = np.sort(num)[::-1]
            zyuni = num - (np.argsort(np.argsort(x)[::-1]))
            zyuni = zyuni**2
            goukei = zyuni.sum()
            rci = (1 - (6 * goukei) / (window * (window**2 - 1))) * 100
            return rci

        momentum_rci_span = [5, 10, 25, 50, 75]
        for window in momentum_rci_span:
            close_roll = close.rolling(window, min_periods=window)
            df_c[f"momentum_rci{window}"] = close_roll.apply(_rci, True)
            self.lag_make(df_c, df_c[f"momentum_rci{window}"])

        # sigma
        """
        windowは標準偏差の期間
        https://search.sbisec.co.jp/v2/popwin/tools/hyper/RM_08.pdf
        """
        momentum_sigma_span = [5, 10, 25, 50, 75]
        for window in momentum_sigma_span:
            sigma_a = close - close.rolling(window).mean()
            sigma_b = close.rolling(window).std(ddof=0)
            sigma = sigma_a / sigma_b
            df_c[f"momentum_sigma{window}"] = sigma
            self.lag_make(df_c, df_c[f"momentum_sigma{window}"])

        # Adaptive Stochastic Oscillator
        """
        Tushar S. Chandeによって開発された可変型ストキャスティックス
        標準の期間は[5, 7, 28]
        window1 = 標準偏差の計算期間
        window2 = ASOの計算期間の最大値
        window3 = ASOの計算期間の最小値
        http://exceltechnical.web.fc2.com/aso.html
        """
        momentum_aso_span = [[5, 7, 28], [10, 14, 56]]
        for window1, window2, window3 in momentum_aso_span:
            s = close.rolling(window1).std(ddof=0)  # 母標準偏差
            s_min = s.rolling(window1).min()
            s_max = s.rolling(window1).max()
            s_stocha = (s - s_min) / (s_max - s_min)
            aso_window = window2 + (window3 - window2) * (1 - s_stocha)
            aso_window = np.trunc(aso_window)
            aso_window = aso_window.fillna(0)
            aso_window = aso_window.astype(int)
            aso_num = aso_window.values  # aso_windowをndarrayに変更する。
            zero_max = np.zeros((len(aso_num)))  # 期間中の最大値空リスト
            zero_min = np.zeros((len(aso_num)))  # 期間中の最小値空リスト
            for i in range(len(aso_num)):  # 期間中の最大値リスト作成ループ
                if (i < (window3 - 1)) | (aso_num[i] < 2):  # window3の値未満だったらnanにする。
                    zero_max[i] = np.nan
                    zero_min[i] = np.nan
                else:
                    x = close[i + 1 - aso_num[i] + 1 : i + 1]
                    # rolling日数分のclose値をスライスして取得。
                    zero_max[i] = np.nanmax(x)
                    zero_min[i] = np.nanmin(x)

            aso = (close - zero_min) / (zero_max - zero_min) * 100
            aso = aso.values
            aso = np.nan_to_num(aso)  # 欠損値を0に変更する。
            aso_num = np.zeros((len(aso)))  # 空リスト作成。
            for i in range(len(aso)):
                aso_num[i] = (aso[i] * 0.5) + (aso_num[i - 1] * 0.5)  # 指数移動平均を求める。
            aso_num = np.where(aso_num == 0, np.nan, aso_num)
            df_c[f"momentum_aso{window1}"] = aso_num
            self.lag_make(df_c, df_c[f"momentum_aso{window1}"])

        # Chande's Momentum Oscillator
        """
        window = 算定期間(9日か14日が一般的)
        http://exceltechnical.web.fc2.com/cm.html
        """
        momentum_cmo_span = [5, 9, 14, 25, 50, 75]

        for window in momentum_cmo_span:
            close_diff = close.diff(1)
            close_diff = close_diff.values
            close_up = pd.Series(np.where(close_diff <= 0, 0, close_diff))
            close_down = pd.Series(np.where(close_diff >= 0, 0, close_diff))
            close_up = close_up.rolling(window).sum()
            close_down = close_down.rolling(window).sum()
            df_c[f"momentum_cmo{window}"] = (
                (close_up - close_down.abs()) / (close_up + close_down.abs()) * 100
            )
            self.lag_make(df_c, df_c[f"momentum_cmo{window}"])

        # Morris RSI
        """
        window = 算定期間
        http://exceltechnical.web.fc2.com/mrsiv.html
        """
        momentum_morrisrsi_span = [5, 10, 25, 50, 75]
        for window in momentum_morrisrsi_span:
            diff = close.diff(1).values
            pos = np.zeros(len(diff))
            neg = np.zeros(len(diff))
            for i in range(len(diff)):
                if diff[i] > 0:
                    pos[i] = diff[i] * volume[i]
                else:
                    neg[i] = np.abs(diff[i] * volume[i])
            pos = pd.Series(pos).rolling(window).sum()
            neg = pd.Series(neg).rolling(window).sum()
            df_c[f"momentum_morrisrsi{window}"] = pos / (pos + neg) * 100
            self.lag_make(df_c, df_c[f"momentum_morrisrsi{window}"])

        # IMI(Intraday Momentum Index)
        """
        window = 期間合計算定期間(14が一般的)
        http://exceltechnical.web.fc2.com/imi.html
        """
        momentum_imi_span = [5, 10, 14, 25, 50, 75]
        for window in momentum_imi_span:
            oc_diff = close.values - open.values
            pos = np.where(oc_diff > 0, oc_diff, 0)
            diff_abs = np.abs(oc_diff)
            pos = pd.Series(pos).rolling(window).sum()
            diff_abs = pd.Series(diff_abs).rolling(window).sum()
            df_c[f"momentum_imi{window}"] = pos / diff_abs * 100
            self.lag_make(df_c, df_c[f"momentum_imi{window}"])

        # Laguerre RSI
        """
        gammaは0～1.0の間の任意の値(0.4が一般的？)(1.0の場合は必ず0になるため意味なし)
        gammaは、「laguerre」の感度を設定するための項目です。
        小さな値に設定すると価格への反応が俊敏になりますが、ノイズも多くなります。
        反対に大きな値に設定すると反応が緩やかになります。
        http://exceltechnical.web.fc2.com/laguersi.html
        """
        gamma_span = [0, 0.2, 0.4, 0.6, 0.8]
        for gamma in gamma_span:
            nakane = (high.values + low.values) / 2
            l0 = np.zeros(len(nakane))
            for i in range(len(l0)):
                if i == 0:
                    l0[0] = (1 - gamma) * nakane[0] + gamma * 0
                else:
                    l0[i] = (1 - gamma) * nakane[i] + gamma * l0[i - 1]
            l1 = np.zeros(len(nakane))
            for i in range(len(l1)):
                if i == 0:
                    l1[0] = -gamma * l0[i]
                else:
                    l1[i] = -gamma * l0[i] + l0[i - 1] + gamma * l1[i - 1]
            l2 = np.zeros(len(nakane))
            for i in range(len(l2)):
                if i == 0:
                    l2[0] = -gamma * l1[i]
                else:
                    l2[i] = -gamma * l1[i] + l1[i - 1] + gamma * l2[i - 1]
            l3 = np.zeros(len(nakane))
            for i in range(len(l3)):
                if i == 0:
                    l3[0] = -gamma * l2[i]
                else:
                    l3[i] = -gamma * l2[i] + l2[i - 1] + gamma * l3[i - 1]
            cu1 = np.zeros(len(nakane))
            for i in range(len(cu1)):
                if l0[i] > l1[i]:
                    cu1[i] = l0[i] - l1[i]
                else:
                    cu1[i] = 0
            cd1 = np.zeros(len(nakane))
            for i in range(len(cd1)):
                if l0[i] > l1[i]:
                    cd1[i] = 0
                else:
                    cd1[i] = l1[i] - l0[i]
            cu2 = np.zeros(len(nakane))
            for i in range(len(cu2)):
                if l1[i] > l2[i]:
                    cu2[i] = cu1[i] + l1[i] - l2[i]
                else:
                    cu2[i] = cu1[i]
            cd2 = np.zeros(len(nakane))
            for i in range(len(cd2)):
                if l1[i] > l2[i]:
                    cd2[i] = cd1[i]
                else:
                    cd2[i] = cd1[i] + l2[i] - l1[i]
            cu3 = np.zeros(len(nakane))
            for i in range(len(cu3)):
                if l2[i] > l3[i]:
                    cu3[i] = cu2[i] + l2[i] - l3[i]
                else:
                    cu3[i] = cu2[i]
            cd3 = np.zeros(len(nakane))
            for i in range(len(cd3)):
                if l2[i] > l3[i]:
                    cd3[i] = cd2[i]
                else:
                    cd3[i] = cd2[i] + l3[i] - l2[i]
            df_c[f"momentum_lagrsi{gamma}"] = cu3 / (cu3 + cd3) * 100
            self.lag_make(df_c, df_c[f"momentum_lagrsi{gamma}"])

        new_columns = df_c.columns
        self.columns_momentums = np.setdiff1d(new_columns, old_columns)

        return df_c

    def trends(self, df, open, high, low, close, volume):
        df_c = df.copy()
        old_columns = df_c.columns

        # MACD
        """
        window_slowとwindow_fastの値は以下参照。
        https://bitjournal.bitcastle.io/post-4888/
        window_sign=9の値は固定してループしている。
        """
        trend_macd_span = [[19, 6], [26, 12], [39, 19]]
        for s, f in trend_macd_span:
            df_c[f"trend_macd{s}_{f}"] = ta.trend.MACD(
                close, window_slow=s, window_fast=f, window_sign=9, fillna=False
            ).macd()
            self.lag_make(df_c, df_c[f"trend_macd{s}_{f}"])
            df_c[f"trend_macd_signal{s}_{f}"] = ta.trend.MACD(
                close, window_slow=s, window_fast=f, window_sign=9, fillna=False
            ).macd_signal()
            self.lag_make(df_c, df_c[f"trend_macd_signal{s}_{f}"])
            df_c[f"trend_macd_diff{s}_{f}"] = ta.trend.MACD(
                close, window_slow=s, window_fast=f, window_sign=9, fillna=False
            ).macd_diff()
            self.lag_make(df_c, df_c[f"trend_macd_diff{s}_{f}"])

        # VortexIndicator
        """
        span = 14 は資料にあったため追加
        """
        trend_vortex_ind_span = [5, 10, 14, 25, 50, 75]
        for span in trend_vortex_ind_span:
            df_c[f"trend_vortex_ind_pos{span}"] = ta.trend.VortexIndicator(
                high, low, close, window=span, fillna=False
            ).vortex_indicator_pos()
            self.lag_make(df_c, df_c[f"trend_vortex_ind_pos{span}"])
            df_c[f"trend_vortex_ind_neg{span}"] = ta.trend.VortexIndicator(
                high, low, close, window=span, fillna=False
            ).vortex_indicator_neg()
            self.lag_make(df_c, df_c[f"trend_vortex_ind_neg{span}"])
            df_c[f"trend_vortex_ind_diff{span}"] = ta.trend.VortexIndicator(
                high, low, close, window=span, fillna=False
            ).vortex_indicator_diff()
            self.lag_make(df_c, df_c[f"trend_vortex_ind_diff{span}"])

        # TRIXIndicator
        """
        span = 15 は資料にあったため追加
        """
        trend_trix_span = [5, 10, 15, 25, 50, 75]
        for span in trend_trix_span:
            trix = ta.trend.TRIXIndicator(close, window=span).trix()
            df_c[f"trend_trix{span}"] = trix
            self.lag_make(df_c, df_c[f"trend_trix{span}"])

        # MassIndex
        """
        window_fast, window_slowは9,25が一般的。より長期の値もいくつか追加。
        """
        trend_mass_index_span = [[9, 25], [20, 50], [50, 150]]
        for f, s in trend_mass_index_span:
            df_c[f"trend_mass_index{f}_{s}"] = ta.trend.MassIndex(
                high, low, window_fast=f, window_slow=s, fillna=False
            ).mass_index()
            self.lag_make(df_c, df_c[f"trend_mass_index{f}_{s}"])

        # DPOIndicator
        """
        span = 15 は資料にあったため追加
        """
        trend_dpo_span = [5, 10, 15, 25, 50, 75]
        for span in trend_dpo_span:
            dpo = ta.trend.DPOIndicator(close, window=span, fillna=0).dpo()
            df_c[f"trend_dpo{span}"] = dpo
            self.lag_make(df_c, df_c[f"trend_dpo{span}"])

        # KSTIndicator
        """
        window等の変数が多いため一般的な値だけ使用する。
        """

        def kstindicator(
            close: pd.Series,
            roc1: int = 10,
            roc2: int = 15,
            roc3: int = 20,
            roc4: int = 30,
            window1: int = 10,
            window2: int = 10,
            window3: int = 10,
            window4: int = 15,
            nsig: int = 9,
            fillna: bool = False,
        ):
            min_periods_n1 = 0 if fillna else window1
            min_periods_n2 = 0 if fillna else window2
            min_periods_n3 = 0 if fillna else window3
            min_periods_n4 = 0 if fillna else window4
            rocma1 = (
                ((close - close.shift(roc1)) / close.shift(roc1))
                .rolling(window1, min_periods=min_periods_n1)
                .mean()
            )
            rocma2 = (
                ((close - close.shift(roc2)) / close.shift(roc2))
                .rolling(window2, min_periods=min_periods_n2)
                .mean()
            )
            rocma3 = (
                ((close - close.shift(roc3)) / close.shift(roc3))
                .rolling(window3, min_periods=min_periods_n3)
                .mean()
            )
            rocma4 = (
                ((close - close.shift(roc4)) / close.shift(roc4))
                .rolling(window4, min_periods=min_periods_n4)
                .mean()
            )
            kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
            kst_sig = kst.rolling(nsig, min_periods=0).mean()
            df_c["trend_kst"] = kst
            self.lag_make(df_c, df_c["trend_kst"])
            df_c["trend_kst_sig"] = kst_sig
            self.lag_make(df_c, df_c["trend_kst_sig"])
            kst_diff = kst - kst_sig
            df_c["trend_kst_diff"] = kst_diff
            self.lag_make(df_c, df_c["trend_kst_diff"])
            return df_c

        df_c = kstindicator(close=close)

        # IchimokuIndicator
        """
        期間で一般的なものは[9, 26, 52],
        海外で人気なもの[7, 22 , 44]
        """
        trend_ichimoku_span = [[9, 26, 52], [7, 22, 44]]
        for w1, w2, w3 in trend_ichimoku_span:
            df_c[f"trend_ichimoku_conv{w1}"] = ta.trend.IchimokuIndicator(
                high, low, window1=w1, window2=w2, window3=w3, visual=False
            ).ichimoku_conversion_line()
            self.lag_make(df_c, df_c[f"trend_ichimoku_conv{w1}"])
            df_c[f"trend_ichimoku_base{w1}"] = ta.trend.IchimokuIndicator(
                high, low, window1=w1, window2=w2, window3=w3, visual=False
            ).ichimoku_base_line()
            self.lag_make(df_c, df_c[f"trend_ichimoku_base{w1}"])
            df_c[f"trend_ichimoku_a{w1}"] = ta.trend.IchimokuIndicator(
                high, low, window1=w1, window2=w2, window3=w3, visual=False
            ).ichimoku_a()
            self.lag_make(df_c, df_c[f"trend_ichimoku_a{w1}"])
            df_c[f"trend_ichimoku_b{w1}"] = ta.trend.IchimokuIndicator(
                high, low, window1=w1, window2=w2, window3=w3, visual=False
            ).ichimoku_b()
            self.lag_make(df_c, df_c[f"trend_ichimoku_b{w1}"])

            ic_a = df_c[f"trend_ichimoku_a{w1}"]
            ic_b = df_c[f"trend_ichimoku_b{w1}"]
            ic_a_null = df_c[f"trend_ichimoku_a{w1}"].isnull()
            ic_b_null = df_c[f"trend_ichimoku_b{w1}"].isnull()
            # ichimokuトレンド評価sig1,sig2
            trend_ichimoku_sig1 = close.copy()
            trend_ichimoku_sig2 = close.copy()
            c = close
            for i in range(0, len(c)):
                if c.iloc[i] < ic_a.iloc[i] and c.iloc[i] < ic_b.iloc[i]:
                    trend_ichimoku_sig1.iloc[i] = 0
                elif c.iloc[i] >= ic_a.iloc[i] and c.iloc[i] <= ic_b.iloc[i]:
                    trend_ichimoku_sig1.iloc[i] = 1
                elif c.iloc[i] <= ic_a.iloc[i] and c.iloc[i] >= ic_b.iloc[i]:
                    trend_ichimoku_sig1.iloc[i] = 1
                elif ic_a_null.iloc[i] or ic_b_null.iloc[i]:
                    trend_ichimoku_sig1.iloc[i] = np.nan
                else:
                    trend_ichimoku_sig1.iloc[i] = 2
            trend_ichimoku_sig2 = np.where(ic_a > ic_b, 1, 0)
            df_c[f"trend_ichimoku_sig_up{w1}"] = np.where(
                (trend_ichimoku_sig1 == 2) & (trend_ichimoku_sig2 == 1), 1, 0
            )
            self.lag_make(df_c, df_c[f"trend_ichimoku_sig_up{w1}"])
            df_c[f"trend_ichimoku_sig_down{w1}"] = np.where(
                (trend_ichimoku_sig1 == 0) & (trend_ichimoku_sig2 == 0), 1, 0
            )
            self.lag_make(df_c, df_c[f"trend_ichimoku_sig_down{w1}"])

        # STCIndicator
        """
        一般的に使われる期間[50, 23, 10]
        短期の期間[34, 13, 4]
        """
        trend_stc_span = [[50, 23, 10], [34, 13, 4]]
        for s, f, c in trend_stc_span:
            df_c[f"trend_stc{s}"] = ta.trend.STCIndicator(
                close,
                window_slow=s,
                window_fast=f,
                cycle=c,
                smooth1=3,
                smooth2=3,
                fillna=False,
            ).stc()
            self.lag_make(df_c, df_c[f"trend_stc{s}"])

        # ADXIndicator
        """
        span = 14 は資料にあったため追加
        """
        trend_adx_span = [5, 10, 14, 25, 50, 75]
        for span in trend_adx_span:
            df_c[f"trend_adx{span}"] = (
                ta.trend.ADXIndicator(high, low, close, window=span, fillna=False)
                .adx()
                .replace(0, np.nan)
            )
            self.lag_make(df_c, df_c[f"trend_adx{span}"])
            df_c[f"trend_adx_pos{span}"] = (
                ta.trend.ADXIndicator(high, low, close, window=span, fillna=False)
                .adx_pos()
                .replace(0, np.nan)
            )
            self.lag_make(df_c, df_c[f"trend_adx_pos{span}"])
            df_c[f"trend_adx_neg{span}"] = (
                ta.trend.ADXIndicator(high, low, close, window=span, fillna=False)
                .adx_neg()
                .replace(0, np.nan)
            )
            self.lag_make(df_c, df_c[f"trend_adx_neg{span}"])

        # CCIIndicator
        """
        span = 20 は資料にあったため追加
        """
        trend_cci_span = [5, 10, 20, 25, 50, 75]
        for span in trend_cci_span:
            df_c[f"trend_cci{span}"] = ta.trend.CCIIndicator(
                high, low, close, window=span, constant=0.015, fillna=False
            ).cci()
            self.lag_make(df_c, df_c[f"trend_cci{span}"])

        # AroonIndicator
        trend_aroon_span = [5, 10, 25, 50, 75]
        for span in trend_aroon_span:
            df_c[f"trend_aroon_up{span}"] = ta.trend.AroonIndicator(
                close, window=span, fillna=False
            ).aroon_up()
            self.lag_make(df_c, df_c[f"trend_aroon_up{span}"])
            df_c[f"trend_aroon_down{span}"] = ta.trend.AroonIndicator(
                close, window=span, fillna=False
            ).aroon_down()
            self.lag_make(df_c, df_c[f"trend_aroon_down{span}"])
            df_c[f"trend_aroon_ind{span}"] = ta.trend.AroonIndicator(
                close, window=span, fillna=False
            ).aroon_indicator()
            self.lag_make(df_c, df_c[f"trend_aroon_ind{span}"])

        # PSARIndicator
        df_c["trend_psar"] = ta.trend.PSARIndicator(
            high, low, close, step=0.02, max_step=0.2, fillna=False
        ).psar()
        self.lag_make(df_c, df_c["trend_psar"])
        df_c["trend_psar_up"] = ta.trend.PSARIndicator(
            high, low, close, step=0.02, max_step=0.2, fillna=False
        ).psar_up()
        self.lag_make(df_c, df_c["trend_psar_up"])
        df_c["trend_psar_down"] = ta.trend.PSARIndicator(
            high, low, close, step=0.02, max_step=0.2, fillna=False
        ).psar_down()
        self.lag_make(df_c, df_c["trend_psar_down"])
        df_c["trend_psar_up_indicator"] = ta.trend.PSARIndicator(
            high, low, close, step=0.02, max_step=0.2, fillna=False
        ).psar_up_indicator()
        self.lag_make(df_c, df_c["trend_psar_up_indicator"])
        df_c["trend_psar_down_indicator"] = ta.trend.PSARIndicator(
            high, low, close, step=0.02, max_step=0.2, fillna=False
        ).psar_down_indicator()
        self.lag_make(df_c, df_c["trend_psar_down_indicator"])

        # DMI(Directional Movement Index)
        """
        window = dmp,dmn,tr算定期間
        http://exceltechnical.web.fc2.com/dmi.html
        """
        trend_dmi_span = [5, 10, 14, 25, 50, 75]
        for window in trend_dmi_span:
            dmp = high - high.shift(1)
            dmp = dmp.values
            dmn = low.shift(1) - low
            dmn = dmn.values
            dmp_num = np.zeros((len(dmp)))
            dmn_num = np.zeros((len(dmn)))
            for i in range(len(dmp)):
                if ((dmp[i] >= 0) | (dmn[i] >= 0)) & (dmp[i] > dmn[i]):
                    dmp_num[i] = dmp[i]
                if ((dmp[i] >= 0) | (dmn[i] >= 0)) & (dmp[i] < dmn[i]):
                    dmn_num[i] = dmn[i]
            dmp_num = pd.Series(dmp_num)
            dmn_num = pd.Series(dmn_num)
            dmp_roll = dmp_num.rolling(window).sum()
            dmn_roll = dmn_num.rolling(window).sum()
            close_shift = close.shift(1)
            volatility_tr = ta.utils.IndicatorMixin()._true_range(
                high, low, close_shift
            )
            tr_roll = volatility_tr.rolling(window).sum()
            df_c[f"trend_dmi_dip{window}"] = dmp_roll / tr_roll * 100
            df_c[f"trend_dmi_din{window}"] = dmn_roll / tr_roll * 100
            sig = (
                df_c[f"trend_dmi_dip{window}"].values
                - df_c[f"trend_dmi_din{window}"].values
            )
            df_c[f"trend_dmi_sig{window}"] = np.where(
                sig >= 0, 1, np.where(sig < 0, 0, sig)
            )
            self.lag_make(df_c, df_c[f"trend_dmi_dip{window}"])
            self.lag_make(df_c, df_c[f"trend_dmi_din{window}"])
            self.lag_make(df_c, df_c[f"trend_dmi_sig{window}"])

        # DeMarker
        """
        window = 算定期間
        http://exceltechnical.web.fc2.com/demarker.html
        """
        trend_dem_span = [5, 10, 25, 50, 75]
        for window in trend_dem_span:
            high_diff = high.values - high.shift(1).values
            demax = np.where(high_diff > 0, high_diff, 0)
            low_diff = low.values - low.shift(1).values
            demin = np.where(low_diff < 0, np.abs(low_diff), 0)
            demax = ta.trend.sma_indicator(pd.Series(demax), window=window)
            demin = ta.trend.sma_indicator(pd.Series(demin), window=window)
            df_c[f"trend_dem{window}"] = 100 * demax / (demax + demin)
            self.lag_make(df_c, df_c[f"trend_dem{window}"])

        new_columns = df_c.columns
        self.columns_trand = np.setdiff1d(new_columns, old_columns)

        # TDsequential
        """
        tradingviewのglazによるTD Sequential参考に作成
        """
        cc_diff = close.values - close.shift(4).values
        up = np.where(cc_diff > 0, 1, 0)
        down = np.where(cc_diff < 0, -1, 0)
        up_count = np.zeros(len(down))
        down_count = np.zeros(len(down))
        for i in range(len(down)):
            if up[i] == 0:
                up_count[i] = 0
            else:
                up_count[i] = up_count[i - 1] + up[i]
            if down[i] == 0:
                down_count[i] = 0
            else:
                down_count[i] = down_count[i - 1] + down[i]
        updown_count = up_count + down_count
        df_c["trend_tdseq_updown"] = updown_count
        sell_signal = np.where(updown_count == 9, 1, 0)
        buy_signal = np.where(updown_count == -9, 1, 0)
        df_c["trend_tdseq_sellsig"] = sell_signal
        df_c["trend_tdseq_buysig"] = buy_signal
        self.lag_make(df_c, df_c["trend_tdseq_updown"])
        self.lag_make(df_c, df_c["trend_tdseq_sellsig"])
        self.lag_make(df_c, df_c["trend_tdseq_buysig"])
        return df_c

    def volatilities(self, df, open, high, low, close, volume):
        df_c = df.copy()
        old_columns = df_c.columns

        # volatility_tr
        close_shift = close.shift(1)
        df_c["volatility_tr"] = ta.utils.IndicatorMixin()._true_range(
            high, low, close_shift
        )

        # volatility_tr_st
        df_c["volatility_tr_st"] = df_c["volatility_tr"] / close
        self.lag_make(df_c, df_c["volatility_tr_st"])

        # BollingerBands
        bb_span = [5, 10, 25, 50, 75]
        bb_dev = 2
        bb_fillna = False
        for bb_window in bb_span:
            df_c[f"volatility_bbm{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_mavg()
            self.lag_make(df_c, df_c[f"volatility_bbm{bb_window}"])
            df_c[f"volatility_bbh{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_hband()
            self.lag_make(df_c, df_c[f"volatility_bbh{bb_window}"])
            df_c[f"volatility_bbl{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_lband()
            self.lag_make(df_c, df_c[f"volatility_bbl{bb_window}"])
            df_c[f"volatility_bbw{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_wband()
            self.lag_make(df_c, df_c[f"volatility_bbw{bb_window}"])
            df_c[f"volatility_bbp{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_pband()
            self.lag_make(df_c, df_c[f"volatility_bbp{bb_window}"])
            df_c[f"volatility_bbhi{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_hband_indicator()
            self.lag_make(df_c, df_c[f"volatility_bbhi{bb_window}"])
            df_c[f"volatility_bbli{bb_window}"] = ta.volatility.BollingerBands(
                close, window=bb_window, window_dev=bb_dev, fillna=bb_fillna
            ).bollinger_lband_indicator()
            self.lag_make(df_c, df_c[f"volatility_bbli{bb_window}"])

        # volatility_kcc
        volatility_kcc_ema_span = [20, 25, 50, 75]
        volatility_kcc_atr_span = [5, 10, 20]
        for ema_span in volatility_kcc_ema_span:
            for atr_span in volatility_kcc_atr_span:
                kcc = ta.volatility.KeltnerChannel(
                    high,
                    low,
                    close,
                    window=ema_span,
                    window_atr=atr_span,
                    fillna=False,
                    original_version=False,
                    multiplier=2,
                )
                df_c[
                    f"volatility_kcc_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_mband()
                self.lag_make(df_c, df_c[f"volatility_kcc_ema{ema_span}atr{atr_span}"])
                df_c[
                    f"volatility_kch_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_hband()
                self.lag_make(df_c, df_c[f"volatility_kch_ema{ema_span}atr{atr_span}"])
                df_c[
                    f"volatility_kcl_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_lband()
                self.lag_make(df_c, df_c[f"volatility_kcl_ema{ema_span}atr{atr_span}"])
                df_c[
                    f"volatility_kcw_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_wband()
                self.lag_make(df_c, df_c[f"volatility_kcw_ema{ema_span}atr{atr_span}"])
                df_c[
                    f"volatility_kcp_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_pband()
                self.lag_make(df_c, df_c[f"volatility_kcp_ema{ema_span}atr{atr_span}"])
                df_c[
                    f"volatility_kchi_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_hband_indicator()
                self.lag_make(df_c, df_c[f"volatility_kchi_ema{ema_span}atr{atr_span}"])
                df_c[
                    f"volatility_kcli_ema{ema_span}atr{atr_span}"
                ] = kcc.keltner_channel_lband_indicator()
                self.lag_make(df_c, df_c[f"volatility_kcli_ema{ema_span}atr{atr_span}"])
        # DonchianChannel
        volatility_dc_span = [5, 10, 25, 50, 75]
        for span in volatility_dc_span:
            df_c[f"volatility_dcl{span}"] = ta.volatility.DonchianChannel(
                high, low, close, window=span, offset=0, fillna=False
            ).donchian_channel_lband()
            self.lag_make(df_c, df_c[f"volatility_dcl{span}"])
            df_c[f"volatility_dch{span}"] = ta.volatility.DonchianChannel(
                high, low, close, window=span, offset=0, fillna=False
            ).donchian_channel_hband()
            self.lag_make(df_c, df_c[f"volatility_dch{span}"])
            df_c[f"volatility_dcm{span}"] = ta.volatility.DonchianChannel(
                high, low, close, window=span, offset=0, fillna=False
            ).donchian_channel_mband()
            self.lag_make(df_c, df_c[f"volatility_dcm{span}"])
            df_c[f"volatility_dcw{span}"] = ta.volatility.DonchianChannel(
                high, low, close, window=span, offset=0, fillna=False
            ).donchian_channel_wband()
            self.lag_make(df_c, df_c[f"volatility_dcw{span}"])
            df_c[f"volatility_dcp{span}"] = ta.volatility.DonchianChannel(
                high, low, close, window=span, offset=0, fillna=False
            ).donchian_channel_pband()
            self.lag_make(df_c, df_c[f"volatility_dcp{span}"])

        # volatility_atr
        volatility_atr_span = [5, 10, 14, 25, 50, 75]
        """
        span = 14 は資料にあったため追加
        volatility_atr_stのstはStandardization=st（標準化）の意味でcloseで除算。
        """
        for span in volatility_atr_span:
            df_c[f"volatility_atr{span}"] = ta.volatility.AverageTrueRange(
                high, low, close, window=span, fillna=False
            ).average_true_range()
            self.lag_make(df_c, df_c[f"volatility_atr{span}"])

            # volatility_atr_st
            st = df_c[f"volatility_atr{span}"] / close
            df_c[f"volatility_atr_st{span}"] = st
            self.lag_make(df_c, df_c[f"volatility_atr_st{span}"])

        # volatility_ui
        volatility_ui_span = [5, 10, 14, 25, 50, 75]
        """
        span = 14 は資料にあったため追加
        """
        for span in volatility_ui_span:
            df_c[f"volatility_ui{span}"] = ta.volatility.UlcerIndex(
                close, window=span, fillna=False
            ).ulcer_index()
            self.lag_make(df_c, df_c[f"volatility_ui{span}"])

        # Mass Index
        """
        window1 指数移動平均日数
        window2 合計日数
        http://exceltechnical.web.fc2.com/mi.html
        """
        volatility_mi_span = [[9, 25], [20, 50], [50, 150]]
        for window1, window2 in volatility_mi_span:
            value1 = high - low
            value1 = talib.EMA(value1, timeperiod=window1)
            value2 = talib.EMA(value1, timeperiod=window1)
            mi = value1 / value2
            df_c[f"volatility_mi{window1}"] = mi.rolling(window2).sum()
            self.lag_make(df_c, df_c[f"volatility_mi{window1}"])

        # Historical Volatility
        """
        株価の変動率を年率換算した指標。オプション取引の世界でよく使われる。
        年間の標準偏差の確認のためならこのwindowがdefault(window=20, window=250)
        http://exceltechnical.web.fc2.com/hv.html
        """
        window1 = 20
        window2 = 250
        cc_change = close.pct_change()
        cc_change_rollstd = cc_change.rolling(window1).std(ddof=0)
        hv = cc_change_rollstd * np.sqrt(window2) * 100
        df_c["volatility_hv"] = hv
        self.lag_make(df_c, df_c["volatility_hv"])

        new_columns = df_c.columns
        self.columns_trand = np.setdiff1d(new_columns, old_columns)

        return df_c

    def volumes(self, df, open, high, low, close, volume):
        df_c = df.copy()
        old_columns = df_c.columns

        # ta volume_adi
        df_c["volume_adi"] = ta.volume.AccDistIndexIndicator(
            high, low, close, volume, fillna=False
        ).acc_dist_index()
        self.lag_make(df_c, df_c["volume_adi"])
        df_c["volume_adi_mfv"] = df_c["volume_adi"].diff()
        self.lag_make(df_c, df_c["volume_adi_mfv"])

        # volume_obv
        obv = np.where(close < close.shift(1), -volume, volume)
        obv = np.where(close == close.shift(1), 0, obv)
        df_c["volume_obv"] = obv.cumsum()
        self.lag_make(df_c, df_c["volume_obv"])
        df_c["volume_obv_diff"] = df_c["volume_obv"].diff()
        self.lag_make(df_c, df_c["volume_obv_diff"])

        # volume_cmf
        """
        window = Money Flow Volumeの合計期間（20デフォルト）
        windowは資料では20で設定されていたためspanに追加した。
        https://school.stockcharts.com/doku.php?id=technical_indicators:chaikin_money_flow_cmf
        """
        volume_cmf_span = [5, 10, 20, 25, 50, 75]
        for span in volume_cmf_span:
            df_c[f"volume_cmf{span}"] = ta.volume.ChaikinMoneyFlowIndicator(
                high, low, close, volume, window=span
            ).chaikin_money_flow()
            self.lag_make(df_c, df_c[f"volume_cmf{span}"])

        # volume_fi
        """
        window = 指数移動平均の期間（13デフォルト）
        https://school.stockcharts.com/doku.php?id=technical_indicators:force_index
        """
        volume_fi_span = [5, 10, 13, 25, 50, 75]
        for span in volume_fi_span:
            df_c[f"volume_fi{span}"] = ta.volume.ForceIndexIndicator(
                close, volume, window=span
            ).force_index()
            self.lag_make(df_c, df_c[f"volume_fi{span}"])
            df_c[f"volume_fi_diff{span}"] = df_c[f"volume_fi{span}"].diff()
            self.lag_make(df_c, df_c[f"volume_fi_diff{span}"])

        # volume_em
        """
        windowは計算に使っていないため期間のループ必要なし
        """
        df_c["volume_em"] = ta.volume.EaseOfMovementIndicator(
            high, low, volume, window=14
        ).ease_of_movement()
        self.lag_make(df_c, df_c["volume_em"])

        # volume_sma_em
        """
        window = smaの期間
        """
        volume_sma_em_span = [5, 10, 14, 25, 50, 75]
        for span in volume_sma_em_span:
            df_c[f"volume_sma_em{span}"] = ta.volume.EaseOfMovementIndicator(
                high, low, volume, window=span
            ).sma_ease_of_movement()
            self.lag_make(df_c, df_c[f"volume_sma_em{span}"])

        # volume_vpt taは足し引きするものが間違えている。自作
        vpt = volume * (
            (close - close.shift(1, fill_value=0)) / close.shift(1, fill_value=0)
        )
        vpt[0] = 0
        df_c["volume_vpt"] = vpt.cumsum()
        self.lag_make(df_c, df_c["volume_vpt"])
        df_c["volume_vpt_diff"] = df_c["volume_vpt"].diff()
        self.lag_make(df_c, df_c["volume_vpt_diff"])

        # volume_vwap 分足用 日をまたいで計算してしまうので注意
        """
        日足の場合、翌日のvwapを求めてるためwindow=1のみ指定する。
        """
        df_c["volume_vwap"] = ta.volume.VolumeWeightedAveragePrice(
            high, low, close, volume, window=1
        ).volume_weighted_average_price()
        self.lag_make(df_c, df_c["volume_vwap"])

        # volume_mfi
        volume_mfi_span = [5, 10, 14, 25, 50, 75]
        for span in volume_mfi_span:
            df_c[f"volume_mfi{span}"] = ta.volume.MFIIndicator(
                high, low, close, volume, window=span, fillna=False
            ).money_flow_index()
            self.lag_make(df_c, df_c[f"volume_mfi{span}"])

        # volume_nvi 任意の値のみ加算。今回は値下がりvolume
        price_change = close.pct_change()
        vol_decrease = volume.shift(1) > volume
        nvi = pd.Series(data=np.nan, index=close.index, dtype="float64")
        nvi.iloc[0] = 1000
        for i in range(1, len(nvi)):
            if vol_decrease.iloc[i]:
                nvi.iloc[i] = nvi.iloc[i - 1] + (100 * price_change.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i - 1]
        df_c["volume_nvi"] = nvi
        self.lag_make(df_c, df_c["volume_nvi"])

        # ボリュームレシオ
        """
        https://www.rakuten-sec.co.jp/MarketSpeed/onLineHelp/msman2_5_2_7.html
        """
        volume_vr_span = [5, 10, 25, 50, 75]
        for span in volume_vr_span:
            df_c["close_up"] = pd.Series(np.where(close.diff() > 0, volume.values, 0))
            up = df_c["close_up"].rolling(window=span, center=False).sum()
            df_c["close_down"] = np.where(close.diff() < 0, volume.values, 0)
            down = df_c["close_down"].rolling(window=span, center=False).sum()
            df_c["close_stay"] = np.where(close.diff() == 0, volume.values, 0)
            stay = df_c["close_stay"].rolling(window=span, center=False).sum()
            drop_columns = ["close_up", "close_down", "close_stay"]
            df_c.drop(drop_columns, inplace=True, axis=1)
            df_c[f"volume_vr1_{span}"] = (up + stay / 2) / (down + stay / 2) * 100
            df_c[f"volume_vr2_{span}"] = (up + stay / 2) / (up + down + stay) * 100
            self.lag_make(df_c, df_c[f"volume_vr1_{span}"])
            self.lag_make(df_c, df_c[f"volume_vr2_{span}"])

        # volumeratio ボリューム・レシオ
        """
        window = 算定期間
        https://search.sbisec.co.jp/v2/popwin/tools/hyper/RM_08.pdf
        """
        volume_volumeratio_span = [5, 10, 20, 25, 50, 75]
        for window in volume_volumeratio_span:
            diff = close.diff(1)
            up_direction = pd.Series(
                np.where(diff < 0, 0.0, np.where(diff == 0, volume / 2, volume))
            )
            up_direction.iloc[0] = 0
            down_direction = pd.Series(
                np.where(diff > 0, 0.0, np.where(diff == 0, volume / 2, volume))
            )
            down_direction.iloc[0] = 0
            up_sum = up_direction.rolling(window).sum()
            down_sum = down_direction.rolling(window).sum()
            df_c[f"volume_volumeratio{window}"] = up_sum / down_sum * 100
            self.lag_make(df_c, df_c[f"volume_volumeratio{window}"])

        # Volume Zone Oscillator
        """
        window = 算定期間
        http://exceltechnical.web.fc2.com/vzo.html
        """
        volume_vzo_span = [5, 10, 25, 50, 75]
        for window in volume_vzo_span:
            cc_diff = close.values - close.shift(1).values
            cc_diff = np.where(cc_diff > 0, 1, -1)
            r = cc_diff * volume
            r.iloc[0] = np.nan
            vp = ta.trend.ema_indicator(r, window=window, fillna=0)
            volume_tv = volume.copy()
            volume_tv.iloc[0] = np.nan
            tv = ta.trend.ema_indicator(volume_tv, window=window, fillna=0)
            df_c[f"volume_vzo{window}"] = vp / tv * 100
            self.lag_make(df_c, df_c[f"volume_vzo{window}"])

        new_columns = df_c.columns
        self.columns_trand = np.setdiff1d(new_columns, old_columns)

        return df_c

    def others(self, df, open, high, low, close, volume):
        df_c = df.copy()
        old_columns = df_c.columns

        # 移動平均カラム作成
        others_ma_span = [5, 10, 25, 50, 75]
        for span in others_ma_span:
            df_c[f"others_ma{span}"] = ta.trend.sma_indicator(close, window=span)

        # DailyReturnIndicator 自作
        dr = (close / close.shift(1)) - 1
        dr *= 100
        df_c["others_dr"] = dr
        self.lag_make(df_c, df_c["others_dr"])

        # DailyLogReturnIndicator
        df_c["others_dlr"] = ta.others.DailyLogReturnIndicator(
            close, fillna=False
        ).daily_log_return()
        self.lag_make(df_c, df_c["others_dlr"])

        # CumulativeReturnIndicator
        df_c["others_cr"] = ta.others.CumulativeReturnIndicator(
            close, fillna=False
        ).cumulative_return()
        self.lag_make(df_c, df_c["others_cr"])

        # 移動平均乖離率
        others_sp_ma_span = [5, 10, 25, 50, 75]
        for span in others_sp_ma_span:
            df_c["others_sp_" + f"ma{span}"] = (close / df_c[f"others_ma{span}"]) - 1
            self.lag_make(df_c, df_c["others_sp_" + f"ma{span}"])

        # pivot　翌日分足用
        df_c["others_pivot_price"] = (close + high + low) / 3
        df_c["others_pivot_hbop"] = df_c["others_pivot_price"] * 2 - low * 2 + high
        df_c["others_pivot_r2"] = df_c["others_pivot_price"] + high - low
        df_c["others_pivot_r1"] = df_c["others_pivot_price"] * 2 - low
        df_c["others_pivot_s1"] = df_c["others_pivot_price"] * 2 - high
        df_c["others_pivot_s2"] = df_c["others_pivot_price"] - high + low
        df_c["others_pivot_lbop"] = df_c["others_pivot_price"] * 2 - high * 2 + low
        self.lag_make(df_c, df_c["others_pivot_price"])
        self.lag_make(df_c, df_c["others_pivot_hbop"])
        self.lag_make(df_c, df_c["others_pivot_r2"])
        self.lag_make(df_c, df_c["others_pivot_r1"])
        self.lag_make(df_c, df_c["others_pivot_s1"])
        self.lag_make(df_c, df_c["others_pivot_s2"])
        self.lag_make(df_c, df_c["others_pivot_lbop"])

        # fibonacci
        # 押しと戻しの違いは結果が反転しているだけ。どちらか削除する？
        """
        https://www.rakuten-sec.co.jp/MarketSpeed/onLineHelp/msman2_5_2_22.html
        """
        fibo_span = [5, 10, 25, 50, 75]
        bairitsu = [0.382, 0.5, 0.618]
        for span in fibo_span:
            df_c[f"max_{span}"] = high.rolling(window=span).max()
            df_c[f"min_{span}"] = low.rolling(window=span).min()
            for b in bairitsu:
                df_c[f"others_fibo{span}_oshi{b}"] = df_c[f"max_{span}"] - (
                    (df_c[f"max_{span}"] - df_c[f"min_{span}"]) * b
                )
                df_c[f"others_fibo{span}_modoshi{b}"] = df_c[f"min_{span}"] + (
                    (df_c[f"max_{span}"] - df_c[f"min_{span}"]) * b
                )
                oshi = df_c[f"others_fibo{span}_oshi{b}"]
                df_c[f"others_fibo{span}_oshi{b}_st"] = oshi / close
                modoshi = df_c[f"others_fibo{span}_modoshi{b}"]
                df_c[f"others_fibo{span}_modoshi{b}_st"] = modoshi / close
                self.lag_make(df_c, df_c[f"others_fibo{span}_oshi{b}"])
                self.lag_make(df_c, df_c[f"others_fibo{span}_modoshi{b}"])
                self.lag_make(df_c, df_c[f"others_fibo{span}_oshi{b}_st"])
                self.lag_make(df_c, df_c[f"others_fibo{span}_modoshi{b}_st"])
            df_c.drop([f"max_{span}", f"min_{span}"], inplace=True, axis=1)

        # サイコロジカル
        """
        前日比プラスのサイコロジカル
        http://exceltechnical.web.fc2.com/psyline.html
        """
        psycholo_span = [5, 10, 12, 25, 50, 75]
        df_close = close.shift(1)
        df_win = close.copy()
        df_win.loc[close - df_close > 0] = 1
        df_win.loc[close - df_close <= 0] = 0
        df_win.iloc[0] = 0
        for span in psycholo_span:
            df_win_roll = df_win.rolling(span).sum()
            df_c[f"others_psychological_{span}day"] = df_win_roll / span * 100
            self.lag_make(df_c, df_c[f"others_psychological_{span}day"])

        new_columns = df_c.columns
        self.columns_trand = np.setdiff1d(new_columns, old_columns)

        return df_c
