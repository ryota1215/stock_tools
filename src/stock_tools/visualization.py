import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

"""
要件
visualization.pyのclass visualizationをimportして  #ok.
4本値と指定(複数)のテクニカル指標を表示する<chart>  #ok.
オシレーター系は下段に図を追加する<chart> #ok.
visualization(df,表示したいカラム名のリスト)でチャートを表示 <chart> #ok.
サンプル : https://ailog.site/2022/04/08/2022/0408/

複数銘柄比較機能を追加する<pair_chart>
複数銘柄比較機能は、楽天を参考
https://marketspeed.jp/ms2/onlinehelp/ohm_007/ohm_007_02.html

docstringを書くようにしてください。
形式は 関数の説明 引数の説明 戻り値の説明
https://qiita.com/simonritchie/items/49e0813508cad4876b5a
"""


class visualization:
    def __init__(self):
        """ """

    def chart(self, df, technical_list=[], oscillator_list=[]):
        """
        チャートとオシレーターを表示する
        :param df 株価のdataframe
        :param technical_list:list of str  表示したいテクニカルのcolumnsリスト
        :param oscillator_list:list of str  表示したいオシレーターのcolumnsリスト
        :return チャート
        """
        # jpxのdateカラムはDateのため変換
        if "date" not in df.columns:
            df = df.rename(
                columns={
                    f"Adjustment{c}": f"fix_{c.lower()}"
                    for c in ["Open", "High", "Low", "Close"]
                }
            )
            df = df.rename(columns={"Date": "date", "Code": "CODE"})
        code = df["CODE"].iloc[0]
        # subplotsで複数のグラフ画面を作成する
        if len(oscillator_list) != 0:
            heights_list = np.ones(1 + len(oscillator_list))  # row_heightsグラフ高さ倍率リストの作成
            heights_list[0] = 3  # 一つ目のグラフとそれ以降の倍率を3：1にしている
            heights_list = heights_list.tolist()  # list形式に変換
            subtitle_name = oscillator_list.copy()
            subtitle_name.insert(0, f"{code}")
            fig = make_subplots(
                rows=1 + len(oscillator_list),  # 行数設定
                cols=1,  # 列数設定
                # shared_yaxes='all', #y軸を共有する
                shared_xaxes="all",  # x軸を共有する
                vertical_spacing=0.1,  # サブプロット行間のスペース
                subplot_titles=(subtitle_name),  # グラフ上のタイトル設定
                row_heights=heights_list,  # グラフの大きさ 相対的比率
            )
        # add_traceでグラフを入れる
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["fix_open"],
                high=df["fix_high"],
                low=df["fix_low"],
                close=df["fix_close"],
                name="ローソク足",
            ),
            row=1,
            col=1,
        )
        # 引数にtechnical_listがあればグラフに表示
        if len(technical_list) != 0:
            for technical in technical_list:
                fig.add_trace(
                    go.Scatter(
                        x=df["date"],
                        y=df[f"{technical}"],
                        mode="lines",
                        name=f"{technical}",
                    ),
                    row=1,
                    col=1,
                )
        # 引数にoscillator_listがあればグラフを作成
        for index, column_name in enumerate(oscillator_list):
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df[f"{column_name}"],
                    mode="lines",
                    name=f"{column_name}",
                ),
                row=index + 2,
                col=1,
            )
        # layoutでレイアウト設定をする
        fig.update_layout(
            # グラフタイトル
            # title_text="OHLC",
            # 凡例表示
            showlegend=True,
            # 凡例の位置変更
            xaxis_rangeslider=dict(
                visible=False,
            ),  # レンジスライダー削除
            yaxis=dict(fixedrange=False),  # y軸のズームを可能にする
            height=800,  # グラフ高さの編集
            width=900,  # グラフ横幅の編集
        )
        # 土日祝の隙間を削除するためにrangebreaks作成して、update_xaxesで反映させる
        # 日付objectをdatetime型に変換
        date = pd.to_datetime(df["date"])
        # 日付リストを取得
        d_all = pd.date_range(start=df["date"].iloc[0], end=df["date"].iloc[-1])
        # 株価データの日付リストを取得
        d_obs = [d.strftime("%Y-%m-%d") for d in date]
        # 株価データの日付データに含まれていない日付を抽出
        d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if d not in d_obs]
        fig.update_xaxes(rangebreaks=[dict(values=d_breaks)])
        fig.show()

    def pairchart(
        self,
        df1,
        df2,
        start_day=19500101,
        end_day=21001231,
        is_firstday_open_start=False,
        is_indexing=True,
        is_corr=True,
        corr_span=5,
        is_add_volume=False,
        is_add_volume_log=False,
        is_diff=True,
    ):
        """
        visualization.pairchart(df1,df2,スタート日付{"20200101"},エンド日付)
        これで出せるようにしてほしい
        """
        """
        ペアチャートを表示する
        :param df1:df 基準とする株価のdataframe
        :param df2:df 株価のdataframe
        :param start_day:int (例20200101) 表示期間の始まりの日
        :param end_day:int(例20200101) 表示期間の終わりの日
        :param is_firstday_open_start:bool Trueなら初日の初値を基準に騰落率を計算する。初値は期間初日の前日の日付で表示している。
        :param is_indexing:bool Trueなら上段の株価終値を100を基準とした終値指数化に変更
        :param is_corr:bool Trueなら下段に相関分析を表示する
        :param corr_span:int corrの計算期間の変更
        :param is_add_volume:bool Trueなら下段に出来高表示する
        :param is_add_volume_log:bool Trueなら出来高の表示をlog表示にする（底は10で固定）
        :param is_diff:bool Trueなら上段の値の差分を2軸目に表示する
        :return ペアチャート
        """
        # １　データの前処理はここ
        # jpxのdateカラムはDateのため変換
        if "date" not in df1.columns:
            df1 = df1.rename(
                columns={
                    f"Adjustment{c}": f"fix_{c.lower()}"
                    for c in ["Open", "High", "Low", "Close", "Volume"]
                }
            )
            df1 = df1.rename(columns={"Date": "date", "Code": "CODE"})
        if "date" not in df2.columns:
            df2 = df2.rename(
                columns={
                    f"Adjustment{c}": f"fix_{c.lower()}"
                    for c in ["Open", "High", "Low", "Close", "Volume"]
                }
            )
            df2 = df2.rename(columns={"Date": "date", "Code": "CODE"})
        code1 = df1["CODE"].iloc[0]
        code2 = df2["CODE"].iloc[0]
        # 期間範囲指定
        df1 = df1[(df1["date"] >= f"{start_day}") & (df1["date"] <= f"{end_day}")]
        df2 = df2[(df2["date"] >= f"{start_day}") & (df2["date"] <= f"{end_day}")]
        # 目的の株価をチャート表示するためのtarget_priceカラム作成
        df1["target_price"] = df1["fix_close"]
        df2["target_price"] = df2["fix_close"]
        # 初日初値を基準にtarget_priceを計算
        if is_firstday_open_start:
            # 1行目を複製して1行目に追加
            df1.loc[-1] = df1.iloc[0]
            df1.index = df1.index + 1
            df1 = df1.sort_index()
            # target_priceの1行目を初値に置換
            df1["target_price"].iloc[0] = df1["fix_open"].iloc[0]
            df1["fix_volume"].iloc[0] = 0
            # 初日の日付を期間初日の前日に設定
            df1["date"].iloc[0] = df1["date"].iloc[0] - pd.Timedelta(days=1)
            # 1行目を複製して1行目に追加
            df2.loc[-1] = df2.iloc[0]
            df2.index = df2.index + 1
            df2 = df2.sort_index()
            # target_priceの1行目を初値に置換
            df2["target_price"].iloc[0] = df2["fix_open"].iloc[0]
            df2["fix_volume"].iloc[0] = 0
            # 初日の日付を期間初日の前日に設定
            df2["date"].iloc[0] = df2["date"].iloc[0] - pd.Timedelta(days=1)
        # 指数化の計算
        if is_indexing:
            df1 = df1.reset_index()
            df1["target_price"] = (
                df1["target_price"] / df1["target_price"].iloc[0] * 100
            )
            df2 = df2.reset_index()
            df2["target_price"] = (
                df2["target_price"] / df2["target_price"].iloc[0] * 100
            )
        # 差分の作成
        if is_diff:
            param_diff = df1["target_price"] - df2["target_price"]
            param_name_diff = "差分"
        param_name = ""
        # 相関係数の作成
        if is_corr:
            param = df1["target_price"].rolling(corr_span).corr(df2["target_price"])
            param_name = "相関分析"
        # 出来高にparam変更
        if is_add_volume:
            param_name = "出来高"
        # ２　make_subplots設定はここ
        # subplotsで複数のグラフ画面を作成する
        row_heights_param = [3, 1]
        row_param = 2
        # 出来高のsubplots設定変更（3:2のおおきさに変更)
        if is_add_volume:
            row_heights_param[1] = 2
        fig = make_subplots(
            rows=row_param,  # 行数設定
            cols=1,  # 列数設定
            # shared_yaxes='all', #y軸を共有する
            shared_xaxes="all",  # x軸を共有する
            vertical_spacing=0.1,  # サブプロット行間のスペース
            row_heights=row_heights_param,  # グラフの大きさ 相対的比率
            subplot_titles=["chart", param_name],  # グラフ上のタイトル設定
            specs=[[{"secondary_y": True}], [{}]],  # 2軸目追加
        )
        # ３　add_trace グラフの挿入はここ
        # add_traceでグラフを入れる
        fig.add_trace(
            go.Scatter(
                x=df1["date"],
                y=df1["target_price"],
                mode="lines",
                name=f"{code1}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df1["date"],
                y=df2["target_price"],
                mode="lines",
                name=f"{code2}",
            ),
            row=1,
            col=1,
        )
        if is_diff:
            fig.add_trace(
                go.Scatter(
                    x=df1["date"],
                    y=param_diff,
                    mode="lines",
                    name=f"{param_name_diff}",
                    marker_color="gold",
                ),
                row=1,
                col=1,
                secondary_y=True,
            )
            fig.update_yaxes(
                title_text="差分",  # 2軸目のラベル設定
                row=1,
                col=1,
                secondary_y=True,  # 2軸目の軸を設定
            )
        # 下段に差分または相関係数のグラフを作成する
        if is_corr and not is_add_volume:
            fig.add_trace(
                go.Scatter(
                    x=df1["date"],
                    y=param,
                    mode="lines",
                    name=param_name,
                    marker_color="limegreen",
                ),
                row=2,
                col=1,
            )
        if is_add_volume:
            fig.add_trace(
                go.Bar(
                    x=df1["date"],
                    y=df1["fix_volume"],
                    # mode="lines",
                    name=f"{code1}",
                    opacity=0.3,
                    marker_color="blue",
                ),
                row=2,
                col=1,
                # secondary_y=True,
            )
            fig.add_trace(
                go.Bar(
                    x=df2["date"],
                    y=df2["fix_volume"],
                    # mode="lines",
                    name=f"{code2}",
                    opacity=0.3,
                    marker_color="red",
                ),
                row=2,
                col=1,
                # secondary_y=True,
            )
        # ４　レイアウト設定はここ
        # layoutでレイアウト設定をする
        height_param = 800
        if is_add_volume:
            height_param += 100
        fig.update_layout(
            # グラフタイトル
            # title_text="OHLC",
            # 凡例表示
            showlegend=True,
            # 凡例の位置変更
            xaxis_rangeslider=dict(
                visible=False,
            ),  # レンジスライダー削除
            yaxis=dict(fixedrange=False),  # y軸のズームを可能にする
            height=height_param,  # グラフ高さの編集
            width=900,  # グラフ横幅の編集
            margin=dict(l=0, r=0, t=30, b=0),  # グラフ間の隙間幅の調整
        )
        # 土日祝の隙間を削除するためにrangebreaks作成して、update_xaxesで反映させる
        # 日付objectをdatetime型に変換
        date = pd.to_datetime(df1["date"])
        # 日付リストを取得
        d_all = pd.date_range(start=df1["date"].iloc[0], end=df1["date"].iloc[-1])
        # 株価データの日付リストを取得
        d_obs = [d.strftime("%Y-%m-%d") for d in date]
        # 株価データの日付データに含まれていない日付を抽出
        d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if d not in d_obs]
        fig.update_xaxes(rangebreaks=[dict(values=d_breaks)])
        # ５　追加レイアウト設定はここ
        # 相関分析時にy軸の最大最小値の固定化
        if is_corr and not is_add_volume:
            fig.update_yaxes(range=[-1, 1], row=2, col=1)
        # y軸をlogに変更する
        if is_add_volume and is_add_volume_log:
            fig.update_yaxes(type="log", row=2, col=1)
        # グラフ出力
        fig.show()

    def multiplot_matplotlib(
        self,
        df_list,
        day_period=30,
        is_plot_mean=False,
        is_plot_mean_alpha=0.3,
        is_log_y=False,
        log_y_base=10,
        log_y_linthresh=10.0,
    ):
        """
        複数の株価チャートを表示する
        :param df_list: list of DataFrame 株価データを含むDataFrameのリスト
        :param day_period: int イベント日からの日数
        :param is_plot_mean: bool 銘柄群平均騰落率をplotするかどうか
        :param is_plot_mean_alpha: int is_plot_meanのplotの濃さ
        :param log_y: bool y軸をlog表示にする
        :param log_y_base: float 対数の底の値
        :param log_y_linthresh: float 対数から線形スケールに変更する。0からどこまでにするか範囲を設定。値が小さいほど0付近の動きが大きくなる。
        :return: 複数銘柄のチャート
        """
        # プロットの設定
        fig, ax = plt.subplots(figsize=(15, 8))
        # 複数の株価を同じグラフにプロット
        for i, df in enumerate(df_list):
            # jpxのdateカラムはDateのため変換
            if "date" not in df.columns:
                df = df.rename(
                    columns={
                        f"Adjustment{c}": f"fix_{c.lower()}"
                        for c in ["Open", "High", "Low", "Close"]
                    }
                )
                df = df.rename(columns={"Date": "date", "Code": "CODE"})
            # インデックスをリセット
            df = df.reset_index(drop=True)
            # 表示する期間を制限
            df = df.iloc[:day_period]
            # 元のdfにも反映させる
            df_list[i] = df
            # 騰落率の計算
            returns = (df["fix_close"] / df["fix_open"].iloc[0] - 1) * 100
            # 0から開始するため最初の行に0を追加して
            returns = pd.Series([0]).append(returns, ignore_index=True)
            # 0から始めるためindexの数を一つ増やす
            x_index_values = df.index.values
            x_index_values = np.append(x_index_values, x_index_values[-1] + 1)
            # グラフの描画
            if is_plot_mean:
                ax.plot(
                    x_index_values,
                    returns,
                    marker=".",
                    alpha=is_plot_mean_alpha,
                    label=df["CODE"].iloc[0],
                )
            else:
                ax.plot(
                    x_index_values,
                    returns,
                    marker=".",
                    alpha=1.0,
                    label=df["CODE"].iloc[0],
                )
        if is_plot_mean:
            # 全体の騰落率の平均値を求める
            all_returns = []
            for df in df_list:
                returns = ((df["fix_close"] / df["fix_open"].iloc[0] - 1) * 100).values
                # day_periodの日数に足りない要素数分np.nanを挿入する
                if len(returns) < day_period:
                    pad_width = day_period - len(returns)
                    returns = np.pad(
                        returns, (0, pad_width), mode="constant", constant_values=np.nan
                    )
                all_returns.append(returns)
            # 各日の騰落率の平均値を求める(np.nanを除いた数値のデータのみで平均値を出す)
            mean_daily_returns = np.nanmean(all_returns, axis=0)
            # 初日の0%を追加する
            mean_daily_returns = np.insert(mean_daily_returns, 0, 0)
            # 全体の騰落率の平均値をプロット
            ax.plot(mean_daily_returns, marker=".", label="Mean Returns", color="red")
        # グラフのラベルとタイトルを設定
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns (%)")
        ax.set_title("Price Returns")
        # 凡例を表示
        fig.legend()
        fig.subplots_adjust(right=0.8)
        # y軸を対数表示
        if is_log_y:
            ax.set_yscale("symlog", base=log_y_base, linthresh=log_y_linthresh)
            ax.set_ylabel("Log Returns (%)")
        # グラフを表示
        plt.show()
