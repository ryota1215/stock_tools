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
        d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if not d in d_obs]
        fig.update_xaxes(rangebreaks=[dict(values=d_breaks)])
        fig.show()

    def pairchart(
        self,
        df1,
        df2,
        start_day=19500101,
        end_day=21001231,
        indexing=False,
        corr=False,
        corr_span=50,
    ):
        # 引数追加 , start_day, end_day
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
        :param indexing:bool Trueなら上段の株価終値を100を基準とした終値指数化に変更
        :param corr:bool Trueなら下段に相関分析を表示する、Falseなら価格差分を表示
        :param corr_span:int corrの計算期間の変更
        :return ペアチャート
        """
        # jpxのdateカラムはDateのため変換
        if "date" not in df1.columns:
            df1 = df1.rename(
                columns={
                    f"Adjustment{c}": f"fix_{c.lower()}"
                    for c in ["Open", "High", "Low", "Close"]
                }
            )
            df1 = df1.rename(columns={"Date": "date", "Code": "CODE"})
        if "date" not in df2.columns:
            df2 = df2.rename(
                columns={
                    f"Adjustment{c}": f"fix_{c.lower()}"
                    for c in ["Open", "High", "Low", "Close"]
                }
            )
            df2 = df2.rename(columns={"Date": "date", "Code": "CODE"})
        code1 = df1["CODE"].iloc[0]
        code2 = df2["CODE"].iloc[0]
        # 期間範囲指定
        df1 = df1[(df1["date"] >= f"{start_day}") & (df1["date"] <= f"{end_day}")]
        df2 = df2[(df2["date"] >= f"{start_day}") & (df2["date"] <= f"{end_day}")]
        # 指数化の計算
        if indexing == True:
            df1 = df1.reset_index()
            df1["fix_close"] = df1["fix_close"] / df1["fix_close"].iloc[0] * 100
            df2 = df2.reset_index()
            df2["fix_close"] = df2["fix_close"] / df2["fix_close"].iloc[0] * 100
        # 差分or相関係数の作成
        if corr == False:
            param = df1["fix_close"] - df2["fix_close"]
            param_name = "差分"
        else:
            param = df1["fix_close"].rolling(corr_span).corr(df2["fix_close"])
            param_name = "相関分析"
        # subplotsで複数のグラフ画面を作成する
        fig = make_subplots(
            rows=2,  # 行数設定
            cols=1,  # 列数設定
            # shared_yaxes='all', #y軸を共有する
            shared_xaxes="all",  # x軸を共有する
            vertical_spacing=0.1,  # サブプロット行間のスペース
            row_heights=[3, 1],  # グラフの大きさ 相対的比率
            subplot_titles=["chart", param_name],  # グラフ上のタイトル設定
        )
        # add_traceでグラフを入れる
        fig.add_trace(
            go.Scatter(
                x=df1["date"],
                y=df1["fix_close"],
                mode="lines",
                name=f"{code1}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df1["date"],
                y=df2["fix_close"],
                mode="lines",
                name=f"{code2}",
            ),
            row=1,
            col=1,
        )
        # 下段に差分または相関係数のグラフを作成する
        fig.add_trace(
            go.Scatter(
                x=df1["date"],
                y=param,
                mode="lines",
                name=param_name,
            ),
            row=2,
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
        date = pd.to_datetime(df1["date"])
        # 日付リストを取得
        d_all = pd.date_range(start=df1["date"].iloc[0], end=df1["date"].iloc[-1])
        # 株価データの日付リストを取得
        d_obs = [d.strftime("%Y-%m-%d") for d in date]
        # 株価データの日付データに含まれていない日付を抽出
        d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if not d in d_obs]
        fig.update_xaxes(rangebreaks=[dict(values=d_breaks)])
        fig.show()

    def multiplot_matplotlib(
        self,
        df_list,
        day_period=30,
        is_plot_mean=False,
        is_plot_mean_alpha=0.3,
        log_y=False,
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
        :param log_y_linthresh: float 対数スケールを線形スケールに変更する値。0からどこまでを線形スケールにするか範囲を設定。値が小さいほど0付近の動きが大きくなる。
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
            all_returns = [
                returns
                for df in df_list
                for returns in (df["fix_close"] / df["fix_open"].iloc[0] - 1) * 100
            ]
            # 各日の騰落率の平均値を求める
            mean_daily_returns = [
                np.mean(
                    [
                        all_returns[i]
                        for i in range(len(all_returns))
                        if i % day_period == j
                    ]
                )
                for j in range(day_period)
            ]
            # 0から始まるようにする
            mean_daily_returns.insert(0, 0)
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
        if log_y:
            ax.set_yscale("symlog", base=log_y_base, linthresh=log_y_linthresh)
            ax.set_ylabel("Log Returns (%)")
        # グラフを表示
        plt.show()
