import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        code = df["CODE"][0]
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

    def pairchart(self, df1, df2):
        # 引数追加 , start_day, end_day
        """
        visualization.pairchart(df1,df2,スタート日付{"20200101"},エンド日付)
        これで出せるようにしてほしい
        """
        """
        ペアチャートを表示する
        :param df1 株価のdataframe
        :param df2 株価のdataframe
        :param start_day{"20200101"} 表示期間の始まりの日
        :param end_day{"20200101"} 表示期間の終わりの日
        :return チャート
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

        code1 = df1["CODE"][0]
        code2 = df2["CODE"][0]
        # subplotsで複数のグラフ画面を作成する
        fig = make_subplots(
            rows=2,  # 行数設定
            cols=1,  # 列数設定
            # shared_yaxes='all', #y軸を共有する
            shared_xaxes="all",  # x軸を共有する
            vertical_spacing=0.1,  # サブプロット行間のスペース
            subplot_titles=("chart", "差分"),  # グラフ上のタイトル設定
            row_heights=[3, 1],  # グラフの大きさ 相対的比率
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
                x=df2["date"],
                y=df2["fix_open"],
                mode="lines",
                name=f"{code2}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df1["date"],
                y=df1["fix_close"] - df2["fix_open"],
                mode="lines",
                name="差分",
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
