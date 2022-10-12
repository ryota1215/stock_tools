"""
要件
4本値と指定(複数)のテクニカル指標を表示する
オシレーター系は下段に図を追加する
複数銘柄比較機能を追加する

visualization.pyのclass visualizationをimportして
visualization(df,表示したいカラム名のリスト)でチャートを表示

サンプル : https://ailog.site/2022/04/08/2022/0408/

複数銘柄比較機能は、楽天を参考
https://marketspeed.jp/ms2/onlinehelp/ohm_007/ohm_007_02.html
"""


class visualization:
    def __init__(self, df):
        """ """

    def pairchart(df1, df2, start_day, end_day):
        """
        visualization.pairchart(df1,df2,スタート日付{"20200101"},エンド日付)
        これで出せるようにしてほしい
        """
