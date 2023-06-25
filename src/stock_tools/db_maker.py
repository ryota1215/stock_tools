import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import psycopg2
from sqlalchemy import create_engine
import requests
import json
from stock_tools import google_auth


class db_maker:
    def __init__(
        self,
        dict_account_info: dict = {},
        dict_postgre_info: dict = {},
        is_make_tabel: bool = False,
    ):
        """
        dict_account_info : jquantsのmailaddressとpassword {"mailaddress":"...","password":"..."}
        dict_postgre_info : postgresqlのDB情報 {"user" : "...", "password" : "..." , "port" : "..." ,"database" : "..."}
        is_make_tabel : テーブルを作成するか否か

        """
        self.dict_account_info = dict_account_info
        self.is_make_tabel = is_make_tabel
        # dbアクセス用
        self.engine = create_engine(
            f"postgresql+psycopg2://{dict_postgre_info['user']}:{dict_postgre_info['password']}@localhost:{dict_postgre_info['port']}/{dict_postgre_info['database']}"
        )
        self.connection = psycopg2.connect(**dict_postgre_info)
        self.cursor = self.connection.cursor()
        self.access_token = self.get_token()

    def db_update(self):
        """
        全てのテーブルを更新
        """
        self.db_listed()
        print("Done : db_listed")
        self.db_stockprice()
        print("Done : db_stockprice")
        self.db_trades_spec()
        print("Done : db_trades_spec")
        self.db_weekly_margin_interest()
        print("Done : db_weekly_margin_interest")
        self.db_short_selling()
        print("Done : db_short_selling")
        self.db_breakdown()
        print("Done : db_breakdown")
        self.db_statements()
        print("Done : db_statements")
        self.db_dividend()
        print("Done : db_dividend")
        self.db_option()
        print("Done : db_option")

    def db_drop_duplicated(self, table_name):
        """
        テーブルの重複を削除する
        table_name : テーブル名
        """

        # カラム名のリストを取得する
        self.cursor.execute(
            f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"
        )
        column_names = [f'"{column[0]}"' for column in self.cursor.fetchall()]

        # 全てのカラムが重複している行を削除する
        self.cursor.execute(
            f"DELETE FROM {table_name} WHERE ctid NOT IN (SELECT MIN(ctid) FROM {table_name} GROUP BY {', '.join(column_names)})"
        )

        # 変更を保存する
        self.connection.commit()

    def get_token(self):
        """
        アクセストークンを取得
        """
        r_post = requests.post(
            "https://api.jquants.com/v1/token/auth_user",
            data=json.dumps(self.dict_account_info),
        )
        ref_token = r_post.json()["refreshToken"]
        r_post = requests.post(
            f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={ref_token}"
        )
        access_token = r_post.json()["idToken"]

        return access_token

    def db_listed(self):
        """
        上場銘柄一覧DB作成
        """
        tabel_name = "listed"

        if self.is_make_tabel:
            dt_origin = "20080507"
            dt_today = datetime.datetime.strftime(
                datetime.date.today(), format="%Y%m%d"
            )
            dt_range = pd.date_range(dt_origin, dt_today)

            tabel_name = "listed"

            datas = []
            for dt in tqdm(dt_range):
                dt = datetime.datetime.strftime(dt, format="%Y%m%d")
                # apiでデータ取得
                headers = {"Authorization": "Bearer {}".format(self.access_token)}
                r = requests.get(
                    f"https://api.jquants.com/v1/listed/info?date={dt}", headers=headers
                )
                data = r.json()

                assert "info" in data.keys(), "APIError listed"

                # assert "info" in data.keys(), "APIError listed"
                datas.append(pd.DataFrame(data["info"]))

            # DB作成
            df_listed = pd.concat(datas)
            df_listed.to_sql(tabel_name, self.engine, if_exists="replace", index=False)

        else:
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get("https://api.jquants.com/v1/listed/info", headers=headers)
            data = r.json()

            assert "info" in data.keys(), "APIError listed"
            df_listed = pd.DataFrame(data["info"])

            # DB更新
            df_listed.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def get_codelist(self):
        """
        銘柄コードリスト取得
        """
        df_listed = pd.read_sql(sql="SELECT * FROM listed;", con=self.connection)
        list_code = np.unique(df_listed["Code"])
        return list_code

    def db_stockprice(self):
        """
        株価DB作成
        """
        list_code = self.get_codelist()
        tabel_name = "stockprice"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20080507"
        else:
            self.cursor.execute(f'SELECT MAX("Date") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]
        dt_today = datetime.datetime.strftime(
            datetime.datetime.today().date(), "%Y%m%d"
        )

        cnt = 0
        for code in tqdm(list_code):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/prices/daily_quotes?code={code}&from={dt_origin}&to={dt_today}",
                headers=headers,
            )
            data = r.json()
            assert (
                "daily_quotes" in data.keys()
            ), f"APIError stockprice code : {code} dt_start : {dt_origin}"
            df = pd.DataFrame(data["daily_quotes"])

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_trades_spec(self):
        """
        投資部門別情報DB作成
        """
        tabel_name = "trades_spec"
        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20080116"
            list_market = [
                "TSE1st",
                "TSE2nd",
                "TSEMothers",
                "TSEJASDAQ",
                "TSEPrime",
                "TSEStandard",
                "TSEGrowth",
                "TokyoNagoya",
            ]
        else:
            self.cursor.execute(f'SELECT MAX("PublishedDate") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]
            list_market = ["TSEPrime", "TSEStandard", "TSEGrowth", "TokyoNagoya"]
        dt_today = datetime.datetime.strftime(
            datetime.datetime.today().date(), "%Y%m%d"
        )

        cnt = 0
        for market in tqdm(list_market):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/markets/trades_spec?section={market}&from={dt_origin}&to={dt_today}",
                headers=headers,
            )
            data = r.json()
            assert (
                "trades_spec" in data.keys()
            ), f"APIError trades_spec market : {market} dt_start : {dt_origin}"
            df = pd.DataFrame(data["trades_spec"])

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_weekly_margin_interest(self):
        """
        信用取引週末残高
        """
        list_code = self.get_codelist()
        tabel_name = "weekly_margin_interest"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20120210"
        else:
            self.cursor.execute(f'SELECT MAX("Date") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]
        dt_today = datetime.datetime.strftime(
            datetime.datetime.today().date(), "%Y%m%d"
        )

        cnt = 0
        for code in tqdm(list_code):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/markets/weekly_margin_interest?code={code}&from={dt_origin}&to={dt_today}",
                headers=headers,
            )
            data = r.json()
            assert (
                "weekly_margin_interest" in data.keys()
            ), f"APIError weekly_margin_interest code : {code} dt_start : {dt_origin}"
            df = pd.DataFrame(data["weekly_margin_interest"])

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_short_selling(self):
        """
        業種別空売り比率
        """
        list_sec = [
            "0050",
            "9999",
            "2050",
            "3500",
            "1050",
            "9050",
            "8050",
            "3600",
            "3550",
            "5250",
            "3050",
            "3250",
            "5050",
            "7200",
            "6100",
            "6050",
            "3800",
            "3200",
            "3100",
            "3650",
            "3400",
            "7100",
            "3700",
            "3300",
            "3150",
            "3750",
            "3350",
            "3450",
            "7050",
            "7150",
            "5200",
            "5100",
            "5150",
            "4050",
        ]
        tabel_name = "short_selling"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20081105"
        else:
            self.cursor.execute(f'SELECT MAX("Date") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]
        dt_today = datetime.datetime.strftime(
            datetime.datetime.today().date(), "%Y%m%d"
        )

        cnt = 0
        for sec in tqdm(list_sec):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/markets/short_selling?sector33code={sec}&from={dt_origin}&to={dt_today}",
                headers=headers,
            )
            data = r.json()
            assert (
                "short_selling" in data.keys()
            ), f"APIError short_selling sec : {sec} dt_start : {dt_origin}"
            df = pd.DataFrame(data["short_selling"])

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_breakdown(self):
        """
        売買内訳データ
        """
        list_code = self.get_codelist()
        tabel_name = "breakdown"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20150401"
        else:
            self.cursor.execute(f'SELECT MAX("Date") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]
        dt_today = datetime.datetime.strftime(
            datetime.datetime.today().date(), "%Y%m%d"
        )

        cnt = 0
        for code in tqdm(list_code):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/markets/breakdown?code={code}&from={dt_origin}&to={dt_today}",
                headers=headers,
            )
            data = r.json()
            assert (
                "breakdown" in data.keys()
            ), f"APIError breakdown code : {code} dt_start : {dt_origin}"
            df = pd.DataFrame(data["breakdown"])

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_statements(self):
        """
        財務データ
        """
        list_code = self.get_codelist()
        tabel_name = "statements"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20080707"
        else:
            self.cursor.execute(f'SELECT MAX("DisclosedDate") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]

        cnt = 0
        for code in tqdm(list_code):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/fins/statements?code={code}",
                headers=headers,
            )
            data = r.json()
            assert "statements" in data.keys(), f"APIError statements code : {code} "
            df = pd.DataFrame(data["statements"])
            if df.empty:
                continue

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df = df[df["DisclosedDate"] > dt_origin]
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_dividend(self):
        """
        配当データ
        """
        list_code = self.get_codelist()
        tabel_name = "dividend"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20130220"
        else:
            self.cursor.execute(f'SELECT MAX("AnnouncementDate") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]

        cnt = 0
        for code in tqdm(list_code):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/fins/dividend?code={code}", headers=headers
            )
            data = r.json()
            assert "dividend" in data.keys(), f"APIError dividend code : {code} "
            df = pd.DataFrame(data["dividend"])
            if df.empty:
                continue

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.replace("-", np.nan, inplace=True)
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                if self.is_make_tabel:
                    df = df[df["AnnouncementDate"] > dt_origin]
                    if df.empty:
                        continue
                df.replace("-", np.nan, inplace=True)
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)

    def db_option(self):
        """
        オプションデータ
        """
        tabel_name = "option"

        # DB作成時には、指定日~今日まで 更新時には、DBの最終日~今日まで
        if self.is_make_tabel:
            dt_origin = "20160719"
        else:
            self.cursor.execute(f'SELECT MAX("Date") FROM {tabel_name}')
            dt_origin = self.cursor.fetchone()[0]
        dt_today = datetime.datetime.strftime(
            datetime.datetime.today().date(), "%Y%m%d"
        )
        # 東証の営業日を取得する
        headers = {"Authorization": "Bearer {}".format(self.access_token)}
        r = requests.get(
            f"https://api.jquants.com/v1/prices/daily_quotes?code=72030&from={dt_origin}&to={dt_today}",
            headers=headers,
        )
        data = r.json()
        assert (
            "daily_quotes" in data.keys()
        ), f"APIError stockprice code : 72030 dt_start : {dt_origin}"
        list_date = pd.DataFrame(data["daily_quotes"])["Date"]

        cnt = 0
        for dt in tqdm(list_date):
            # apiでデータ取得
            headers = {"Authorization": "Bearer {}".format(self.access_token)}
            r = requests.get(
                f"https://api.jquants.com/v1/option/index_option?date={dt}",
                headers=headers,
            )
            data = r.json()
            assert (
                "index_option" in data.keys()
            ), f"APIError index_option dt_start : {dt}"
            df = pd.DataFrame(data["index_option"])

            if self.is_make_tabel & (cnt == 0):
                # DB作成
                df.replace("-", np.nan, inplace=True)
                df.replace("", np.nan, inplace=True)
                df.to_sql(tabel_name, self.engine, if_exists="replace", index=False)
                cnt += 1
            else:
                # DB更新
                df.replace("-", np.nan, inplace=True)
                df.replace("", np.nan, inplace=True)
                df.to_sql(tabel_name, self.engine, if_exists="append", index=False)


if __name__ == "__main__":
    with open(r"../../../jquants.json") as f:
        d = json.load(f)

    jquants_api_userinfo = d["user_info"]
    dict_postgre_info = d["db_info02"]  # db_info
    db = db_maker(
        dict_account_info=jquants_api_userinfo,
        dict_postgre_info=dict_postgre_info,
        is_make_tabel=True,
    )
    db.db_update()
    db.cursor.close()
    db.connection.close()
    google_auth.send_message("Daily update Done")
