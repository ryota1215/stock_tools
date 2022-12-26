import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from urllib import request
import requests
from bs4 import BeautifulSoup
import datetime
import re
import japanize_matplotlib
japanize_matplotlib.japanize()

with open("../../../jsons/tokens.json", 'r') as j:
    WEBHOOK_ID = json.load(j)["discode_webhook_economic_notice"]
IMG_PATH = "../../imgs/economic_schedule.png"


def str_2_val(s):
    if "*" in s | "" == s:
        return np.nan
    s = s[:-1] if "%" in s else s
    return float(s)


def is_date_digit_number(string):
    if len(string[:9]) != 9:
        return False
    for c in string:
        if not c.isdigit():
            return False
    return True


if __name__ == "__main__":
    # データ取得
    with open("../../../jsons/url.json", 'r') as j:
        target = json.load(j)["economic_schedule"]
    response = request.urlopen(target)
    soup = BeautifulSoup(response)
    response.close()

    contents = soup.find_all(id=True)
    records = [
        contents[i]
        for i in np.arange(0, len(contents))[
            [is_date_digit_number(content["id"]) for content in contents]
        ]
    ]

    lst = []
    for record in records:
        flag = False
        flag_date = False
        for i, content in enumerate(record.find_all("td")):
            try:
                if i == 0:
                    str_date = str(datetime.date.today().year) + "/" + \
                        re.sub("[(月火水木金)]", "", content.text)
                    date = datetime.datetime.strptime(str_date, "%Y/%m/%d").date()
            except Exception:
                flag = True
                if i == 0:
                    time = content.text
                    if "--:--" == time:
                        time = "未定 or 全日"
                    elif int(time[:2]) >= 24:
                        time = '{:02}'.format(int(time[:2]) - 24) + time[2:]
                        flag_date = True
                        time = pd.to_datetime(time).time()
                    else:
                        time = pd.to_datetime(time).time()
            # 日付ないパターン
            if flag:
                if i == 1:
                    country = content.text
                elif i == 2:
                    event = content.text
                elif i == 3:
                    star = content.text.count("★")
                elif i == 4:
                    if '' == content.text or "*" in content.text:
                        val1 = np.nan
                    else:
                        val1 = content.text
                elif i == 5:
                    if '' == content.text or "*" in content.text:
                        val2 = np.nan
                    else:
                        val2 = content.text
                elif i == 6:
                    if '' == content.text or "*" in content.text:
                        val3 = np.nan
                    else:
                        val3 = content.text

            # 日付あるパターン
            else:
                if i == 1:
                    time = content.text
                    if "--:--" == time:
                        time = "未定 or 全日"
                    elif int(time[:2]) >= 24:
                        time = '{:02}'.format(int(time[:2]) - 24) + time[2:]
                        flag_date = True
                        time = pd.to_datetime(time).time()
                    else:
                        time = pd.to_datetime(time).time()
                elif i == 2:
                    country = content.text
                elif i == 3:
                    event = content.text
                elif i == 4:
                    star = record.text.count("★")
                    vals = np.array([str_2_val(s.text) for s in content.find_all("td")])
                    if len(vals) == 0:
                        vals = np.array([np.nan, np.nan, np.nan])
        if flag:
            if flag_date:
                lst.append([date + datetime.timedelta(days=1), time,
                           country, event, star, val1, val2, val3])
            else:
                lst.append([date, time, country, event, star, val1, val2, val3])
        else:
            if flag_date:
                lst.append([date + datetime.timedelta(days=1), time,
                           country, event, star, vals[0], vals[1], vals[2]])
            else:
                lst.append([date, time, country, event, star, vals[0], vals[1], vals[2]])
    lst = np.array(lst)

    df_economic_event = pd.DataFrame(
        lst,
        columns=[
            "event_date",
            "event_time",
            "country",
            "event",
            "importance",
            "pred",
            "act",
            "previous",
        ],
    )
    if (
        (1 in pd.to_datetime(df_economic_event["event_date"]).dt.month)
        and (datetime.date.today().day >= 24)
        and (datetime.date.today().month == 12)
    ):
        df_economic_event.loc[
            df_economic_event.event_date < datetime.date.today(), "event_date"
        ] = df_economic_event.event_date[
            df_economic_event.event_date < datetime.date.today()
        ] + datetime.timedelta(
            days=365
        )
    df_economic_event = df_economic_event.drop_duplicates(
        subset=["event_date", "event", "country"]
    ).sort_values(by=["event_date", "event_time"])
    df_economic_event_select = df_economic_event[
        (df_economic_event["event"] == "休場") | (df_economic_event["importance"] > 1)
    ]

    w1, h1 = df_economic_event_select.shape
    # セルの大きさの定数
    w, h = 74, 8
    # スケジュール表を作成し保存する
    fig, ax = plt.subplots(
        figsize=(
            int(df_economic_event_select.shape[0] / 2),
            int(df_economic_event_select.shape[1] / 2),
        )
    )

    ax.axis("off")
    ax.axis("tight")

    color = np.full_like(df_economic_event_select.values, "white", dtype=object)
    idx_importance_3 = np.where(df_economic_event_select["importance"] == 3)[0]
    idx_market_holiday = np.where(df_economic_event_select["event"] == "休場")[0]
    color[idx_importance_3, :] = "yellow"
    color[idx_market_holiday, 3] = "lightskyblue"

    day_color = ["yellow", "pink", "green", "orange", "blue", "purple", "red"]
    color[:, 0] = (
        pd.to_datetime(df_economic_event_select.event_date)
        .dt.weekday.apply(lambda x: day_color[x])
        .values
    )

    table = ax.table(
        cellText=df_economic_event_select.values,
        colLabels=df_economic_event_select.columns,
        cellColours=color,
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    cellDict = table.get_celld()
    for i in range(0, len(df_economic_event_select.columns)):
        cellDict[(0, i)].set_height(0.3)
        cellDict[(0, i)].set_fontsize(15)
        if i <= 1:
            cellDict[(0, i)].set_width(0.05 * (w / w1))
        elif i == 2:
            cellDict[(0, i)].set_width(0.07 * (w / w1))
        elif i == 3:
            cellDict[(0, i)].set_width(0.15 * (w / w1))
        else:
            cellDict[(0, i)].set_width(0.05 * (w / w1))
        for j in range(1, len(df_economic_event_select) + 1):
            cellDict[(j, i)].set_height(0.2)
            cellDict[(j, 0)].set_alpha(.3)
            if i <= 1:
                cellDict[(j, i)].set_width(0.05 * (w / w1))
            elif i == 2:
                cellDict[(j, i)].set_width(0.07 * (w / w1))
            elif i == 3:
                cellDict[(j, i)].set_width(0.15 * (w / w1))
            else:
                cellDict[(j, i)].set_width(0.05 * (w / w1))

    for i in range(len(df_economic_event_select.columns)):
        table[0, i].set_facecolor("#363636")
        table[0, i].set_text_props(color="w")
    plt.savefig(IMG_PATH, bbox_inches="tight", dpi=300)

    # discordに画像送信
    with open(IMG_PATH, "rb") as f:
        img = f.read()
    dict_img = {
        "favicon": (IMG_PATH, img)
    }
    res = requests.post(WEBHOOK_ID, files=dict_img)
    print(res.status_code)
