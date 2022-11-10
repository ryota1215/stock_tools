import pandas as pd
import requests
import json
import sys

sys.path.append(
    "c:\\Users\\xxp2p\\anaconda3\\envs\\yt_38\\lib\\site-packages\\stock_tools-0.0.1.dev34+ge1c79e5-py3.8.egg\\stock_tools\\"
)
import add_technical

# リフレッシュトークン1週間に一度は取得。
REFRESH_TOKEN = "eyJjdHkiOiJKV1QiLCJlbmMiOiJBMjU2R0NNIiwiYWxnIjoiUlNBLU9BRVAifQ.CH6yYnbF4pJPBdVa8Sdaw6LqDe3s0hUhD7VYM4HSyi0jfjZYS1wAo0NYLnh7La60njW8dKfKEDoxJTtQPY8szDtdpOb7XfbuT8Q7JB-BI-K5Xm3ANZOIy1eUsvpKw2u29-vIYP44FXFYHtzKdcb5rVo4_XyeE8H-OJD8__8OSd06_PDdv7wfqB791hqmVkAzpOtO0tsaaXQFhn2vsq3TdIM5cN4xaFLYmcNJ-Ywaz35fN6M6Gci5RpxwRyPvSvR6bg_s3P17fSb878OLWaJopqWWnLRnBVJa3aV91ODAZTKgJYH-FSuP-xIy626X-lTjHTbuq_rXFgVDE3lnwzN8dg.7eLM0y1QlwrZ1tVg.l2UhKarue6Lup-uWWJH7rBF5MSkJ3g8eAAYjT3MkKwXdzGVKQPsuJwu0RAbQn6xWMRt8sJKA5-qWL_nFw2DM4Ju6jqEVWvs46afosaU4DXVedZPirhfsZtr5k6-Wmv1qGwV7OakxNqN5ESxL4ITkwg8FUNJ2Itw-_rPZv2aQlVgDZo6KpDBcnqTTYDs0IcDmNr8j_l_pVAx6w0NrEgfwnNGGe2g_TmeBigBbUPwNX7_EZ40D1u44ABzMxUmiosFlmOr7Cyd0gcIxohLrPxBBlQs82lc3x5GJDhYpXVU4UhOmt9Z2TFffEQqpH2e0e_c-ybfRgZIfIMIHCZ-AVKeMwtBbpPRxmUezy99iBlcY_Ui73QLIwvSdz1pVpxTAMRVqRcbbNrZiXrwpDWdEngBEHfG8UaewchsA9tcv3btHUFA1wmr-A_j3erUy4G-WK-mP9BfhiGkfJsm8d4LXGXiMMD0yja0x3jpZ1CWAh85GK2mNt7EvQKyS2Z99Hdb5T6lo4lv7hjd_8VROOmR8aQ8CDvrxvPn5aPDEjnjgsajHHMo9Pc52cWAL9jhejNDoH03nh94jq8r93eG_TWttFM9D7wAP08gXe1NJLVNTl9ISQB_u_PEre3rKNDrym_byVHPyyKVeL1Lm1P1F-Nki17q7cKnIE-uBsSf-yPzL99WoZbwIDrwjq_ljSHJ0BgwZuXXeMJCiEhkrybN3TaWEcT4WBhpzzjBrWFQdvN3RCI_cFAavJTULB1ngfNzPdG9eW5gDfdxvF1FK0QbzelonP4nhE-XqA2p4PrUkflnI3795O61PIwq89NuglmW1BuvEpjEVJPUERSBfX_nmDxuGaVSUy4Alr2I3ORBcuto350ccZWdD28-ROf5U3W_30sIhTKzFJVCpSj6o6rk9zFu1t_9jcP364kJna6S-v5jwlfQBDI4UXEIzR011sll8VvkA_RO_KFiaGSmegQBzdI7Sp1JxRfAFTkbtFWYewzWVj0Zo8ofedOMH8P_xf3NcO0xvjLGcEV4JrUv5jxkB-53zSYSNVr6vTArTqc6xMny0-XwMqZHANHjpKsVllorGoVfyGc-IuXJdznqjr_rIJi1S9ue4lqJkOu_DPX5e-SZ8LP8PN0atz0MT5BtXsSzqkHPQLeXiPdiQ6Lw-vt_J7CmVRmGvodXZCbCWMWoXanv0AgqOvQRFU1M2jVOa4lm1JisGCct1BF0C5gz3yiVHFWQ19-80fA8OVOJlFOT19xYlWM_s3KF443zr5RS1LBQCYMPDvr6oSgGS36UtmFWzrdReH7ui1gNwwWwiDlvcTZbHiu2cUxCAU4JipJnzcNBY37Kr47yDN4vbpu9DGSOxxQ.jXdUCJyN6sohQCkA4H03kQ"


def jpx_df(REFRESH_TOKEN):
    r_post = requests.post(
        f"https://api.jpx-jquants.com/v1/token/auth_refresh?refreshtoken={REFRESH_TOKEN}"
    )
    idToken = r_post.json()["idToken"]
    headers = {"Authorization": "Bearer {}".format(idToken)}
    r = requests.get(
        "https://api.jpx-jquants.com/v1/prices/daily_quotes?code=86970", headers=headers
    )
    df = pd.DataFrame(r.json()["daily_quotes"])
    return df


df = jpx_df(REFRESH_TOKEN)


def test_1():
    df_add = add_technical.add_technical(df, True)
    print(df_add.df)


test_1()
