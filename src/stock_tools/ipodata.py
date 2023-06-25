# 初期設定
import os
import pandas as pd
import numpy as np
import time
import datetime
import lxml
import re
import requests
from bs4 import BeautifulSoup as bs4
import tqdm
import sys
import inspect

# chromeのwebdriver自動更新用
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 自然言語処理　文章の名詞解析用
import spacy  # ja-ginzaモデルもインストール必要

# pip install ja-ginza
# https://qiita.com/wf-yamaday/items/3ffdcc15a5878b279d61
# https://yu-nix.com/archives/spacy-pos-list/

# bs4参考
# https://web-kiwami.com/python-beautyfulsoup4.html
# http://kondou.com/BS4/
import warnings

warnings.simplefilter("ignore")


def ipodata(savepath=None):
    """
    IPOのデータ取得
    """


if __name__ == "__main__":
    pass
