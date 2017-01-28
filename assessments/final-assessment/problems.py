import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests


def column_averages(filename):
    return pd.read_csv(filename).mean().to_dict()


def filter_by_class(X, y, label):
    return X[y == label]


def etsy_query(query):
    html = requests.get("https://www.etsy.com/search?q=%s" % query).text
    soup = BeautifulSoup(html, "html.parser")
    listings = soup.find_all("div", class_="listing-maker")
    return [x.text.strip() for x in listings]
