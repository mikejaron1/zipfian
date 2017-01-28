import yelp.search as yelp

from pymongo import MongoClient
from pymongo import errors
from bs4 import BeautifulSoup
import requests

import time
import pdb

'''
Using Yelp's python client:

https://github.com/Yelp/yelp-api/tree/master/v2/python

'''

client = MongoClient()
db = client.yelp
coll = db.gastro

key = "ZuHsCMxL2_YeMqVooEbhug"
c_secret = "mybzEwJOT42x8TfKZB-Z8nSObcg"
token = "fbOkIQ5wqnzbv1jR0FHU8Zq4MQnhaxEO"
t_secret = "0-e4mIa1Awoh7z_w9naAMW6D-Qs"

def make_request(params):
    return yelp.request(params, key, c_secret, token, t_secret)

def insert_business(busi, collection):
        if not collection.find_one({"id" : busi['id']}):
            try:
                print "Inserting restaurant " + str(busi['name'])
                collection.insert(busi)
            except errors.DuplicateKeyError:
                print "Duplicates"
        else:
            print "In collection already"

def get_meta(collection):
    url_params = {'location': 'sf', 'category_filter':'gastropubs'}

    url_params['limit'] = 20
    url_params['offset'] = 0

    total_num = gastro = make_request(url_params)['total']
    total_results = collection.find().count()

    while total_results < total_num and url_params['offset'] < total_num:
        response = make_request(url_params)

        for business in response['businesses']:
            insert_business(business, collection)

        url_params['offset'] += 20
        time.sleep(1)

if __name__ == '__main__':
    get_meta(coll)
