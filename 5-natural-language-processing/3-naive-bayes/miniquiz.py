import yelp.search as yelp

from pymongo import MongoClient
from pymongo import errors
from bs4 import BeautifulSoup
import requests

import time
import pdb

# Using Yelp's python client: 
# https://github.com/Yelp/yelp-api/tree/master/v2/python


client = MongoClient()
db = client.yelp
coll = db.gastro

key = "xxxx"
c_secret = "xxxx"
token = "xxxx"
t_secret = "xxxx"

def get_reviews(busi, collection):
        print "getting review for " + busi['name']
        soup = BeautifulSoup(requests.get(busi['url']).text, "html.parser")
        rev = soup.find_all('div', class_ = 'review')
        collection.update({"id" : busi['id']}, { '$set' : { 'reviews': [ { 'html' : str(item) } for item in rev ]} })
        time.sleep(1)

def parse_reviews(busi, collection):
    print "getting stars for " + busi['name']

    business = collection.find_one({"id" : busi['id'] })
    for review in business['reviews']:
        soup = BeautifulSoup(review['html'], "html.parser")
 
        if soup.find(itemprop = "reviewRating"):
            review['rating'] = soup.find(itemprop = "ratingValue")['content']
            collection.save(business)
            
if __name__ == '__main__':
    # exploratory analysis
    #
    # url_params = {'location': 'sf', 'category_filter':'gastropubs', 'limit':30}
    #
    # gastro = make_request(url_params)
    #
    # {u'error': {u'description': u'Limit maximum is 20',
    #  u'field': u'limit',
    #  u'id': u'INVALID_PARAMETER',
    #  u'text': u'One or more parameters are invalid in request'}}
    
    # get_meta(coll)

    for busi in coll.find():
        get_reviews(busi, coll)
        parse_reviews(busi, coll)
        # five_stars = filter(lambda x: x['rating'] == '5.0', coll.find_one({"id" : busi['id']})['reviews'])
