import pymongo
import requests
from bs4 import BeautifulSoup

cli = pymongo.MongoClient()
db = cli.snacks
coll = db.snack_data

index = 'http://www.snackdata.com'

res = requests.get(index)
soup = BeautifulSoup(res.content, 'html.parser')

ind = soup.findAll(id='indexholder')
items = ind[0].findAll('li')

for item in items:
    path = item.find('a')['href']
    url = index + path

    res = requests.get(url)
    parser_snack(res, url)
    
def parse_snack(res, url):
    soup = BeautifulSoup(res.content, 'html.parser')
    data = soup.find('div', class_='data')
    
    # parse table
    tables = zip(data.findAll('dt'), data.findAll('dd'))
    data_dict = dict([ (tab[0].text, map(lambda x: x.strip(), tab[1].text.strip().split(','))) for tab in tables ])

    data_dict['_id']

    # parse header
    header = data.find('div', class_='header')
    data_dict['number'] = header.find('em').text

    # extract irrelevant info
    header.find('h4').find('span').extract()
    data_dict['name'] = header.find('h4').text

    # add composition links
    composition = data.findAll('dd')[-1]

    data_dict['composition'] = [ link['href'] for link in composition.findAll('a')]

    # parse right cell
    right = soup.find(id='rightstuff')
    right.find('h1').extract()

    division = right.text.split('\nTaste')
    data_dict['description'] = division[0].strip()

    second = division[1].split('\n\n')
    data_dict['taste'] = second[0].strip()
    data_dict['date'] = second[1].split('on')[-1]

    return data_dict
