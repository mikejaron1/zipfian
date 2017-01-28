from pymongo import MongoClient
import requests
import bs4 as soup

baseurl = 'http://www.snackdata.com/'
client = MongoClient('mongodb://localhost:27017/')
db = client.snackdata
collection = db.snacks
r = requests.get(baseurl)
s = soup.BeautifulSoup(r.text, 'html.parser')
for link in s.ol.findAll('a'):
    l = link.get('href')
    rl = requests.get(baseurl+l)
    sl = soup.BeautifulSoup(rl.text)
    content1 = sl.find("div", attrs={"id": "rightstuff"})
    content2 = sl.find("div", attrs={"id": "middlestuff"})

    d = {}
    d['name'] = content1.h1.text
    d['number of snack'] = int(sl.h4.span.em.text)
    for i in content2.findAll("dt")[:3]:
        # flavor, cuisine, series
        d[i.text] = i.find_next_sibling().text
    comps = content2.find("dd", attrs={"class":"clearfix notdropped"})
    complist = [a.get('href').split('/')[3] for a in comps.findAll("a")]
    d['Composition'] = complist # TODO: link to other db records

    descs = content1.contents[5].text.split('\n')
    d['Date'] = descs[3].split('snack on ')[1].strip('.')
    d['Description'] = descs[0]
    d['Taste Description'] = descs[1].strip('Taste')

    if collection.find({'name':d['name']}) == None:
        collection.insert(d)