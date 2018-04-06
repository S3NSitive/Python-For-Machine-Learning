import urllib.request as req
import os.path
import json

from bs4 import BeautifulSoup
'''
### XML 분석하기
url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stnId=108"
savename = "Data/forecast.xml"
if not os.path.exists(savename):
    req.urlretrieve(url, savename)

xml = open(savename, "r", encoding="utf-8").read()
soup = BeautifulSoup(xml, "html.parser")

info = {}
for location in soup.find_all("location"):
    name = location.find('city').string
    weather = location.find('wf').string
    if not (weather in info):
        info[weather] = []
    info[weather].append(name)

for weather in info.keys():
    print("+", weather)
    for name in info[weather]:
        print("| - ", name)
'''
### JSON 분석하기
url = "https://api.github.com/repositories"
savename = "Data/repo.json"
if not os.path.exists(url):
    req.urlretrieve(url, savename)

items = json.load(open(savename, "r", encoding="utf-8"))

# s = open(savename, "r", encoding="utf-8").read()
# items = join.loads(s)

for item in items:
    print(item["name"] + " - " + item["owner"]["login"])