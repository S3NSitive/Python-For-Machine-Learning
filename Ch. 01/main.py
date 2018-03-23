import sys
import urllib.parse as parse
import urllib.request as req

from bs4 import BeautifulSoup
'''
### urlretrieve 와 urlopen으로 파일 가져오기
url = "https://www.naver.com/favicon.ico"
savename = "Data/naverfavicon2.ico"

#urllib.request.urlretrieve(url, savename)
#mem = urllib.request.urlopen(url).read()

url = "http://api.aoikujira.com/ip/ini"
res = urllib.request.urlopen(url)
data = res.read()

text = data.decode('utf-8')
print(text)

### url에 매개변수를 추가해 기상청 RSS 서비스를 사용하기
if len(sys.argv) <= 1:
    print("USAGE: download-forecast-argv <Region Number>")
    sys.exit()
regionNumber = sys.argv[1]

API = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"
values = {
    'stnId': regionNumber
}
params = parse.urlencode(values)

url = API + "?" + params
print("url=", url)

data = req.urlopen(url).read()
text = data.decode("utf-8")
print(text)
'''
### BeautifulSoup 사용해보기
html = """
<html><body>
  <h1>스크레이핑이란?</h1>
  <p>웹 페이지를 분석하는 것</p>
  <p>원하는 부분을 추출하는 것</p>
</body></html>
"""

soup = BeautifulSoup(html, 'html.parser')
