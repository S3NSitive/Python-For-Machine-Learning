import sys
#import urllib.parse as parse
#import urllib.request as req
import os.path, time, re

from bs4 import BeautifulSoup
from urllib.parse import *
from urllib.request import *
from os import makedirs
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

### BeautifulSoup 사용해보기
html = """
<html><body>
  <h1 id="title">스크레이핑이란?</h1>
  <p id="body">웹 페이지를 분석하는 것</p>
  <p>원하는 부분을 추출하는 것</p>
  <ul>
    <li><a href="http://www.naver.com">naver</a></li>
    <li><a href="http://www.daum.net">daum</a></li>
  </ul>
</body></html>
"""

soup = BeautifulSoup(html, 'html.parser')

title = soup.find(id="title")
body = soup.find(id="body")
p = soup.html.body.p.next_sibling.next_sibling
links = soup.find_all("a")

print(soup.prettify())
print("#title = " + title.string)
print("#body = " + body.string)
print("p = " + p.string)

for a in links:
    print(type(a.attrs))
    href = a.attrs['href']
    text = a.string
    print(text, ">", href)
    
### urlopen()과 BeautifulSoup 조합

url = "http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp"

res = req.urlopen(url)

soup = BeautifulSoup(res, "html.parser")
print(soup)

title = soup.find("title").string
wf = soup.find("wf").string
print(title)
print(wf)

## CSS 선택자 사용하기
html = """
<html><body>
<div id="meigen">
  <h1>위키북스 도서</h1>
  <ul class="items">
    <li>유니티 게임 이펙트 입문</li>
    <li>스위프트로 시작하는 아이폰 앱 개발 교과서</li>
    <li>모던 웹사이트 디자인의 정석</li>
  </ul>
</div>
</body></html>
"""

soup = BeautifulSoup(html, "html.parser")

h1 = soup.select_one("div#meigen > h1").string
print("h1 =", h1)

li_list = soup.select("div#meigen > ul.items > li")
for li in li_list:
    print("li =", li.string)

### 네이버 금융에서 환율 정보 추출하기
url = "http://finance.naver.com/marketindex/"
res = req.urlopen(url)

soup = BeautifulSoup(res, "html.parser")

price = soup.select_one("#exchangeList > li.on > a.head.usd > div > span.value").string
print("usd/krw =", price)

### 정규 표현식과 함께 조합하기
html = """
<ul>
  <li><a href="hoge.html">hoge</li>
  <li><a href="https://example.com/fuga">fuga*</li>
  <li><a href="https://example.com/foo">foo*</li>
  <li><a href="http://example.com/aaa">aaa</li>
</ul>
"""

soup = BeautifulSoup(html, "html.parser")

li = soup.find_all(href=re.compile(r"^https://"))
for e in li: print(e.attrs['href'])

### 링크에 있는 것을 한꺼번에 내려받기
### 상대 경로를 전개하는 방법
base = "http://example.com/html/a.html"

print(urljoin(base, "b.html"))
print(urljoin(base, "sub/c.html"))
print(urljoin(base, "../index.html"))
print(urljoin(base, "../img/hoge.png"))
print(urljoin(base, "../css/hoge.css"))
print(urljoin(base, "/hoge.html"))
print(urljoin(base, "http://otherExample.com/wiki"))
print(urljoin(base, "//anotherExample.org/test"))
'''
### 재귀적으로 모든 페이지를 한꺼번에 다운받는 프로그램
# 이미 처리한 파일인지 확인하기 위한 변수
proc_files = {}

# HTML 내부에 있는 링크를 추출하는 함수
def enum_links(html, base):
    soup = BeautifulSoup(html, "html.parser")
    links = soup.select("link[rel='stylesheet']")
    links += soup.select("a[href]") # 링크
    result = []
    # href 속성을 추출하고, 링크를 절대 경로로 변환
    for a in links:
        href = a.attrs['href']
        url = urljoin(base, href)
        result.append(url)

    return result

# 파일을 다운받고 저장하는 함수
def download_file(url):
    o = urlparse(url)
    savepath = "./" + o.netloc + o.path
    if re.search(r"/$", savepath): # 폴더라면 index.html
        savepath += "index.html"
    savedir = os.path.dirname(savepath)
    # 모두 다운됬는지 확인
    if os.path.exists(savepath): return savepath
    # 다운받을 폴더 생성
    if not os.path.exists(savedir):
        print("mkdir=", savedir)
        makedirs(savedir)
    # 파일 다운받기
    try:
        print("download=", url)
        urlretrieve(url, savepath)
        time.sleep(1) # 1초 휴식
        return savepath
    except:
        print("다운 실패: ", url)
        return None

# HTML을 분석하고 다운받는 함수
def analyze_html(url, root_url):
    savepath = download_file(url)
    if savepath is None: return
    if savepath in proc_files: return # 이미 처리됐다면 실행하지 않음

    proc_files[savepath] = True
    print("analyze_html=", url)
    # 링크 추출
    html = open(savepath, "r", encoding="utf-8").read()
    links = enum_links(html, url)

    for link_url in links:
        # 링크가 루트 이외의 경로를 나타낸다면 무시
        if link_url.find(root_url) != 0:
            if not re.search(r".css$", link_url): continue
        # HTML 파일 이라면
        if re.search(r".(html|htm)$", link_url):
            #재귀적으로 HTML 파일 분석하기
            analyze_html(link_url, root_url)
            continue
        # 기타 파일
        download_file(link_url)

if __name__ == "__main__":
    # URL에 있는 모든 것 다운받기
    url = "https://docs.python.org/3.5/library/"
    analyze_html(url, url)