import requests
import json

from bs4 import BeautifulSoup
from urllib.parse import urljoin
from selenium import webdriver
'''
### 파이썬 requests 모듈을 사용한 로그인
USER = "admin"
PASSWD = "privacy"

session = requests.session()

login_info = {
    "mb_id": USER,
    "mb_password": PASSWD
}

url_login = "http://192.168.100.193/bbs/login_check.php"
res = session.post(url_login, data=login_info)
res.raise_for_status()

url_adminpage = "http://192.168.100.193/adm/"
res = session.get(url_adminpage)
res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser")
name = soup.select_one("td:nth-of-type(2)").get_text()
num = soup.select_one("td.td_num").get_text()
point = soup.select_one("td > a[href]").get_text()
print("이름 : ", name)
print("권한 : ", num)
print("포인트 : ", point)
### requests 모듈의 메서드
r = requests.get("http://api.aoikujira.com/time/get.php")

text = r.text
print(text)

bin = r.content
print(bin)

options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument('disable-gpu')

driver = webdriver.Chrome('Driver/chromedriver.exe', chrome_options=options)

driver.get("https://nid.naver.com/nidlogin.login")
#driver.get("https://www.naver.com")
driver.implicitly_wait(3)

driver.find_element_by_id("id").send_keys("doble0309")
driver.find_element_by_id("pw").send_keys("privacy!@34")

driver.find_element_by_xpath('//*[@id="frmNIDLogin"]/fieldset/input').click()

driver.get_screenshot_as_file("Data/Naver_main_headless.png")

driver.quit()
'''
### 웹 API 사용해보기
apikey = "ed60c0caac36257f61810ec105c15bcf"

cities = ["Seoul", "Tokyo", "New York"]

api = "http://api.openweathermap.org/data/2.5/weather?q={city}&APPID={key}"

k2c = lambda k: k - 273.15

for name in cities:
    url = api.format(city=name, key=apikey)
    r = requests.get(url)

    data = json.loads(r.text)
    print("+ 도시 =", data["name"])
    print("| 날씨 =", data["weather"][0]["description"])
    print("| 최저 기온 =", k2c(data["main"]["temp_min"]))
    print("| 최고 기온 =", k2c(data["main"]["temp_max"]))
