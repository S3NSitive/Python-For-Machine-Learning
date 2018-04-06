import pandas as pd
import urllib.request as req
import matplotlib.pyplot as plt
import glob, gzip, os, os.path, re, json, random, struct

from sklearn import svm, metrics, model_selection
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
'''
### XOR 연산
xor_input = [
    #P, Q, result
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
]

xor_df = pd.DataFrame(xor_input)
xor_data = xor_df.ix[:,0:1]
xor_label = xor_df.ix[:,2]

clf = svm.SVC()
clf.fit(xor_data, xor_label)
pre = clf.predict(xor_data)

ac_score = metrics.accuracy_score(xor_label, pre)
print("정답률 = ", ac_score*100)

### 붓꽃 품종 분류하기
csv = []
with open('Data/iris.csv', 'r', encoding='utf-8') as fp:
    for line in fp:
        line = line.strip()
        cols = line.split(',')
        
        fn = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
        cols = list(map(fn, cols))
        
        csv.append(cols)

del csv[0]

random.shuffle(csv)

total_len = len(csv)
train_len = int(total_len * 2 / 3)
train_data = []
train_label = []
test_data = []
test_label = []

for i in range(total_len):
    data = csv[i][0:4]
    label = csv[i][4]
    if i < train_len:
        train_data.append(data)
        train_label.append(label)
    else:
        test_data.append(data)
        test_label.append(label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print("정답률 = ", ac_score*100)

### 데이터 분류하여 붓꽃 분류하기
csv = pd.read_csv('Data/iris.csv')

csv_data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
csv_label = csv["Name"]

train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

ac_score = metrics.accuracy_score(test_label, pre)
print("정답률 = ", ac_score*100)

### MNIST 파일 다운로드
savepath = "Data"
baseurl = "http://yann.lecun.com/exdb/mnist/"
files = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz"
]

if not os.path.exists(savepath):
    os.mkdir(savepath)

for f in files:
    url = baseurl + "/" + f
    loc = savepath + "/" + f
    print("download : ", url)
    if not os.path.exists(loc):
        req.urlretrieve(url, loc)

for f in files:
    gz_file = savepath + "/" + f
    raw_file = savepath + "/" + f.replace(".gz", "")
    print("gzip : ", f)
    with gzip.open(gz_file, "rb") as fp:
        body = fp.read()
        with open(raw_file, "wb") as w:
            w.write(body)

print("ok")

### MNIST 파일 CSV로
def to_csv(name, maxdata):
    lbl_f = open("Data/" + name + "-labels-idx1-ubyte", "rb")
    img_f = open("Data/" + name + "-images-idx3-ubyte", "rb")
    csv_f = open("Data/" + name + ".csv", "w", encoding="utf-8")

    mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
    mag, img_count = struct.unpack(">II", img_f.read(8))
    rows, cols = struct.unpack(">II", img_f.read(8))
    pixels = rows * cols

    res = []
    for idx in range(lbl_count):
        if idx > maxdata: break
        label = struct.unpack("B", lbl_f.read(1))[0]
        bdata = img_f.read(pixels)
        sdata = list(map(lambda n: str(n), bdata))
        csv_f.write(str(label) + ",")
        csv_f.write(",".join(sdata) + "\r\n")

        if idx < 10:
            s = "P2 28 28 255\n"
            s += " ".join(sdata)
            iname = "Data/{0}-{1}-{2}.pgm".format(name, idx, label)
            with open(iname, "w", encoding="utf-8") as f:
                f.write(s)

    csv_f.close()
    lbl_f.close()
    img_f.close()

to_csv("train", 99999)
to_csv("t10k", 99999)

### 이미지 데이터 학습시키기
def load_csv(fname):
    labels = []
    images = []
    with open(fname, "r") as f:
        for line in f:
            cols = line.split(",")
            if len(cols) < 2: continue
            labels.append(int(cols.pop(0)))
            vals = list(map(lambda n: int(n) / 256, cols))
            images.append(vals)
    return {"labels":labels, "images":images}

data = load_csv("Data/train.csv")
test = load_csv("Data/t10k.csv")

clf = svm.SVC()
clf.fit(data["images"], data["labels"])

predict = clf.predict(test["images"])

ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("정답률 = ", ac_score)
print("리포트 = ")
print(cl_report)

### 외국어 문장 판별하기
def check_freq(fname):
    name = os.path.basename(fname)
    lang = re.match(r'^[a-z]{2,}', name).group()
    with open(fname, "r", encoding="utf-8") as f:
        text = f.read()
    text = text.lower()

    cnt = [0 for n in range(0, 26)]
    code_a = ord("a")
    code_z = ord("z")

    for ch in text:
        n = ord(ch)
        if code_a <= n <= code_z:
            cnt[n - code_a] += 1

    total = sum(cnt)
    freq = list(map(lambda n: n / total, cnt))
    return (freq, lang)

def load_files(path):
    freqs = []
    labels = []
    file_list = glob.glob(path)
    for fname in file_list:
        r = check_freq(fname)
        freqs.append(r[0])
        labels.append(r[1])
    return {"freqs":freqs, "labels":labels}

data = load_files("Data/lang/train/*.txt")
test = load_files("Data/lang/test/*.txt")

with open("Data/lang/freq.json", "w", encoding="utf-8") as fp:
    json.dump([data, test], fp)

clf = svm.SVC()
clf.fit(data["freqs"], data["labels"])

predict = clf.predict(test["freqs"])

ac_score = metrics.accuracy_score(test["labels"], predict)
cl_report = metrics.classification_report(test["labels"], predict)
print("정답률 = ", ac_score)
print("리포트 = ")
print(cl_report)

### 데이터 분포를 그래프로 확인하기
with open("Data/lang/freq.json", "r", encoding="utf-8") as fp:
    freq = json.load(fp)

lang_dic = {}
for i, lbl in enumerate(freq[0]["labels"]):
    fq = freq[0]["freqs"][i]
    if not (lbl in lang_dic):
        lang_dic[lbl] = fq
        continue
    for idx, v in enumerate(fq):
        lang_dic[lbl][idx] = (lang_dic[lbl][idx] + v) / 2

asclist = [[chr(n) for n in range(97, 97+26)]]
df = pd.DataFrame(lang_dic, index=asclist)

plt.style.use('ggplot')
df.plot(kind="bar", subplots=True, ylim=(0, 0.15))
plt.savefig("Data/lang-plot.png")

### 학습한 매개변수를 저장하기
with open("Data/lang/freq.json", "r", encoding="utf-8") as fp:
    d = json.load(fp)
    data = d[0]

clf = svm.SVC()
clf.fit(data["freqs"], data["labels"])

joblib.dump(clf, "Data/lang/freq.pkl")

### SVM을 사용해보기
def calc_bmi(h, w):
    bmi = w / (h/100) ** 2
    if bmi < 18.5: return "thin"
    if bmi < 25: return "normal"
    return "fat"

fp = open("Data/bmi.csv", "w", encoding="utf-8")
fp.write("height,weight,label\r\n")

cnt = {"thin": 0, "normal": 0, "fat": 0}
for i in range(20000):
    h = random.randint(120, 200)
    w = random.randint(35, 80)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write("{0},{1},{2}\r\n".format(h, w, label))

fp.close()

tbl = pd.read_csv("Data/bmi.csv")

label = tbl["label"]
w = tbl["weight"] / 100
h = tbl["height"] / 200
wh = pd.concat([w, h], axis=1)

data_train, data_test, label_train, label_test = train_test_split(wh, label)

clf = svm.SVC()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답률 = ", ac_score)
print("리포트 = \n", cl_report)

local = "Data/mushroom.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
req.urlretrieve(url, local)

### 랜덤 포레스트 사용해보기
mr = pd.read_csv("Data/mushroom.csv", header=None)

label = []
data = []
attr_list = []

for row_index, row in mr.iterrows():
    label.append(row.ix[0])
    row_data = []
    for v in row.ix[1:]:
        row_data.append(ord(v))
    data.append(row_data)

for row_index, row in mr.iterrows():
    label.append(row.ix[0])
    exdata = []
    for col, v in enumerate(row.ix[1:]):
        if row_index == 0:
            attr = {"dic": {}, "cnt": 0}
            attr_list.append(attr)
        else:
            attr = attr_list[col]
        d = [0,0,0,0,0,0,0,0,0,0,0,0]
        if v in attr["dic"]:
            idx = attr["dic"][v]
        else:
            idx = attr["cnt"]
            attr["dic"][v] = idx
            attr["cnt"] += 1
        d[idx] = 1
        exdata += d
    data.append(exdata)
data_train, data_test, label_train, label_test = train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(data_train, label_train)

predict = clf.predict(data_test)

ac_score = metrics.accuracy_score(label_test, predict)
cl_report = metrics.classification_report(label_test, predict)
print("정답률 =", ac_score)
print("리포트 =\n", cl_report)

### 크로스 밸리데이션
lines = open("Data/iris.csv", encoding="utf-8").read().split('\n')
f_tonum = lambda n : float(n) if re.match(r'^[0-9\.]+$', n) else n
f_cols = lambda li : list(map(f_tonum, li.strip().split(',')))
csv = list(map(f_cols, lines))
del csv[0]
random.shuffle(csv)

K = 5
csvk = [ [] for i in range(K) ]
for i in range(len(csv)):
    csvk[i % K].append(csv[i])

def split_data_label(rows):
    data = []; label = []
    for row in rows:
        data.append(row[0:4])
        label.append(row[4])
    return (data, label)

def calc_score(test, train):
    test_f, test_l = split_data_label(test)
    train_f, train_l = split_data_label(train)

    clf = svm.SVC()
    clf.fit(train_f, train_l)
    pre = clf.predict(test_f)
    return metrics.accuracy_score(test_l, pre)

score_list = []
for testc in csvk:
    trainc = []
    for i in csvk:
        if i != testc: trainc += i
    sc = calc_score(testc, trainc)
    score_list.append(sc)
print("각각의 정답률 = ", score_list)
print("평균 정답률 = ", sum(score_list) / len(score_list))

### scikit-learn 크로스 밸리데이션
csv = pd.read_csv("Data/iris.csv")

data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
label = csv["Name"]

clf = svm.SVC()
scores = model_selection.cross_val_score(clf, data, label, cv=5)
print("각각의 정답률 = ", scores)
print("평균 정답률 = ", scores.mean())
'''
### 그리드 서치
train_csv = pd.read_csv("Data/train.csv")
test_csv = pd.read_csv("Data/t10k.csv")

train_label = train_csv.ix[:, 0]
train_data = train_csv.ix[:, 1:577]
test_label = test_csv.ix[:, 0]
test_data = test_csv.ix[:, 1:577]
print("학습 데이터의 수 =", len(train_label))

params = [
    {"C": [1,10,100,1000], "kernel":["linear"]},
    {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
]

clf = GridSearchCV(svm.SVC(), params, n_jobs=-1)
clf.fit(train_data, train_label)
print("학습기 =", clf.best_estimator_)

pre = clf.predict(test_data)
ac_score = metrics.accuracy_score(pre, test_label)
print("정답률 =", ac_score)