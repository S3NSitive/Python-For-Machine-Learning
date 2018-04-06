import tensorflow as tf
import numpy as np
import pandas as pd
'''
## 텐서플로우 기본
a = tf.constant(120, name="a")
b = tf.constant(130, name="b")
c = tf.constant(140, name="c")

v = tf.Variable(0, name="v")

calc_op = a + b + c
assign_op = tf.assign(v, calc_op)

sess = tf.Session()
sess.run(assign_op)

print(sess.run(v))

### 플레이스 홀더
a = tf.placeholder(tf.int32, [None])

b = tf.constant(2)
x2_op = a * b

sess = tf.Session()

r1 = sess.run(x2_op, feed_dict={a: [1, 2, 3, 4, 5]})
print(r1)
r2 = sess.run(x2_op, feed_dict={a: [10]})
print(r2)

### 머신러닝 해보기
csv = pd.read_csv("Data/bmi.csv")

csv["height"] = csv["height"] / 200
csv["weight"] = csv["weight"] / 100

# - thin=(1, 0, 0) / normal=(0, 1, 0) / fat=(0, 0, 1)
bclass = {"thin": [1, 0, 0], "normal": [0, 1, 0], "fat": [0, 0, 1]}
csv["label_pat"] = csv["label"].apply(lambda x : np.array(bclass[x]))

test_csv = csv[15000:20000]
test_pat = test_csv[["weight", "height"]]
test_ans = list(test_csv["label_pat"])

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 3])

W = tf.Variable(tf.zeros([2, 3]))
b = tf.Variable(tf.zeros([3]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3500):
    i = (step * 100) % 14000
    rows = csv[1 + i : 1 + i + 100]
    x_pat = rows[["weight", "height"]]
    y_ans = list(rows["label_pat"])
    fd = {x: x_pat, y_: y_ans}
    sess.run(train, feed_dict=fd)
    if step % 500 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
        print("step =", step, "cre =", cre, "acc =", acc)

acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})
print("정답률 =", acc)
'''
### 곱셈하는 코드 시각화 하기
a = tf.constant(20, name="a")
b = tf.constant(30, name="b")
mul_op = a * b

sess = tf.Session()

tw = tf.summary.FileWriter("log_dir", graph=sess.graph)

print(sess.run(mul_op))