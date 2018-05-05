from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("data/", one_hot=True)

# Layer定義
x = tf.placeholder(tf.float32, [None, 784])
w_1 = tf.Variable(tf.truncated_normal([784, 64], stddev=0.1), name="w1")
b_1 = tf.Variable(tf.zeros([64]), name="b1")
h_1 = tf.nn.relu(tf.matmul(x, w_1) + b_1)
w_2 = tf.Variable(tf.truncated_normal([64, 10], stddev=0.1), name="w2")
b_2 = tf.Variable(tf.zeros([10]), name="b2")
out = tf.nn.softmax(tf.matmul(h_1, w_2) + b_2)

# 誤差関数
y = tf.placeholder(tf.float32, [None, 10])
loss = tf.reduce_mean(tf.square(y - out))

# 訓練 学習率0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# 評価
correct = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# 初期化
init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep = 3)

with tf.Session() as sess:
  ckpt_state = tf.train.get_checkpoint_state('ckpt/')
  if ckpt_state:
    last_model = ckpt_state.model_checkpoint_path
    saver.restore(sess, last_model)
    print("model was loaded:", last_model)
  else:
    sess.run(init)

    # テストデータをロード
    test_images = mnist.test.images
    test_labels = mnist.test.labels

    for i in range(1000):
      step = i + 1
      train_images, train_labels = mnist.train.next_batch(50)
      sess.run(train_step, feed_dict={x:train_images, y:train_labels})

      if step % 100 == 0:
        acc_val = sess.run(accuracy, feed_dict={x:test_images, y:test_labels})
        print('Step {0:d}: accuracy = {1:.2f}'.format(step, acc_val))
        # モデルの保存
        saver.save(sess, 'ckpt/my_model', global_step = step, write_meta_graph = False)
  




