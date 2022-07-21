
import tensorflow as tf
from deepmodel.data import TFRecordLoader, Feature
from deepmodel.models import DeepFM

class Args(object):
    K = 4
    epoch = 10
    batch_size = 32
    learning_rate = 0.001
    bn_training = True

features = [
    Feature(name='id', dtype='string', dim=1, for_train=False),
    Feature(name='a', dtype='int32', dim=1, emb_count=11),
    Feature(name='b', dtype='int64', dim=1, emb_count=101),
    Feature(name='c', dtype='int64', dim=2, dense=True, emb_count=101),
    Feature(name='c_len', dtype='int64', dim=1, for_train=False),
    Feature(name='d', dtype='float32', dim=1, emb_count=1),
    Feature(name='e', dtype='float32', dim=2, dense=True, emb_count=10),
    Feature(name='label', dtype='float32', dim=1, for_train=False)
]

args = Args()

inputs = TFRecordLoader(features, 'examples/test.tfrecord', repeats=args.epoch)
batch_x = inputs.load_batch(args.batch_size)

model = DeepFM(args, features, batch_x, 0.8)
model.train()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
l1_sum, l2_sum, l3_sum, step = 0, 0, 0, 0
while True:
    try:
        l1, l2, l3, _ = sess.run([model.loss, model.log_loss, model.mse_loss, model.train_op])
        l1_sum += l1
        l2_sum += l2
        l3_sum += l3
        step += 1
        if step % 10 == 0:
            print(f'step: {step}   loss: {l1_sum}    log_loss: {l2_sum}   mse_loss: {l3_sum}')
    except:
        print("End of dataset")
        break

out_cols = ['id', 'label']
args.bn_training = False
model = DeepFM(args, features, batch_x, 1.0)
model.test(out_cols)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
output = sess.run(model.output)


