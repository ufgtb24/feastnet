import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
tf.enable_eager_execution(config=config) #1.x


class Net(tf.keras.Model):
    """A simple linear model."""
    
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = tf.keras.layers.Dense(5)
    
    def call(self, x):
        return self.l1(x)


def toy_dataset():
    inputs = tf.range(10.)[:, None]
    labels = inputs * 5. + tf.range(5.)[None, :]
    return tf.data.Dataset.from_tensor_slices(
        dict(x=inputs, y=labels)).repeat(10).batch(2)


def train_step(net, example, optimizer):
    """Trains `net` on `example` using `optimizer`."""
    with tf.GradientTape() as tape:
        output = net(example['x'])
        loss = tf.reduce_mean(tf.abs(output - example['y']))
    variables = net.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss


opt = tf.train.AdamOptimizer(0.1)
net = Net()
ckpt = tf.train.Checkpoint(optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)
print(tf.train.list_variables(tf.train.latest_checkpoint('./tf_ckpts')))

ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")
iter=0
for example in toy_dataset():
    loss = train_step(net, example, opt)
    # ckpt.step.assign_add(1)
    if iter % 10 == 0:
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(iter, save_path))
        print("loss {:1.2f}".format(loss.numpy()))
    iter+=1
