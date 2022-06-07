import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import os
import functools
import numpy as np
import time
from tqdm import tqdm
from tensorflow.python.ops import array_ops
import tensorflow_gan as tfgan2
tfgan = tfgan2
import sys
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session=tf.compat.v1.Session(config=config)

# A smaller BATCH_SIZE reduces GPU memory usage, but at the cost of a slight slowdown
BATCH_SIZE = 64
INCEPTION_URL = 'http://download.tensorflow.org/models/frozen_inception_v1_2015_12_05.tar.gz'
INCEPTION_FROZEN_GRAPH = 'inceptionv1_for_inception_score.pb'
def get_graph_def_from_url_tarball(url, filename, tar_filename=None):
  """Get a GraphDef proto from a tarball on the web.
  Args:
    url: Web address of tarball
    filename: Filename of graph definition within tarball
    tar_filename: Temporary download filename (None = always download)
  Returns:
    A GraphDef loaded from a file in the downloaded tarball.
  """
  if not (tar_filename and os.path.exists(tar_filename)):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (url,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    tar_filename, _ = urllib.request.urlretrieve(url, tar_filename, _progress)
  with tarfile.open(tar_filename, 'r:gz') as tar:
    proto_str = tar.extractfile(filename).read()
  return graph_pb2.GraphDef.FromString(proto_str)
# Run images through Inception.
inception_images = tf.compat.v1.placeholder(tf.float32, [None, 3, None, None])
def inception_logits(images = inception_images, num_splits = 1):
    pass
    return 0

logits=inception_logits()

def get_inception_probs(inps):
    n_batches = int(np.ceil(float(inps.shape[0]) / BATCH_SIZE))
    preds = np.zeros([inps.shape[0], 1000], dtype = np.float32)
    for i in tqdm(range(n_batches)):
        inp = inps[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        assert(np.min(inp[0]) <= 1 and np.max(inp[0]) >= -1), 'Image values should be in the range [-1, 1]'
        preds[i * BATCH_SIZE : i * BATCH_SIZE + min(BATCH_SIZE, inp.shape[0])] = session.run(logits,{inception_images: inp})[:, :1000]
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    return preds

def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)

def get_inception_score(images, splits=10):
    assert(type(images) == np.ndarray)
    assert(len(images.shape) == 4)
    assert(images.shape[1] == 3)

    print('Calculating Inception Score with %i images in %i splits' % (images.shape[0], splits))
    start_time=time.time()
    preds = get_inception_probs(images)
    mean, std = preds2score(preds, splits)
    print('Inception Score calculation time: %f s' % (time.time() - start_time))
    return mean, std