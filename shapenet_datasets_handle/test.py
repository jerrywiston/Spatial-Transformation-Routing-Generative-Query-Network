import tensorflow as tf
import cv2
import numpy as np

def decode(example):
    feature_names = ['images', 'zmaps']
    feature_names.extend(['segs', 'voxel', 'obj1', 'obj2'])
        
    stuff = tf.parse_single_example(
        example,
        features={k: tf.FixedLenFeature([], tf.string) for k in feature_names}
    )

    HV = 18
    VV = 3
    Hdata = 128
    Wdata = 128

    N = VV*HV

    images = tf.decode_raw(stuff['images'], tf.float32)
    images = tf.reshape(images, (N, Hdata, Wdata, 4))

    masks = tf.slice(images, [0, 0, 0, 3], [-1, -1, -1, 1])
    images = tf.slice(images, [0, 0, 0, 0], [-1, -1, -1, 3])

    zmaps = tf.decode_raw(stuff['zmaps'], tf.float32)
    zmaps = tf.reshape(zmaps, (N, Hdata, Wdata, 1))
    rvals = [images, masks, zmaps]

    segs = tf.decode_raw(stuff['segs'], tf.float32)
    segs = tf.reshape(segs, (N, Hdata, Wdata, 2))
    voxel = tf.decode_raw(stuff['voxel'], tf.float32)
    voxel = tf.reshape(voxel, (128, 128, 128))
    obj1 = tf.decode_raw(stuff['obj1'], tf.float32)
    obj1 = tf.reshape(obj1, (128, 128, 128))
    obj2 = tf.decode_raw(stuff['obj2'], tf.float32)
    obj2 = tf.reshape(obj2, (128, 128, 128))

    rvals.extend([segs, voxel, obj1, obj2])
    return rvals

sess = tf.Session()
path = "double_tfrs/"
filenames = [path+"0", path+"1", path+"2"]
data = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
data = data.map(decode, num_parallel_calls = 1)
print(data)

iterator = data.make_one_shot_iterator()
next_element = iterator.get_next()
for i in range(100):
    test = sess.run(next_element)
    print(i,test[0].shape)
    cv2.imshow("test", test[0][0].astype(np.uint8))
    cv2.waitKey(0)
    

