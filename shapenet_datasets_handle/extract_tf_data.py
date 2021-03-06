import tensorflow as tf
import cv2
import os
import numpy as np

def decode(example):
    feature_names = ['images', 'zmaps']
    #feature_names.extend(['segs', 'voxel', 'obj1', 'obj2'])
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
    '''
    segs = tf.decode_raw(stuff['segs'], tf.float32)
    segs = tf.reshape(segs, (N, Hdata, Wdata, 2))

    voxel = tf.decode_raw(stuff['voxel'], tf.float32)
    voxel = tf.reshape(voxel, (128, 128, 128))

    obj1 = tf.decode_raw(stuff['obj1'], tf.float32)
    obj1 = tf.reshape(obj1, (128, 128, 128))
    
    obj2 = tf.decode_raw(stuff['obj2'], tf.float32)
    obj2 = tf.reshape(obj2, (128, 128, 128))

    rvals.extend([segs, voxel, obj1, obj2])
    '''
    return rvals

sess = tf.Session()

def extract_data(path, sess, size=332):
    filenames = [os.path.join(path,fn) for fn in os.listdir(path)]
    data = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
    data = data.map(decode, num_parallel_calls = 1)
    print(data)

    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()
    rec = []
    for i in range(size):
        test = sess.run(next_element)
        print(i,test[0].shape)
        if i < 300:
            rec.append(test[0][np.newaxis,...].astype(np.uint8))
        else:
            rec.append(test[0][np.newaxis,...].astype(np.uint8))
    
    rec = np.concatenate(rec, 0)
    return rec

train_test_data = extract_data("double_tfrs/", sess, 332)
four_obj_data = extract_data("4_tfrs/", sess, 139)
arith_data = extract_data("arith_tfrs/", sess, 40)
multi_data = extract_data("multi_tfrs/", sess, 500)
print(train_test_data.shape, four_obj_data.shape, arith_data.shape, multi_data.shape)

np.savez("shapenet_data.npz", 
    train=train_test_data[:300], 
    test=train_test_data[300:],
    obj4=four_obj_data,
    arith=arith_data,
    multi=multi_data
    )