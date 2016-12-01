import json
import os
import sys
import argparse
import tensorflow as tf

FLAGS = None

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    

def doJSONtoTF(filename, name):
    with open(filename, 'r') as f:
        data = json.loads(f.read())
        #print data
        cwd = os.getcwd()
        tfdir = os.path.join(cwd, "tfdata/")
        if not os.path.exists(tfdir):
            os.makedirs(tfdir)
        filename = os.path.join(tfdir, name + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)
        for (imgname, attr) in data.items():
            print imgname
            '''fileID = int(i[:4])
            label = ""
            if fileID >= 1 and fileID <=110:
                label = "Walking"
            elif fileID >= 112 and fileID <=221:
                label = "Running"
            elif fileID >= 222 and fileID <=330:
                label = "Bend"
            elif fileID >= 331 and fileID <=429:
                label = "Boxing"
            elif fileID >= 430 and fileID <=529:
                label = "HandClap"
            elif fileID >= 684 and fileID <=692:
                label = "HandClap"
            elif fileID >= 530 and fileID <=538:
                label = "Jack"
            elif fileID >= 539 and fileID <=638:
                label = "Jogging"
            elif fileID >= 639 and fileID <=647:
                label = "Jump"
            elif fileID >= 648 and fileID <=656:
                label = "PJump"
            elif fileID >= 657 and fileID <=665:
                label = "Side"
            elif fileID >= 666 and fileID <=675:
                label = "Skip"
            elif fileID >= 676 and fileID <=684:
                label = "Wave1"
            else:'''
            label = "Misc"
            f = {}
            for (feat, val) in attr.items():
                f[feat] = _float_feature(val)
                #feat: _float64_feature(i[feat])
                #print feat
            f["label"] = _bytes_feature(label)
            #f["fileID"] = fileID
            #f["frame_num"] = framenum
            example = tf.train.Example(features=tf.train.Features(feature=f))
            writer.write(example.SerializeToString())
        writer.close()
