import os, sys, re, subprocess as sb
# import feature_extractor  # old version
import tensorflow as tf
import glob
import numpy
import cv2
import shutil
from ast import literal_eval
import pandas as pd

"""
    2 kinds of video:
        1. video with audio --> use new extractor which extracts rgb&audio. *default*
        2. video without audio --> use old extractor which only extracts rgb and put 0 into audio
"""

# old version
model_dir = '/data/thunder_video/ucf101/thunder/extract_model'

import time
from time import strftime
from datetime import datetime 
from time import gmtime

def start_time_():    
    #import time
    start_time = time.time()
    return(start_time)

def end_time_():
    #import time
    end_time = time.time()
    return(end_time)

def Execution_time(start_time_,end_time_):
   #import time
   #from time import strftime
   #from datetime import datetime 
   #from time import gmtime
   return(strftime("%H:%M:%S",gmtime(int('{:.0f}'.format(float(str((end_time-start_time))))))))


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _int64_list_feature(int64_list):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature_list(values):
    """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
    return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _make_bytes(int_array):
    if bytes == str:  # Python2
        return ''.join(map(chr, int_array))
    else:
        return bytes(int_array)


def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    """Quantizes float32 `features` into string."""
    assert features.dtype == 'float32'
    assert len(features.shape) == 1  # 1-D array
    features = numpy.clip(features, min_quantized_value, max_quantized_value)
    quantize_range = max_quantized_value - min_quantized_value
    features = (features - min_quantized_value) * (255.0 / quantize_range)
    features = [int(round(f)) for f in features]

    return _make_bytes(features)


def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
    """
    Dequantize the feature from the byte format to the float format.

    Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

    Returns:
    A float vector which has the same shape as feat_vector.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias


def check_converted(file_name, dst_path):
    """
    :param file_name: filename with full path
    :param dst_path: tfrecord save path
    :return: flag of converted or new file
    """
    base_name = file_name.split('/')[-1]
    base_name = base_name.split('.')[0]
    dst_file = os.path.join(dst_path, base_name + '.tfrecord')
    return os.path.exists(dst_file)


##### old version just extract rgb feature and put 0 verc in audio feature list
# In OpenCV3.X, this is available as cv2.CAP_PROP_POS_MSEC
# In OpenCV2.X, this is available as cv2.cv.CV_CAP_PROP_POS_MSEC
CAP_PROP_POS_MSEC = 0


def frame_iterator(filename, max_num_frames=300):
    """Uses OpenCV to iterate over all frames of filename at a given frequency.

    Args:
      filename: Path to video file (e.g. mp4)
      every_ms: The duration (in milliseconds) to skip between frames.
      max_num_frames: Maximum number of frames to process, taken from the
        beginning of the video.

    Yields:
      RGB frame with shape (image height, image width, channels)
    """
    video_capture = cv2.VideoCapture()
    if not video_capture.open(filename):
        print >> sys.stderr, 'Error: Cannot open video file ' + filename
        return
    last_ts = -99999  # The timestamp of last retrieved frame.
    num_retrieved = 0

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = video_capture.get(cv2.cv.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

    every_ms = 1000.0 / fps

    while num_retrieved < max_num_frames:
        # Skip frames
        while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
            if not video_capture.read()[0]:
                return

        last_ts = video_capture.get(CAP_PROP_POS_MSEC)
        has_frames, frame = video_capture.read()
        if not has_frames:
            break
        yield frame
        num_retrieved += 1


def extract_rgb(video_file, output_tfrecords_file, labels, id,
                model_dir=model_dir,
                insert_zero_audio_features=True,
                skip_frame_level_features=False):
    extractor = feature_extractor.YouTube8MFeatureExtractor(model_dir)
    writer = tf.python_io.TFRecordWriter(output_tfrecords_file)
    total_written = 0
    total_error = 0

    rgb_features = []
    sum_rgb_features = None
    for rgb in frame_iterator(video_file):
        features = extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
        if sum_rgb_features is None:
            sum_rgb_features = features
        else:
            sum_rgb_features += features
        rgb_features.append(_bytes_feature(quantize(features)))

    if not rgb_features:
        print('Could not get features for ' + video_file)
        return

    mean_rgb_features = sum_rgb_features / len(rgb_features)

    # Create SequenceExample proto and write to output.
    feature_list = {
        'rgb': tf.train.FeatureList(feature=rgb_features),
    }
    context_features = {
        'labels':
            _int64_list_feature(labels),
        'id':
            _bytes_feature(_make_bytes(id.encode('utf-8'))),
        'mean_rgb':
            tf.train.Feature(
                float_list=tf.train.FloatList(value=mean_rgb_features)),
    }

    if insert_zero_audio_features:
        zero_vec = [0] * 128
        feature_list['audio'] = tf.train.FeatureList(
            feature=[_bytes_feature(_make_bytes(zero_vec))] * len(rgb_features))
        context_features['mean_audio'] = tf.train.Feature(
            float_list=tf.train.FloatList(value=zero_vec))

    if skip_frame_level_features:
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features))
    else:
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature=context_features),
            feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    print(example)
    writer.write(example.SerializeToString())
    total_written += 1

    writer.close()
    print('Successfully encoded %i out of %i videos' %
          (total_written, total_written + total_error))


##### new extractor for mediapipe converted feature.pb include both rgb&audio features

contexts = {
    'AUDIO/feature/dimensions': tf.io.FixedLenFeature([], tf.int64),
    'AUDIO/feature/rate': tf.io.FixedLenFeature([], tf.float32),
    'RGB/feature/dimensions': tf.io.FixedLenFeature([], tf.int64),
    'RGB/feature/rate': tf.io.FixedLenFeature([], tf.float32),
    'clip/data_path': tf.io.FixedLenFeature([], tf.string),
    'clip/end/timestamp': tf.io.FixedLenFeature([], tf.int64),
    'clip/start/timestamp': tf.io.FixedLenFeature([], tf.int64)
}

features = {
    'AUDIO/feature/floats': tf.io.VarLenFeature(dtype=tf.float32),
    'AUDIO/feature/timestamp': tf.io.VarLenFeature(tf.int64),
    'RGB/feature/floats': tf.io.VarLenFeature(dtype=tf.float32),
    'RGB/feature/timestamp': tf.io.VarLenFeature(tf.int64)
}


def parse_exmp(serial_exmp):
    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serial_exmp,
        context_features=contexts,
        sequence_features=features)

    sequence_parsed = tf.contrib.learn.run_n(sequence_parsed)[0]

    # audio : [10, 128]
    # rgb : [10, 1024]

    audio = sequence_parsed['AUDIO/feature/floats'].values
    rgb = sequence_parsed['RGB/feature/floats'].values

#     print(audio)
    
    audio_slices = [audio[128 * i: 128 * (i + 1)] for i in range(len(audio) // 128)]
    rgb_slices = [rgb[1024 * i: 1024 * (i + 1)] for i in range(len(rgb) // 1024)]

    byte_audio = []
    byte_rgb = []

    for seg in audio_slices:
        audio_seg = quantize(seg)
        byte_audio.append(audio_seg)

    for seg in rgb_slices:
        rgb_seg = quantize(seg)
        byte_rgb.append(rgb_seg)

    return byte_audio, byte_rgb


def make_exmp(id, labels, audio, rgb):
    audio_features = []
    rgb_features = []

    for embedding in audio:
        embedding_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[embedding]))
        audio_features.append(embedding_feature)

    for embedding in rgb:
        embedding_feature = tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[embedding]))
        rgb_features.append(embedding_feature)

    seq_exmp = tf.train.SequenceExample(
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'audio': tf.train.FeatureList(
                    feature=audio_features
                ),
                'rgb': tf.train.FeatureList(
                    feature=rgb_features
                )
            }),
        context=tf.train.Features(
            feature={
                'id': tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[id.encode('utf-8')])),
                'labels': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=labels))
            })
    )
    serialized = seq_exmp.SerializeToString()
    return serialized

# only for new version

data_root = '/curated_videos'
csv_path = '/attention_cluster_data_encoded.csv'
data_dict = pd.read_csv(csv_path, header=None, dtype={0: str}).set_index(0).squeeze().to_dict()
start_time = start_time_()
for file_name in glob.glob(data_root + '/*'):
    
    try:
   
        tmp_dir = '/tmp_pb/'

        dst_dir = '/tfrecord/'

        base_name_comp = os.path.basename(file_name)
        if("_trimmed_1" in base_name_comp):
            base_name = os.path.splitext(base_name_comp)[0].rstrip("_trimmed_1")[-11:]
        else:
            base_name = os.path.splitext(base_name_comp)[0][-11:]     
        # print(base_name)

        #checking if the pb file exists already
        if(os.path.exists(tmp_dir + base_name + '.pb') or "." in base_name):
            pass
        else:
            #change clip_end_time_sec acc to the max duration amongst the videos, so you dont end up clippng the video before feature extraction
            step_1 = f'python -m mediapipe.examples.desktop.youtube8m.generate_input_sequence_example --path_to_input_video="{file_name}" --clip_end_time_sec=600'

            status = os.system(step_1)
            #generating the pb files that will contain the frame level features, both audio and rgb
            if status == 0:
                step_2 = 'GLOG_logtostderr=1 bazel-bin/mediapipe/examples/desktop/youtube8m/extract_yt8m_features \
                        --calculator_graph_config_file=mediapipe/graphs/youtube8m/feature_extraction.pbtxt \
                        --input_side_packets=input_sequence_example=/tmp/mediapipe/metadata.pb  \
                        --output_side_packets=output_sequence_example=' + tmp_dir + base_name + '.pb'
        
                status = os.system(step_2)
                print(status)

                pb_filename = tmp_dir + base_name + '.pb'

                sequence_example = open(pb_filename, 'rb').read()

                audio, rgb = parse_exmp(sequence_example)
                id = base_name

                labels = literal_eval(data_dict[base_name_comp]) 

                tmp_example = make_exmp(id, labels, audio, rgb)

                decoded = tf.train.SequenceExample.FromString(tmp_example)
    

                with tf.python_io.TFRecordWriter(f'{dst_dir}{base_name}.tfrecord') as writer:
                    writer.write(tmp_example)
                temp_csv = "time_inference.csv"  
                #running the inference script along, to check the full cycle duration of the inference pipeline 
                #os.system("python inference.py --train_dir=AttentionCluster/model_chkpts_2 --input_data_pattern="+ dst_dir +" --top_k=5 --output_file="+temp_csv+"")
                
                
    except Exception as e:
        print(e)
        pass
end_time = end_time_()
print("Execution_time is :", Execution_time(start_time,end_time))
