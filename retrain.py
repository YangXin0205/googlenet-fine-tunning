"""简单调用Inception V3架构模型的学习在tensorboard显示了摘要。

这个例子展示了如何采取一个Inception V3架构模型训练ImageNet图像和训练新的顶层，可以识别其他类的图像。 

每个图像里，顶层接收作为输入的一个2048维向量。这表示顶层我们训练一个softmax层。假设softmax层包含n个标签，这对应于学习N + 2048 * N模型参数对应于学习偏差和权重。 

这里有一个例子，假设你有一个文件夹，里面是包含类名的子文件夹，每一个里面放置每个标签的图像。示例flower_photos文件夹应该有这样的结构： 
~/flower_photos/daisy/photo1.jpg 
~/flower_photos/daisy/photo2.jpg 
... 
~/flower_photos/rose/anotherphoto77.jpg 
... 
~/flower_photos/sunflower/somepicture.jpg 
子文件夹的名字很重要，它们定义了每张图片的归类标签，而每张图片的名字是什么本身是没关系的。一旦你的图片准备好了，你可以使用如下命令启动训练: 
bazel build tensorflow/examples/image_retraining:retrain && \ 
bazel-bin/tensorflow/examples/image_retraining/retrain \ 
--image_dir ~/flower_photos 

你可以替换image_dir 参数为包含所需图片子文件夹的任何文件。每张图片的标签来自子文件夹的名字。 
程序将产生一个新的模型文件用于任何TensorFlow项目的加载和运行，例如label_image样例代码。 
为了使用 tensorboard。 
默认情况下，脚本的日志摘要生成在/tmp/retrain_logs目录 
可以使用这个命令来可视化这些摘要: 
tensorboard --logdir /tmp/retrain_logs 
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import hashlib
import os.path
import random
import re
import struct
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

FLAGS = None

# 这些是所有的参数，我们使用这些参数绑定到特定的InceptionV3模型结构。
# 这些包括张量名称和它们的尺寸。如果您想使此脚本与其他模型相适应，您将需要更新这些映射你在网络中使用的值。
# pylint：disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
BOTTLENECK_TENSOR_SIZE = 2048
MODEL_INPUT_WIDTH = 299
MODEL_INPUT_HEIGHT = 299
MODEL_INPUT_DEPTH = 3
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """从文件系统生成训练图像列表。分析图像目录中的子文件夹，将其分割成稳定的训练、测试和验证集，并返回数据结构，描述每个标签及其路径的图像列表。
    Args：
      image_dir：一个包含图片子文件夹的文件夹的字符串路径。
      testing_percentage：预留测试图像的整数百分比。
      validation_percentage：预留验证图像的整数百分比。
    Returns：
      一个字典包含进入每一个标签的子文件夹和分割到每个标签的训练，测试和验证集的图像。
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
    # 首先进入根目录，所以先跳过它。
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        print("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        if not file_list:
            print('No files found')
            continue
        if len(file_list) < 20:
            print('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            print('WARNING: Folder {} has more than {} images. Some images will '
                  'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # 当决定将图像放入哪个数据集时，我们想要忽略文件名里_nohash_之后的所有，数据集的创建者，有办法将有密切变化的图片分组。
            # 例如：这是用于设置相同的叶子的多组图片的植物疾病的数据集。
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # 这看起来有点不可思议，但是我们需要决定该文件是否应该进入训练、测试或验证集，我们要保持现有文件在同一数据集甚至更多的文件随后添加进来。
            # 为了这样做，我们需要一个稳定的方式决定只是基于文件名本身，因此我们做一个哈希值，然后使用其产生一个概率值供我们使用分配。
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result


def get_image_path(image_lists, label_name, index, image_dir, category):
    """"返回给定索引中标签的图像路径。
    Args：
      image_lists：训练图像每个标签的词典。
      label_name：我们想得到的一个图像的标签字符串。
      index：我们想要图像的Int 偏移量。这将以标签的可用的图像数为模，因此它可以任意大。
      image_dir：包含训练图像的子文件夹的根文件夹字符串。
      category：从图像训练、测试或验证集提取的图像的字符串名称。
    Returns：
      将文件系统路径字符串映射到符合要求参数的图像。
    """
    if label_name not in image_lists:
        tf.logging.fatal('Label does not exist %s.', label_name)
    label_lists = image_lists[label_name]
    if category not in label_lists:
        tf.logging.fatal('Category does not exist %s.', category)
    category_list = label_lists[category]
    if not category_list:
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path


def get_bottleneck_path(image_lists, label_name, index, bottleneck_dir,
                        category):
    """"返回给定索引中的标签的瓶颈文件的路径。
    Args：
      image_lists：训练图像每个标签的词典。
      label_name：我们想得到的一个图像的标签字符串。
      index：我们想要图像的Int 偏移量。这将以标签的可用的图像数为模，因此它可以任意大。
      bottleneck_dir：文件夹字符串保持缓存文件的瓶颈值。
      category：从图像训练、测试或验证集提取的图像的字符串名称。
    Returns：
      将文件系统路径字符串映射到符合要求参数的图像。
    """
    return get_image_path(image_lists, label_name, index, bottleneck_dir,
                          category) + '.txt'


def create_inception_graph():
    """"从保存的GraphDef文件创建一个图像并返回一个图像对象。
    Returns：
      我们将操作的持有训练的Inception网络和各种张量的图像。
    """
    with tf.Session() as sess:
        model_filename = os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb')
        with gfile.FastGFile(model_filename, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(graph_def, name='', return_elements=[
                    BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
                    RESIZED_INPUT_TENSOR_NAME]))
    return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor


def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    """在图像上运行推理以提取“瓶颈”摘要层。
    Args：
      sess：当前活动的tensorflow会话。
      image_data：原JPEG数据字符串。
      image_data_tensor：图中的输入数据层。
      bottleneck_tensor：最后一个softmax之前的层。
    Returns：
      NumPy数组的瓶颈值。
    """
    bottleneck_values = sess.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values


def maybe_download_and_extract():
    """下载并提取模型的tar文件。
      如果我们使用的pretrained模型已经不存在，这个函数会从tensorflow.org网站下载它并解压缩到一个目录。
    """
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %
                             (filename,
                              float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                                 filepath,
                                                 _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def ensure_dir_exists(dir_name):
    """确保文件夹已经在磁盘上存在。
    Args:
      dir_name: 我们想创建的文件夹路径的字符串。
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def write_list_of_floats_to_file(list_of_floats, file_path):
    """将一个已给定的floats列表写入到一个二进制文件。
    Args:
      list_of_floats: 我们想写入到一个文件的floats列表。
      file_path: floats列表文件将要存储的路径。
    """

    s = struct.pack('d' * BOTTLENECK_TENSOR_SIZE, *list_of_floats)
    with open(file_path, 'wb') as f:
        f.write(s)


def read_list_of_floats_from_file(file_path):
    """从一个给定的文件读取floats列表。
    Args:
      file_path: floats列表文件存储的的路径。
    Returns: 瓶颈值的数组 (floats列表)。
    """
    with open(file_path, 'rb') as f:
        s = struct.unpack('d' * BOTTLENECK_TENSOR_SIZE, f.read())
        return list(s)

bottleneck_path_2_bottleneck_values = {}


def create_bottleneck_file(bottleneck_path, image_lists, label_name, index,
                           image_dir, category, sess, jpeg_data_tensor, bottleneck_tensor):
    print('Creating bottleneck at ' + bottleneck_path)
    image_path = get_image_path(image_lists, label_name, index, image_dir, category)
    if not gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = gfile.FastGFile(image_path, 'rb').read()
    bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    with open(bottleneck_path, 'w') as bottleneck_file:
        bottleneck_file.write(bottleneck_string)


def get_or_create_bottleneck(sess, image_lists, label_name, index, image_dir,
                             category, bottleneck_dir, jpeg_data_tensor,
                             bottleneck_tensor):
    """检索或计算图像的瓶颈值。
      如果磁盘上存在瓶颈数据的缓存版本，则返回，否则计算数据并将其保存到磁盘以备将来使用。
    Args:
      sess:当前活动的tensorflow会话。
      image_lists：每个标签的训练图像的词典。
      label_name：我们想得到一个图像的标签字符串。
      index：我们想要的图像的整数偏移量。这将以标签图像的可用数为模，所以它可以任意大。
      image_dir：包含训练图像的子文件夹的根文件夹字符串。
      category：从图像训练、测试或验证集提取的图像的字符串名称。
      bottleneck_dir：保存着缓存文件瓶颈值的文件夹字符串。
      jpeg_data_tensor：满足加载的JPEG数据进入的张量。
      bottleneck_tensor：瓶颈值的输出张量。
    Returns:
      通过图像的瓶颈层产生的NumPy数组值。
     """


    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    ensure_dir_exists(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, bottleneck_dir, category)
    if not os.path.exists(bottleneck_path):
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    except:
        print("Invalid float found, recreating bottleneck")
        did_hit_error = True
    if did_hit_error:
        create_bottleneck_file(bottleneck_path, image_lists, label_name, index, image_dir, category, sess, jpeg_data_tensor,
                               bottleneck_tensor)
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
            # 允许在这里传递异常，因为异常不应该发生在一个新的bottleneck创建之后。
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def cache_bottlenecks(sess, image_lists, image_dir, bottleneck_dir,
                      jpeg_data_tensor, bottleneck_tensor):
    """确保所有的训练，测试和验证瓶颈被缓存。
    因为我们可能会多次读取同一个图像（如果在训练中没有应用扭曲）。如果我们每个图像预处理期间的瓶颈层值只计算一次，在训练时只需反复读取这些缓存值，能大幅的加快速度。在这里，我们检测所有发现的图像，计算那些值，并保存。
    Args：
      sess：当前活动的tensorflow会话。
      image_lists：每个标签的训练图像的词典。
      image_dir：包含训练图像的子文件夹的根文件夹字符串。
      bottleneck_dir：保存着缓存文件瓶颈值的文件夹字符串。
      jpeg_data_tensor：从文件输入的JPEG数据的张量。
      bottleneck_tensor：图中的倒数第二输出层。
    Returns:
     无。
    """


    how_many_bottlenecks = 0
    ensure_dir_exists(bottleneck_dir)
    for label_name, label_lists in image_lists.items():
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            for index, unused_base_name in enumerate(category_list):
                get_or_create_bottleneck(sess, image_lists, label_name, index,
                                         image_dir, category, bottleneck_dir,
                                         jpeg_data_tensor, bottleneck_tensor)

                how_many_bottlenecks += 1
                if how_many_bottlenecks % 100 == 0:
                    print(str(how_many_bottlenecks) + ' bottleneck files created.')


def get_random_cached_bottlenecks(sess, image_lists, how_many, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):
    """检索缓存图像的瓶颈值。
     如果没有应用扭曲，这个函数可以直接从磁盘检索图像缓存的瓶颈值。它从指定类别的图像挑选了一套随机的数据集。
    Args：
      sess：当前活动的tensorflow会话。
      image_lists：每个标签的训练图像的词典。
      how_many：如果为正数，将选择一个随机样本的尺寸大小。如果为负数，则将检索所有瓶颈。
      category：从图像训练、测试或验证集提取的图像的字符串名称。
      bottleneck_dir：保存着缓存文件瓶颈值的文件夹字符串。
      image_dir：包含训练图像的子文件夹的根文件夹字符串。
      jpeg_data_tensor：JPEG图像数据导入的层。
      bottleneck_tensor：CNN图的瓶颈输出层。
    Returns:
      瓶颈数组的列表，它们对应于ground truths和相关的文件名。
    """


    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
    # 检索瓶颈的一个随机样本。
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_image_path(image_lists, label_name, image_index,
                                        image_dir, category)
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                  image_index, image_dir, category,
                                                  bottleneck_dir, jpeg_data_tensor,
                                                  bottleneck_tensor)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # 检索所有的瓶颈。
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = get_image_path(image_lists, label_name, image_index,
                                            image_dir, category)
                bottleneck = get_or_create_bottleneck(sess, image_lists, label_name,
                                                      image_index, image_dir, category,
                                                      bottleneck_dir, jpeg_data_tensor,
                                                      bottleneck_tensor)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return bottlenecks, ground_truths, filenames


def get_random_distorted_bottlenecks(
        sess, image_lists, how_many, category, image_dir, input_jpeg_tensor,
        distorted_image, resized_input_tensor, bottleneck_tensor):
    """检索训练图像扭曲后的瓶颈值。
     如果我们训练使用扭曲变换，如裁剪，缩放，或翻转，我们必须重新计算每个图像的完整模型，所以我们不能使用缓存的瓶颈值。相反，我们找出所要求类别的随机图像，通过扭曲图运行它们，然后得到每个瓶颈结果完整的图。
    Args：
      sess：当前的tensorflow会话。
      image_lists：每个标签的训练图像的词典。
      how_many：返回瓶颈值的整数个数。
      category：要获取的图像训练、测试，或验证集的名称字符串。
      image_dir：包含训练图像的子文件夹的根文件夹字符串.
      input_jpeg_tensor：给定图像数据的输入层。
      distorted_image：畸变图形的输出节点。
      resized_input_tensor：识别图的输入节点。
      bottleneck_tensor：CNN图的瓶颈输出层。
    Returns:
      瓶颈阵列及其对应的ground truths列表。
    """


    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for unused_i in range(how_many):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
        image_path = get_image_path(image_lists, label_name, image_index, image_dir,
                                    category)
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)
    jpeg_data = gfile.FastGFile(image_path, 'rb').read()
    # 注意我们实现distorted_image_data作为NumPy数组是在发送运行推理的图像之前。这涉及2个内存副本和可能在其他实现里优化。
    distorted_image_data = sess.run(distorted_image,
                                    {input_jpeg_tensor: jpeg_data})
    bottleneck = run_bottleneck_on_image(sess, distorted_image_data,
                                         resized_input_tensor,
                                         bottleneck_tensor)
    ground_truth = np.zeros(class_count, dtype=np.float32)
    ground_truth[label_index] = 1.0
    bottlenecks.append(bottleneck)
    ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def should_distort_images(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """从输入标志是否已启用任何扭曲。
    Args：
      flip_left_right：是否随机镜像水平的布尔值。
      random_crop：在裁切框设置总的边缘的整数百分比。
      random_scale：缩放变化多少的整数百分比。
      random_brightness：随机像素值的整数范围。
   Returns：
     布尔值，指示是否应用任何扭曲。
   """


    return (flip_left_right or (random_crop != 0) or (random_scale != 0) or
            (random_brightness != 0))


def add_input_distortions(flip_left_right, random_crop, random_scale,
                          random_brightness):
    """创建用于应用指定扭曲的操作。

     在训练过程中，如果我们运行的图像通过简单的扭曲，如裁剪，缩放和翻转，可以帮助改进结果。这些反映我们期望在现实世界中的变化，因此可以帮助训练模型，以更有效地应对自然数据。在这里，我们采取的供应参数并构造一个操作网络以将它们应用到图像中。

    裁剪
    ~~~~~~~~

    裁剪是通过在完整的图像上一个随机的位置放置一个边界框。裁剪参数控制该框相对于输入图像的尺寸大小。如果它是零，那么该框以输入图像相同的大小作为输入不进行裁剪。如果值是50%，则裁剪框将是输入的宽度和高度的一半。在图中看起来像这样：
    <       width         >
     +---------------------+
     |                     |
     |   width - crop%     |
     |    <      >         |
     |    +------+         |
     |    |      |         |
     |    |      |         |
     |    |      |         |
     |    +------+         |
     |                     |
     |                     |
     +---------------------+

    缩放
    ~~~~~~~
    缩放是非常像裁剪，除了边界框总是在中心和它的大小在给定的范围内随机变化。例如，如果缩放比例百分比为零，则边界框与输入尺寸大小相同，没有缩放应用。如果它是50%，那么边界框将是宽度和高度的一半和全尺寸之间的随机范围。
    Args：
      flip_left_right：是否随机镜像水平的布尔值。
      random_crop：在裁切框设置总的边缘的整数百分比。
      random_scale：缩放变化多少的整数百分比。
      random_brightness：随机像素值的整数范围。
    Returns：
     JPEG输入层和扭曲结果的张量。
    """


    jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
    decoded_image_as_float = tf.cast(decoded_image, dtype=tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    margin_scale = 1.0 + (random_crop / 100.0)
    resize_scale = 1.0 + (random_scale / 100.0)
    margin_scale_value = tf.constant(margin_scale)
    resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
                                           minval=1.0,
                                           maxval=resize_scale)
    scale_value = tf.multiply(margin_scale_value, resize_scale_value)
    precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
    precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
    precrop_shape = tf.stack([precrop_height, precrop_width])
    precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
    precropped_image = tf.image.resize_bilinear(decoded_image_4d,
                                                precrop_shape_as_int)
    precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
    cropped_image = tf.random_crop(precropped_image_3d,
                                   [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
                                    MODEL_INPUT_DEPTH])
    if flip_left_right:
        flipped_image = tf.image.random_flip_left_right(cropped_image)
    else:
        flipped_image = cropped_image
    brightness_min = 1.0 - (random_brightness / 100.0)
    brightness_max = 1.0 + (random_brightness / 100.0)
    brightness_value = tf.random_uniform(tensor_shape.scalar(),
                                         minval=brightness_min,
                                         maxval=brightness_max)
    brightened_image = tf.multiply(flipped_image, brightness_value)
    distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')
    return jpeg_data, distort_result


def variable_summaries(var):
    """附加一个张量的很多总结（为tensorboard可视化）。"""


    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def add_final_training_ops(class_count, final_tensor_name, bottleneck_tensor):
    """为训练增加了一个新的softmax和全连接层。
    我们需要重新训练顶层识别我们新的类，所以这个函数向图表添加正确的操作，以及一些变量来保持
    权重，然后设置所有的梯度向后传递。

    softmax和全连接层的设置是基于：
     https://tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    Args：
      class_count：我们需要识别多少种类东西的整数数目。
      final_tensor_name：产生结果时新的最后节点的字符串名称。
      bottleneck_tensor：主CNN图像的输出。
    Returns：
    训练的张量和交叉熵的结果，瓶颈输入和groud truth输入的张量。
    """


    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor, shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(tf.float32,
                                            [None, class_count],
                                            name='GroundTruthInput')
        # 组织以下的ops作为‘final_training_ops’,这样在TensorBoard里更容易看到。
    layer_name = 'final_training_ops'
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            layer_weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, class_count], stddev=0.001),
                                        name='final_weights')
            variable_summaries(layer_weights)
        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
            variable_summaries(layer_biases)
        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input, logits=logits)
        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)
    tf.summary.scalar('cross_entropy', cross_entropy_mean)

    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(
            cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input, ground_truth_input,
            final_tensor)


def add_evaluation_step(result_tensor, ground_truth_tensor):
    """插入我们需要的操作，以评估我们结果的准确性。
    Args：
      result_tensor：产生结果的新的最后节点。
      ground_truth_tensor：我们提供的groud truth数据的节点。
    Returns：
      元组（评价步骤，预测）。
    """


    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            prediction = tf.argmax(result_tensor, 1)
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))
        with tf.name_scope('accuracy'):
            evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)
    return evaluation_step, prediction


def main(_):


    # 设置我们写入TensorBoard摘要的目录。
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    # 设置预训练图像。
    maybe_download_and_extract()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = (
        create_inception_graph())

    # 查看文件夹结构，创建所有图像的列表。
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    class_count = len(image_lists.keys())
    if class_count == 0:
        print('No valid folders of images found at ' + FLAGS.image_dir)
        return -1
    if class_count == 1:
        print('Only one valid folder of images found at ' + FLAGS.image_dir +
              ' - multiple classes are needed for classification.')
        return -1

        # 看命令行标记是否意味着我们应用任何扭曲操作。
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)
    sess = tf.Session()

    if do_distort_images:
        # 我们将应用扭曲，因此设置我们需要的操作。
        distorted_jpeg_data_tensor, distorted_image_tensor = add_input_distortions(
            FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
            FLAGS.random_brightness)
    else:
        # 我们确定计算bottleneck图像总结并缓存在磁盘上。
        cache_bottlenecks(sess, image_lists, FLAGS.image_dir, FLAGS.bottleneck_dir,
                          jpeg_data_tensor, bottleneck_tensor)

        # 添加我们将要训练的新层。
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(len(image_lists.keys()),
                                            FLAGS.final_tensor_name,
                                            bottleneck_tensor)

    # 创建操作，我们需要评估新层的准确性。
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)

    # 合并所有的摘要，写到/tmp/retrain_logs(默认)。
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')

    # 设置所有的权重到初始的默认值。
    init = tf.global_variables_initializer()
    sess.run(init)

    # 按照命令行的要求运行多个周期的训练。
    for i in range(FLAGS.how_many_training_steps):
        # 获得一批输入瓶颈值，或是用应用的扭曲每一次计算新的值，或是缓存并存储在磁盘上的值。
        if do_distort_images:
            train_bottlenecks, train_ground_truth = get_random_distorted_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.image_dir, distorted_jpeg_data_tensor,
                distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
            train_bottlenecks, train_ground_truth, _ = get_random_cached_bottlenecks(
                sess, image_lists, FLAGS.train_batch_size, 'training',
                FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                bottleneck_tensor)
            # 给图像提供瓶颈和groud truth，运行一个训练阶。用‘合并’op计算训练的TensorBoard摘要。
        train_summary, _ = sess.run([merged, train_step],
                                    feed_dict={bottleneck_input: train_bottlenecks,
                                               ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)

        # 每隔一段时间，打印出来图像是如何训练的。
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                       cross_entropy_value))
            validation_bottlenecks, validation_ground_truth, _ = (
                get_random_cached_bottlenecks(
                    sess, image_lists, FLAGS.validation_batch_size, 'validation',
                    FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
                    bottleneck_tensor))
            # 运行一个验证阶。用‘合并’op计算训练的TensorBoard摘要。
            validation_summary, validation_accuracy = sess.run(
                [merged, evaluation_step],
                feed_dict={bottleneck_input: validation_bottlenecks,
                           ground_truth_input: validation_ground_truth})
            validation_writer.add_summary(validation_summary, i)
            print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                  (datetime.now(), i, validation_accuracy * 100,
                   len(validation_bottlenecks)))

            # 我们已完成了所有的训练，在一些我们从未用过的新的图像上，运行一个最后的测试评估。
    test_bottlenecks, test_ground_truth, test_filenames = (
        get_random_cached_bottlenecks(sess, image_lists, FLAGS.test_batch_size,
                                      'testing', FLAGS.bottleneck_dir,
                                      FLAGS.image_dir, jpeg_data_tensor,
                                      bottleneck_tensor))
    test_accuracy, predictions = sess.run(
        [evaluation_step, prediction],
        feed_dict={bottleneck_input: test_bottlenecks,
                   ground_truth_input: test_ground_truth})
    print('Final test accuracy = %.1f%% (N=%d)' % (
        test_accuracy * 100, len(test_bottlenecks)))

    if FLAGS.print_misclassified_test_images:
        print('=== MISCLASSIFIED TEST IMAGES ===')
        for i, test_filename in enumerate(test_filenames):
            if predictions[i] != test_ground_truth[i].argmax():
                print('%70s  %s' % (test_filename,
                                    list(image_lists.keys())[predictions[i]]))

                # 写出训练的图像和以常数形式存储的权重标签。
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='/tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='/tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
        '--summaries_dir',
        type=str,
        default='/tmp/retrain_logs',
        help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--how_many_training_steps',
        type=int,
        default=4000,
        help='How many training steps to run before ending.'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='How large a learning rate to use when training.'
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=10,
        help='How often to evaluate the training results.'
    )
    parser.add_argument(
        '--train_batch_size',
        type=int,
        default=100,
        help='How many images to train on at a time.'
    )
    parser.add_argument(
        '--test_batch_size',
        type=int,
        default=-1,
        help="""\ 
      How many images to test on. This test set is only used once, to evaluate 
      the final accuracy of the model after training completes. 
      A value of -1 causes the entire test set to be used, which leads to more 
      stable results across runs.\ 
      """
    )
    parser.add_argument(
        '--validation_batch_size',
        type=int,
        default=100,
        help="""\ 
      How many images to use in an evaluation batch. This validation set is 
      used much more often than the test set, and is an early indicator of how 
      accurate the model is during training. 
      A value of -1 causes the entire validation set to be used, which leads to 
      more stable results across training iterations, but may be slower on large 
      training sets.\ 
      """
    )
    parser.add_argument(
        '--print_misclassified_test_images',
        default=False,
        help="""\ 
      Whether to print out a list of all misclassified test images.\ 
      """,
        action='store_true'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\ 
      Path to classify_image_graph_def.pb, 
      imagenet_synset_to_human_label_map.txt, and 
      imagenet_2012_challenge_label_map_proto.pbtxt.\ 
      """
    )
    parser.add_argument(
        '--bottleneck_dir',
        type=str,
        default='/tmp/bottleneck',
        help='Path to cache bottleneck layer values as files.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\ 
      The name of the output classification layer in the retrained graph.\ 
      """
    )
    parser.add_argument(
        '--flip_left_right',
        default=False,
        help="""\ 
      Whether to randomly flip half of the training images horizontally.\ 
      """,
        action='store_true'
    )
    parser.add_argument(
        '--random_crop',
        type=int,
        default=0,
        help="""\ 
      A percentage determining how much of a margin to randomly crop off the 
      training images.\ 
      """
    )
    parser.add_argument(
        '--random_scale',
        type=int,
        default=0,
        help="""\ 
      A percentage determining how much to randomly scale up the size of the 
      training images by.\ 
      """
    )
    parser.add_argument(
        '--random_brightness',
        type=int,
        default=0,
        help="""\ 
      A percentage determining how much to randomly multiply the training image 
      input pixels up or down by.\ 
      """
    )
    FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
