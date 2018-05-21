import tensorflow as tf
import os
import numpy as np
import shutil
import align_dataset_mtcnn
import sys

lines = tf.gfile.GFile('output_labels.txt').readlines()
uid_to_human = {}
# 一行一行读取数据
for uid, line in enumerate(lines):
    # 去掉换行符
    line = line.strip('\n')
    uid_to_human[uid] = line



def id_to_string(node_id):
    if node_id not in uid_to_human:
        return ''
    return uid_to_human[node_id]


# 创建一个图来存放google训练好的模型
with tf.gfile.FastGFile('output_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')



align_dataset_mtcnn.main(align_dataset_mtcnn.parse_arguments(sys.argv[1:]))

with tf.Session() as sess:

    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    path11 =r"D:\PycharmProjects\Test\retrain\test_crop\1"
    num = 0
    length = len(os.listdir(path11))

    for file in os.listdir(path11):

        if  os.path.isfile(os.path.join(path11,file)):

            image_data = tf.gfile.FastGFile(os.path.join(path11, file), 'rb').read()
            predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})  # 图片格式是jpg格式
            predictions = np.squeeze(predictions)  # 把结果转为1维数据

            # 排序
            top_k = predictions.argsort()[::-1]

            name = id_to_string(top_k[0])

            if name in file:
                num += 1
            else:
                print(file)

            path = r"D:\PycharmProjects\Test\retrain\test_crop\1"
            absPath = os.path.join(path, id_to_string(top_k[0]))
            # 对path路径下的图片进行分类
            if not os.path.exists(absPath):
                os.mkdir(absPath)
                # os.renames重命名文件，从而实现移动文件
                shutil.move(os.path.join(path, file), os.path.join(absPath, file))
            else:
                shutil.move(os.path.join(path, file), os.path.join(absPath, file))

    print("test acc: %d/%d = %.2f%%" %(num,length,num/length*100))
