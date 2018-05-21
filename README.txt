# googlenet-fine-tunning
用googLeNet 网络进行迁移学习
只对最后输出层进行训练，需要训练 n+2048*n个参数（n = classes）
--retrain
  --googlenet_fine
    --align_dataset_mtcnn.py
    --det1.npy
    --det2.npy
    --det3.npy
    --detect_face.py
    --facenet.py
    --google_fine_acc.py
    --output_graph.pb
    --output_labels.txt
    retrain.bat
    retain.py
