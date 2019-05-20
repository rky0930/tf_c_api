## Tensorflow C API - Object Detection
 - Only Inference not for training
 
### Library requirements
- [Tensorflow C API](https://www.tensorflow.org/install/lang_c)
  - Version 1.13.1
- OpenCV
  - Version 3.4.2

### Docker Environment
```
# For CPU only
docker pull rky0930/tf_c_api:opencv
# For GPU 
docker pull rky0930/tf_c_api:cuda-10-cudnn7-opencv
```

### Comple
```
cd object_detection
make
```
### Model 
Use model of [Object detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

### Run
```
Usage : ./object_detection --(f)rozen_graph_path --(i)mage_path [--(c)onfidence_score_threshold] [--(m)ax_detections] [--(v)erbose] [--(s)how]
# ex_1) frozen graph / image dir 
./object_detection -f ckpt/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb -i image_dir
# ex_2) frozen graph / image dir / verbose mode / Show object detection result image
./object_detection -f ckpt/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb -i image_dir -v -s
```

### Reference
https://www.tensorflow.org/install/lang_c  
https://github.com/Neargye/hello_tf_c_api  
https://github.com/PatWie/tensorflow-cmake  
