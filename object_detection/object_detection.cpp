#include "tensorflow/c/c_api.h"
#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>

using namespace std;

TF_Buffer* read_file(const char* file);
void free_buffer(void* data, size_t length) { free(data); }
static void Deallocator(void* data, size_t length, void* arg) { free(data); }
TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len);

int main(int argc, char* argv[]) {
  // Use read_file to get graph_def as TF_Buffer*
  // TF_Buffer* graph_def = read_file("/home/yoon/workspace/repos/rky0930/MyTensorflowBenchmark/models/object_detection/tensorflow/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb");
  // TF_Buffer* graph_def = read_file("/home/yoon/workspace/tf_c_api/ckpt/ssdlite_mobilenet_v2_1_400_225_coco_toastcam_v5/frozen_inference_graph.pb");
  TF_Buffer* graph_def = read_file("/home/yoon/workspace/tf_c_api/ckpt/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb");
  // TF_Buffer* graph_def = read_file("/home/yoon/workspace/tf_c_api/examples/tensorflow-object-detection-cpp/demo/ssd_mobilenet_v1_egohands/frozen_inference_graph.pb");

  // Import graph_def into graph
  TF_Graph* graph = TF_NewGraph();
  TF_Status* status = TF_NewStatus();
  TF_ImportGraphDefOptions* graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(graph, graph_def, graph_opts, status);
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(status));
    return 1;
  }
  else {
    fprintf(stdout, "Successfully imported graph\n");
  }
    
  // Input Image
  cout<<"Load image:"<<argv[1]<<endl;
  IplImage* img = cvLoadImage(argv[1], CV_LOAD_IMAGE_COLOR);
  cvCvtColor(img, img, CV_BGR2RGB);
  // cvNormalize(img, img, 10, 0);
  if (!img)
  {
    printf("Image can NOT Load!!!\n");
    return 1;
  }
  printf("First pixel: (");
  for (int i=0; i<3; i++) {
    printf("%u ", (uint8_t)img->imageData[i]);
  }
  printf(") \n");
  cout<<"Image size: "<<img->width<<", "<<img->height<<", "<<img->nChannels<<endl;
  cout<<"img->depth: "<<img->depth<<endl;
  cout<<"img->imageSize: "<<img->imageSize<<endl;
  cout<<"img->alphaChannel: "<<img->alphaChannel<<endl;
  cout<<"img->channelSeq : "<<img->channelSeq <<endl;
  cout<<"img->colorModel: "<<img->colorModel<<endl;
  cout<<"img->dataOrder: "<<img->dataOrder<<endl;
  cout<<"img->nSize: "<<img->nSize<<endl;
  cout<<"img->widthStep: "<<img->widthStep<<endl;
  int img_width = img->width;
  int img_height = img->height;
  int img_channel = img->nChannels;
  // ######################
  // Set up graph inputs
  // ######################

  // Pass the graph and a string name of your input operation
  // (make sure the operation name is correct)
  TF_Operation* input_op = TF_GraphOperationByName(graph, "image_tensor");
  TF_Output input_opout = {input_op, 0};
  TF_Output inputs[1] = {input_opout};
  // variables created earlier
  // const std::vector<std::int64_t> input_dims = {1, img_width, img_height, img_channel};
  const std::vector<std::int64_t> input_dims = {1, img_height, img_width, img_channel};
  cout<<img->imageSize<<endl;
  // TF_Tensor* input_value = TF_NewTensor(TF_UINT8, in_dims, 4, img->imageData, img->imageSize, &Deallocator, 0);
  TF_Tensor* input_value = CreateTensor(TF_UINT8,
                                        input_dims.data(), input_dims.size(),
                                        img->imageData, img->imageSize);
  TF_Tensor* input_values[1] = {input_value};
  // Optionally, you can check that your input_op and input tensors are correct
  // by using some of the functions provided by the C API.
  cout << "Input op info: " << TF_OperationNumInputs(input_op) << "\n";
  cout << "Input data info: " << TF_Dim(input_value, 1) << "\n";

  // ######################
  // Set up graph outputs (similar to setting up graph inputs)
  // ######################

  // Create vector to store graph output operations
  TF_Operation* boxes = TF_GraphOperationByName(graph, "detection_boxes");
  TF_Operation* scores = TF_GraphOperationByName(graph, "detection_scores");
  TF_Operation* classes = TF_GraphOperationByName(graph, "detection_classes");
  TF_Operation* num_detections = TF_GraphOperationByName(graph, "num_detections");
  TF_Output boxes_opout = {boxes, 0};
  TF_Output scores_opout = {scores, 0};
  TF_Output classes_opout = {classes, 0};
  TF_Output num_detections_opout = {num_detections, 0};
  TF_Output outputs[4] = {boxes_opout, scores_opout, classes_opout, num_detections_opout};

  // Similar to creating the input tensor, however here we don't yet have the
  // output values, so we use TF_AllocateTensor()
    // Create variables to store the size of the input and output variables
  int max_detections = 100;
  int64_t box_dims[] = {1, 100, 4};
  int64_t scores_dims[] = {1, 100};
  int64_t classes_dims[] = {1, 100};
  int64_t num_detections_dims[] = {1, 1};
  TF_Tensor* boxes_value = TF_AllocateTensor(TF_FLOAT, box_dims, 3, sizeof(float) * 4 * max_detections);
  TF_Tensor* scores_value = TF_AllocateTensor(TF_FLOAT, scores_dims, 2, sizeof(float) * max_detections);
  TF_Tensor* classes_value = TF_AllocateTensor(TF_FLOAT, classes_dims, 2, sizeof(float) * max_detections);
  TF_Tensor* num_detections_value = TF_AllocateTensor(TF_FLOAT, num_detections_dims, 2, sizeof(float));
  TF_Tensor* output_values[4] = {boxes_value, scores_value, classes_value, num_detections_value};
  
  // As with inputs, check the values for the output operation and output tensor
  cout << "Output: " << TF_OperationName(boxes) << "\n";
  cout << "Output: " << TF_OperationName(scores) << "\n";
  cout << "Output: " << TF_OperationName(classes) << "\n";
  cout << "Output: " << TF_OperationName(num_detections) << "\n";

  // ######################
  // Run graph
  // ######################
  fprintf(stdout, "Running session...\n");
  TF_SessionOptions* sess_opts = TF_NewSessionOptions();
  TF_Session* session = TF_NewSession(graph, sess_opts, status);
  assert(TF_GetCode(status) == TF_OK);

  // Call TF_SessionRun
  TF_SessionRun(session, nullptr,
                inputs, input_values, 1,
                outputs, output_values, 4,
                nullptr, 0, nullptr, status);

  cout<<"Print output: "<<endl;
  if(TF_GetCode(status) != TF_OK) {
    printf("It's coming here\n");
    printf("%s\n", TF_Message(status));
  }
  else {
    printf("Ran successfully\n");
    float* f = (float*)TF_TensorData(output_values[1]);
    float* b = (float*)TF_TensorData(output_values[0]);
    float* c = (float*)TF_TensorData(output_values[2]);
    float* n = (float*)TF_TensorData(output_values[3]);
    int num_detections = (int)n[0];
    // cout << "Output: " << TF_OperationName(outputs[0].oper) << "\n";
    // cout << "Output: " << TF_OperationName(outputs[1].oper) << "\n";
    // cout << "Output: " << TF_OperationName(outputs[2].oper) << "\n";
    // cout << "Output: " << TF_OperationName(outputs[3].oper) << "\n";
    printf("num_detections: %d\n", num_detections);
    for (int i=0; i<num_detections; i++) {
      if (f[i] >= 0.5) {
        printf("1 TF data %f\n", f[i]);
        int xmin = (int)(b[i*4+1] * img_width);
        int ymin = (int)(b[i*4+0] * img_height);
        int xmax = (int)(b[i*4+3] * img_width);
        int ymax = (int)(b[i*4+2] * img_height);
        cvRectangle(img, cvPoint(xmin, ymin), cvPoint(xmax, ymax), CV_RGB(0, 255, 255));
        cvShowImage("Drawing Graphics", img);
        cvWaitKey(0);
        // break;
      }
    }
    fprintf(stdout, "Successfully run session\n");
  }

  // Delete variables
  TF_CloseSession(session, status);
  TF_DeleteSession(session, status);
  TF_DeleteSessionOptions(sess_opts);
  TF_DeleteImportGraphDefOptions(graph_opts);
  TF_DeleteGraph(graph);
  TF_DeleteStatus(status);
  return 0;
}

TF_Buffer* read_file(const char* file) {
  FILE *f = fopen(file, "rb");
  fseek(f, 0, SEEK_END);
  long fsize = ftell(f);
  fseek(f, 0, SEEK_SET);  //same as rewind(f);

  void* data = malloc(fsize);
  fread(data, fsize, 1, f);
  fclose(f);

  TF_Buffer* buf = TF_NewBuffer();
  buf->data = data;
  buf->length = fsize;
  buf->data_deallocator = free_buffer;
  return buf;
}

TF_Tensor* CreateTensor(TF_DataType data_type,
                        const std::int64_t* dims, std::size_t num_dims,
                        const void* data, std::size_t len) {
  if (dims == nullptr || data == nullptr) {
    return nullptr;
  }

  TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), len);
  if (tensor == nullptr) {
    return nullptr;
  }

  void* tensor_data = TF_TensorData(tensor);
  if (tensor_data == nullptr) {
    TF_DeleteTensor(tensor);
    return nullptr;
  }

  std::memcpy(tensor_data, data, std::min(len, TF_TensorByteSize(tensor)));

  return tensor;
}