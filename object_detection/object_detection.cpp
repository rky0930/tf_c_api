#include "object_detection.hpp"

void free_buffer(void* data, size_t length) { free(data); }

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
  // void* tensor_data = TF_TensorData(tensor);
  // if (tensor_data == nullptr) {
  //   TF_DeleteTensor(tensor);
  //   return nullptr;
  // }
  // std::memcpy(tensor_data, data, std::min(len, TF_TensorByteSize(tensor)));
  std::memcpy(TF_TensorData(tensor), data, std::min(len, TF_TensorByteSize(tensor)));
  return tensor;
}

ObjectDetection::ObjectDetection(std::string frozen_graph_path, float confidence_score_threshold, 
                                 int max_detections) {
  this->frozen_graph_path = frozen_graph_path;
  this->confidence_score_threshold = confidence_score_threshold;
  this->max_detections = max_detections;
}

void ObjectDetection::init() {
  this->set_graph();
}

void ObjectDetection::set_graph() {
  std::cout<<"Load Model: "<<this->frozen_graph_path<<std::endl;
  this->graph_def = this->read_file(this->frozen_graph_path);
  this->graph = TF_NewGraph();
  this->graph_status = TF_NewStatus();
  this->graph_opts = TF_NewImportGraphDefOptions();
  TF_GraphImportGraphDef(this->graph, this->graph_def, 
                         this->graph_opts, this->graph_status);
  if (TF_GetCode(this->graph_status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to import graph %s", TF_Message(this->graph_status));
    exit(1);
  } else {
    fprintf(stdout, "Successfully imported graph\n");
  }

  // Create Session 
  this->sess_opts = TF_NewSessionOptions();
  this->sess_status = TF_NewStatus();
  this->sess = TF_NewSession(this->graph, this->sess_opts, this->sess_status);
  if(TF_GetCode(this->sess_status) != TF_OK) {
    fprintf(stderr, "ERROR: Unable to create session %s", TF_Message(this->sess_status));
  }
  // Set up input op
  this->input_op = TF_GraphOperationByName(graph, "image_tensor");
  this->input_opout = {input_op, 0};
  this->input_ops.push_back(input_opout);
  // Set up output ops
  this->boxes_op = TF_GraphOperationByName(graph, "detection_boxes");
  this->scores_op = TF_GraphOperationByName(graph, "detection_scores");
  this->classes_op = TF_GraphOperationByName(graph, "detection_classes");
  this->num_detections_op = TF_GraphOperationByName(graph, "num_detections");
  this->boxes_opout = {boxes_op, 0};
  this->scores_opout = {scores_op, 0};
  this->classes_opout = {classes_op, 0};
  this->num_detections_opout = {num_detections_op, 0};
  this->output_ops.push_back(boxes_opout);
  this->output_ops.push_back(scores_opout);
  this->output_ops.push_back(classes_opout);
  this->output_ops.push_back(num_detections_opout);
  if (this->verbose) {
    std::cout << "Input Op Name: " << TF_OperationName(input_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(boxes_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(scores_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(classes_op) << "\n";
    std::cout << "Output Op Name: " << TF_OperationName(num_detections_op) << "\n";
  }
}

void ObjectDetection::preprocessing(IplImage* src, IplImage* dst) {
  cvCvtColor(src, dst, CV_BGR2RGB);
}

OD_Result ObjectDetection::sess_run(IplImage* img) {
  ResetOutputValues();
  // Create input variable
  int img_width = img->width;
  int img_height = img->height;
  int img_channel = img->nChannels;  
  const std::vector<std::int64_t> input_dims = {1, img_height, img_width, img_channel};
  int image_size_by_dims = img_height*img_width*img_channel;
  int image_tensor_size = std::min(image_size_by_dims, img->imageSize);
  if (this->verbose) {
    std::cout<<"image_tensor_size: "<<image_tensor_size<<std::endl;
  }
  TF_Tensor* input_value = CreateTensor(TF_UINT8,
                                        input_dims.data(), input_dims.size(),
                                        img->imageData, image_tensor_size);
  // TF_Tensor* input_values[1] = {input_value};
  input_values.emplace_back(input_value);
  // Create output variable
  const std::vector<std::int64_t> box_dims = {1, this->max_detections, 4};
  const std::vector<std::int64_t> scores_dims = {1, this->max_detections};
  const std::vector<std::int64_t> classes_dims = {1, this->max_detections};
  const std::vector<std::int64_t> num_detections_dims = {1, 1};
  TF_Tensor* boxes_value = TF_AllocateTensor(TF_FLOAT, box_dims.data(), box_dims.size(), sizeof(float) * 4 * this->max_detections);
  TF_Tensor* scores_value = TF_AllocateTensor(TF_FLOAT, scores_dims.data(), scores_dims.size(), sizeof(float) * this->max_detections);
  TF_Tensor* classes_value = TF_AllocateTensor(TF_FLOAT, classes_dims.data(), classes_dims.size(), sizeof(float) * this->max_detections);
  TF_Tensor* num_detections_value = TF_AllocateTensor(TF_FLOAT, num_detections_dims.data(), num_detections_dims.size(), sizeof(float));
  // TF_Tensor* output_values[4] = {boxes_value, scores_value, classes_value, num_detections_value};
  output_values.emplace_back(boxes_value);
  output_values.emplace_back(scores_value);
  output_values.emplace_back(classes_value);
  output_values.emplace_back(num_detections_value);
  if (this->verbose) {
    std::cout << "Input op info: " << TF_OperationNumInputs(input_op) << "\n";
    std::cout << "Input dims info: (" << TF_Dim(input_value, 0) <<", "<< TF_Dim(input_value, 1) <<", "\
                                      << TF_Dim(input_value, 2) <<", "<< TF_Dim(input_value, 3) <<")"<< "\n";
  }  
  const TF_Output* inputs_ptr = input_ops.empty() ? nullptr : &input_ops[0];
  TF_Tensor* const* input_values_ptr =
      input_values.empty() ? nullptr : &input_values[0];
  const TF_Output* outputs_ptr = output_ops.empty() ? nullptr : &output_ops[0];
  TF_Tensor** output_values_ptr =
      output_values.empty() ? nullptr : &output_values[0];
      
  // Create session
  TF_SessionRun(this->sess, nullptr,
                inputs_ptr, input_values_ptr, this->input_ops.size(),
                outputs_ptr, output_values_ptr, this->output_ops.size(),
                nullptr, 0, nullptr, this->sess_status);
  
  OD_Result od_result; 
  od_result.boxes = (float*)TF_TensorData(output_values[0]);
  od_result.scores = (float*)TF_TensorData(output_values[1]);
  od_result.label_ids = (float*)TF_TensorData(output_values[2]);
  od_result.num_detections = (float*)TF_TensorData(output_values[3]);
  TF_DeleteTensor(boxes_value);
  TF_DeleteTensor(scores_value);
  TF_DeleteTensor(classes_value);
  TF_DeleteTensor(num_detections_value);
  DeleteInputValues();
  return od_result;
}

OD_Result ObjectDetection::run(const char* img_path) {
  IplImage* img = cvLoadImage(img_path, CV_LOAD_IMAGE_COLOR);
  if (!img)
  {
    std::cout<<"Image load failed: "<<img_path<<std::endl;
    exit(1);
  }
  if (this->verbose) {
    std::cout<<"First pixel: (";
    // std::cout<<(uint8_t)img->imageData[0]<<", ";
    // std::cout<<(uint8_t)img->imageData[1]<<", ";
    // std::cout<<(uint8_t)img->imageData[2]<<")"<<std::endl;;
    std::cout<<unsigned((uint8_t)img->imageData[0])<<", ";
    std::cout<<unsigned((uint8_t)img->imageData[1])<<", ";
    std::cout<<unsigned((uint8_t)img->imageData[2])<<")"<<std::endl;;
    std::cout<<"img size: "<<img->width<<", "<<img->height<<", "<<img->nChannels<<std::endl;
    std::cout<<"img->depth: "<<img->depth<<std::endl;
    std::cout<<"img->imgSize: "<<img->imageSize<<std::endl;
    std::cout<<"img->width: "<<img->width<<std::endl;
    std::cout<<"img->height: "<<img->height<<std::endl;
    std::cout<<"img->nCHannels: "<<img->nChannels<<std::endl;
    std::cout<<"img->alphaChannel: "<<img->alphaChannel<<std::endl;
    std::cout<<"img->channelSeq : "<<img->channelSeq <<std::endl;
    std::cout<<"img->colorModel: "<<img->colorModel<<std::endl;
    std::cout<<"img->dataOrder: "<<img->dataOrder<<std::endl;
    std::cout<<"img->nSize: "<<img->nSize<<std::endl;
    std::cout<<"img->widthStep: "<<img->widthStep<<std::endl;
  }
  this->preprocessing(img, img);
  OD_Result od_result;
  od_result = this->sess_run(img);
  od_result = this->postprocessing(img, od_result);
  cvReleaseImage(&img);
  return od_result;
}

OD_Result ObjectDetection::postprocessing(IplImage* img, OD_Result od_result) {
  int img_width = img->width;
  int img_height = img->height;
  int img_channel = img->nChannels;  
  int num_detections = (int)od_result.num_detections[0];
  int box_cnt = 0; 
  for (int i=0; i<num_detections; i++) {
    if (od_result.scores[i] >= 0.5) {
      int xmin = (int)(od_result.boxes[i*4+1] * img_width);
      int ymin = (int)(od_result.boxes[i*4+0] * img_height);
      int xmax = (int)(od_result.boxes[i*4+3] * img_width);
      int ymax = (int)(od_result.boxes[i*4+2] * img_height);
      if (this->visible) {
        cvRectangle(img, cvPoint(xmin, ymin), cvPoint(xmax, ymax), CV_RGB(0, 255, 255));
      }
      std::cout<<"Box_"<<box_cnt<<"("<<od_result.scores[i]<<", "<<od_result.label_ids[i]<<"): ["<<xmin<<", "<<ymin<<", "<<xmax<<", "<<ymax<<"]"<<std::endl;
      box_cnt++;
    }
  }
  std::cout<<"Total box number: "<<box_cnt<<std::endl;
  if (this->visible) {
    cvShowImage("Drawing Graphics", img);
    cvWaitKey(0);
  }
  return od_result;
}

TF_Buffer* ObjectDetection::read_file(std::string path) {
  const char* file = path.c_str();
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

void ObjectDetection::DeleteInputValues() {
  for (size_t i = 0; i < input_values.size(); ++i) {
    if (input_values[i] != nullptr) TF_DeleteTensor(input_values[i]);
  }
  input_values.clear();
}

void ObjectDetection::ResetOutputValues() {
  for (size_t i = 0; i < output_values.size(); ++i) {
    if (output_values[i] != nullptr) TF_DeleteTensor(output_values[i]);
  }
  output_values.clear();
}

void ObjectDetection::close() {
  TF_CloseSession(this->sess, this->sess_status);
  TF_DeleteSession(this->sess, this->sess_status);
  TF_DeleteSessionOptions(this->sess_opts);
  TF_DeleteImportGraphDefOptions(this->graph_opts);
  TF_DeleteGraph(this->graph);
  TF_DeleteStatus(this->graph_status);
}

