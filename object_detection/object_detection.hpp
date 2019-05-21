#include <iostream>
#include <opencv2/opencv.hpp>
#include "tensorflow/c/c_api.h"

struct OD_Result {
  float* boxes;
  float* scores;
  float* label_ids;
  float* num_detections;
};

class ObjectDetection {
  private:
    std::string frozen_graph_path="";
    float confidence_score_threshold;
    int max_detections;
    TF_Graph* graph;
    TF_Buffer* graph_def;
    TF_ImportGraphDefOptions* graph_opts;
    TF_Status* graph_status;
    TF_SessionOptions* sess_opts = TF_NewSessionOptions();
    TF_Status* sess_status;
    TF_Session* sess;  
    TF_Operation* input_op;
    TF_Output input_opout;
    std::vector<TF_Output> input_ops;
    std::vector<TF_Tensor*> input_values;
    TF_Operation* boxes_op;
    TF_Operation* scores_op;
    TF_Operation* classes_op;
    TF_Operation* num_detections_op;
    TF_Output boxes_opout, scores_opout, classes_opout, num_detections_opout;
    std::vector<TF_Output> output_ops;
    std::vector<TF_Tensor*> output_values;
    
    int verbose = 0;
    int visible = 0;
  public:
    ObjectDetection(std::string frozen_graph_path, float confidence_score_threshold, 
                    int max_detections);
    ~ObjectDetection() { close(); }
    void init();
    void set_graph();
    void set_froze_graph_path(std::string path) { frozen_graph_path = path; }
    void set_verbose_mode(bool mode) { verbose = mode; }
    void set_visible_mode(bool mode) { visible = mode; }
    void preprocessing(IplImage* src, IplImage* dst);
    OD_Result sess_run(IplImage* img);
    OD_Result run(const char* img_path);
    OD_Result postprocessing(IplImage* src, OD_Result od_result);
    TF_Buffer* read_file(std::string path);
    void DeleteInputValues();
    void ResetOutputValues();
    void close();
};