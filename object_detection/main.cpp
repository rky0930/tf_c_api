#include <iostream>
#include <getopt.h>
#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>
#include "object_detection.hpp"

void help(char *argv);
void run(ObjectDetection* object_detection, std::string image_path);
void close(const char *s);

int main (int argc, char **argv)
{
  int verbose_flag = 0;
  int visible_flag = 0;
  std::string image_path = "";
  std::string image_dir_path = "";
  std::string frozen_graph_path = "";
  float confidence_score_threshold = 0.5;
  int max_detections = 100;
  int c;
  while (1) {
    static struct option long_options[] =
      {
        {"frozen_graph_path", required_argument, 0, 'f'},
        {"image_path", required_argument, 0, 'i'},          
        {"confidence_score_threshold", required_argument, 0, 'c'},
        {"max_detections", required_argument, 0, 'm'},
        {"verbose", no_argument, 0, 'v'},
        {"show", no_argument, 0, 's'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
      };
    /* getopt_long stores the option index here. */
    int option_index = 0;
    c = getopt_long (argc, argv, "f:i:c:m:vsh",
                      long_options, &option_index);
    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c)
    {
      case 'f':
        frozen_graph_path = std::string(optarg);
        break;
      case 'i':
        image_path = std::string(optarg);
        break;
      case 'c':
        confidence_score_threshold = atof(optarg);
        break;
      case 'm':
        max_detections = atoi(optarg);
        break;
      case 'v':
        verbose_flag = 1;
        break;
      case 's':
        visible_flag = 1;
        break;
      case 'h':
        help(argv[0]);
        break;
      case '?':
        help(argv[0]);
        break;
      default:
        help(argv[0]);
    }
  }
    if (frozen_graph_path == "") {
      std::cerr<<"Please set --frozen_graph_path(-f)"<<std::endl;
      exit(1);
    }
    if (image_path == "") {
      std::cerr<<"Please set --image_path(-i)"<<std::endl;
      exit(1);
    }
    std::cout<<verbose_flag<<", "<<visible_flag<<std::endl;
    ObjectDetection object_detection(frozen_graph_path, confidence_score_threshold, max_detections);
    object_detection.set_verbose_mode(verbose_flag);
    object_detection.set_visible_mode(visible_flag);
    object_detection.init();
    run(&object_detection, image_path);
    return 0;
}

void help(char *argv) {
    std::cout<<"Usage : "<<argv<<" --(f)rozen_graph_path --(i)mage_path [--(c)onfidence_score_threshold] [--(m)ax_detections] [--(v)erbose] [--(s)how]"<<std::endl;
    exit(0);
}

void run(ObjectDetection* object_detection, std::string image_path) {
  struct stat st;
  if (stat(image_path.c_str(), &st) < 0) close("stat");
  if (!S_ISDIR(st.st_mode)) {
    // File
    std::cout<<image_path<<std::endl;
    object_detection->run(image_path.c_str());
  }else{
    // Directory
    DIR *d;
    struct dirent *ent;
    d = opendir(image_path.c_str());
    while (ent = readdir(d)) {
      if (strcmp(ent->d_name, ".") == 0) continue;
      if (strcmp(ent->d_name, "..") == 0) continue;
      // std::cout<<image_path.c_str()<<", "<<ent->d_name<<std::endl;

      std::string full_path = image_path;
      if (full_path.back() != '/') {
        full_path.append("/");
      }
      full_path.append(ent->d_name);
      std::cout<<full_path<<std::endl;
      object_detection->run(full_path.c_str());
    }
  }
}

void close(const char *s) {
    perror(s);
    exit(1);
}
