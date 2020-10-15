#include <iostream>
#include <cassert>
#include <filesystem>
#include "glog/logging.h"
#include <stdlib.h>
#include <ctime>
#include <memory>
#include "runners/Experiment.hpp"
#include "runners/DataGenerationExperiment.hpp"
namespace fs = std::filesystem;

bool runExperiment(std::string typeOfExperiment, std::string pathToConfigurationFile, std::string pathToResultsFolder){
  std::unique_ptr<Experiment> experiment;
  if (typeOfExperiment == "Testing"){
    experiment = std::unique_ptr<Experiment>(new TestingExperiment(pathToConfigurationFile, pathToResultsFolder));
  } else if (typeOfExperiment == "Planning"){
    experiment = std::unique_ptr<Experiment>(new PlanningExperiment(pathToConfigurationFile, pathToResultsFolder));
  } else if (typeOfExperiment == "DataGeneration"){
    experiment = std::unique_ptr<Experiment>(new DataGenerationExperiment(pathToConfigurationFile, pathToResultsFolder));
  } else {
    LOG(FATAL) << "Error: Experiment type not supported.";
    return false;
  }
  return experiment->run();
}

int main(int argc, char** argv){

  if (argc != 4) {
    std::cerr << "Three arguments are required for lanuching an experiment." << std::endl;
    std::cerr << "1. typeOfExperiment" << std::endl;
    std::cerr << "2. pathToConfigurationFile" << std::endl;
    std::cerr << "3. pathToResultsFolder" << std::endl;
    return 1;
  }

  // the type of the experiment to run
  std::string typeOfExperiment = argv[1];

  // the path to the configuration file
  std::string pathToConfigurationFile = argv[2];

  // the path to the folder where we store experiment results
  std::string pathToResultsFolder = argv[3];

  // create the folder to store results if not already existing
  fs::path pathToLogsFolder = pathToResultsFolder + "/logs/";
  fs::create_directories(pathToLogsFolder);

  fs::path pathToReplayFolder = pathToResultsFolder + "/replays/";
  fs::create_directories(pathToReplayFolder);

  // set up logging
  google::InitGoogleLogging(argv[0]);
  google::SetLogDestination(google::GLOG_INFO, pathToLogsFolder.c_str());
  FLAGS_alsologtostderr = 1;

  LOG(INFO) << "--------------------------------------------------";
  LOG(INFO) << "Program started.";
  LOG(INFO) << "Type of the experiment: " << typeOfExperiment;
  LOG(INFO) << "Path to configuration file: " << pathToConfigurationFile;
  LOG(INFO) << "Path to results: " << pathToResultsFolder;
  LOG(INFO) << "Path to logs: " << pathToLogsFolder;
  LOG(INFO) << "--------------------------------------------------";

  clock_t begin = std::clock();

  auto status = runExperiment(typeOfExperiment, pathToConfigurationFile, pathToResultsFolder);

  clock_t end = std::clock();
  double elapsed_seconds = double(end-begin) / CLOCKS_PER_SEC;

  LOG(INFO) << "Elapsed time: " << elapsed_seconds << " seconds.";

  if (status == true){
    LOG(INFO) << "Experiment exited successfully.";
  } else {
    LOG(FATAL) << "Experiment failed.";
  }

  google::ShutdownGoogleLogging();

  return 0;
}
