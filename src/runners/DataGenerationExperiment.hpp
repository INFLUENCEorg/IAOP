// data generation for influence predictor training
// can we directly generate data that can be readily used for learning?
// for example, numpy data, or other data that can be quickly read
// will try to use pickle_save 

#include "Experiment.hpp"
#include <torch/torch.h>
#include <iostream>
#include "Utils.hpp"

class DataGenerationExperiment: public Experiment {

  public:

    DataGenerationExperiment(std::string pathToConfigurationFile, std::string pathToResultsFolder):Experiment(pathToConfigurationFile, pathToResultsFolder){};

    bool run(){
      
      // read configurations
      std::string domainName = parameters["General"]["domain"].as<std::string>();
      int horizon = parameters["General"]["horizon"].as<int>();
      std::string agentID = parameters["General"]["IDOfAgentToControl"].as<std::string>();
      int numOfRepeats = parameters["AgentComponent"][agentID]["Simulator"]["InfluencePredictor"]["numberOfSampledEpisodesForTraining"].as<int>();

      // set up the domain
      Domain *domainPtr = makeDomain(domainName);
      int numberOfActions = domainPtr->getAgentsSpecification()[agentID];

      // set up the global simulator
      Domain::SingleAgentGlobalSimulator *globalSimulatorPtr = new Domain::SingleAgentGlobalSimulator(
        agentID,
        domainPtr,
        parameters["AgentComponent"]
      );

      // create placeholder tensors for inputs and outputs
      std::vector<std::string> localStates;
      std::vector<std::string> localFactors;
      std::vector<std::string> influenceSourceStates;
      std::vector<std::string> influenceDestinationStates;
      std::vector<std::string> localStatesActions;
      domainPtr->getDBNPtr()->constructLocalModel(agentID, localFactors, localStates, influenceSourceStates, influenceDestinationStates, localStatesActions);
      
      int sizeOfInputs = localStatesActions.size();
      int sizeOfOutputs = influenceSourceStates.size();
      auto inputs = torch::zeros({numOfRepeats, horizon-1, sizeOfInputs}, torch::TensorOptions().dtype(torch::kInt32));
      auto outputs = torch::zeros({numOfRepeats, horizon-1, sizeOfOutputs}, torch::TensorOptions().dtype(torch::kInt32));

      LOG(INFO) << "[Influence Predictor Training] inputs: " << PrintUtils::vectorToTupleString(localFactors);
      LOG(INFO) << "[Influence Predictor Training] size of inputs: " << inputs.sizes();
      LOG(INFO) << "[Influence Predictor Training] outputs: " << PrintUtils::vectorToTupleString(influenceSourceStates);
      LOG(INFO) << "[Influence Predictor Training] size of outputs: " << outputs.sizes();

      // data collection
      int observation;
      float reward;
      bool done;
      for (int i=0; i<=numOfRepeats-1; i++) {
        // sample one state
        auto state = globalSimulatorPtr->sampleInitialState();
        // do the trajectory simulation
        for (int step=0; step<=horizon-2; step++) {
          int action = std::experimental::randint(0, numberOfActions-1);
          globalSimulatorPtr->step(state, action, observation, reward, done);
          // extract local states and actions and influence sources
          for (int j=0; j<=localStates.size()-1; j++){
            inputs[i][step][j] = state.environmentState[localStates[j]];
          }
          inputs[i][step][localStates.size()] = action;
          // outputs
          for (int j=0; j<=influenceSourceStates.size()-1; j++) {
            outputs[i][step][j] = state.environmentState[influenceSourceStates[j]];
          }
        }
      }

      // save data
      torch::save(inputs, pathToResultsFolder+"/inputs.pt");
      torch::save(outputs, pathToResultsFolder+"/outputs.pt");

      delete domainPtr;
      delete globalSimulatorPtr;

      return true;
  }

};