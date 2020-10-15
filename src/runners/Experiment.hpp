#ifndef EXPERIMENT_HPP_
#define EXPERIMENT_HPP_

#include <fstream>
#include <map>
#include <any>
#include "yaml-cpp/yaml.h"
#include "omp.h"
#include "domains/Domain.hpp"
#include "domains/GridTraffic/GridTrafficDomain.hpp"
#include "domains/GrabAChair/GrabAChairDomain.hpp"
#include "domains/FireFighter/FireFighterDomain.hpp"
#include "agents/AgentComponent.hpp"
#include "agents/AtomicAgent.hpp"
#include "agents/PlanningAgent.hpp"
#include "Episode.hpp"

class Experiment {

public:

    Experiment(std::string pathToConfigurationFile, std::string pathToResultsFolder){
        this->pathToResultsFolder = pathToResultsFolder;
        parameters = YAML::LoadFile(pathToConfigurationFile);
        LOG(INFO) << "\n-------------Experimental Parameters:-------------\n" << parameters << "\n--------------------------------------------------\n";
    }

    virtual bool run(){
        std::cerr << "Error: Experiment Not Implemented." << std::endl;
        return false;
    }

    virtual ~Experiment(){
    }

    Domain *makeDomain(const std::string &domain) {
      if (domain == "GridTraffic") {
        return new GridTrafficDomain(parameters);
      } else if (domain == "GrabAChair") {
        return new GrabAChairDomain(parameters);
      } else if (domain == "FireFighter") {
        return new FireFighterDomain(parameters);
      } else {
        std::string message = "domain " + domain + " is not supported.";
        throw std::invalid_argument(message);
      }
    }

    AgentComponent *makeAgentComponent(Domain *domainPtr) {
      std::map<std::string, AtomicAgent*> atomicAgents;
      for (auto &[agentID, numberOfActions]: domainPtr->getAgentsSpecification()) {
        std::string agentType = parameters["AgentComponent"][agentID]["Type"].as<std::string>();
        atomicAgents[agentID] = domainPtr->makeAtomicAgent(agentID, agentType);
      }
      return new SimpleAgentComponent(atomicAgents);
    }

protected:
    std::string pathToResultsFolder;
    YAML::Node parameters;
};

class TestingExperiment: public Experiment {

public:

    TestingExperiment(std::string pathToConfigurationFile, std::string pathToResultsFolder):Experiment(pathToConfigurationFile, pathToResultsFolder){};

    bool run(){
        LOG(INFO) << "Testing Experiment Finished.";
        return true;
    }

};

class PlanningExperiment: public Experiment {

  public:

    PlanningExperiment(std::string pathToConfigurationFile, std::string pathToResultsFolder):Experiment(pathToConfigurationFile, pathToResultsFolder){};

    ~PlanningExperiment() {
    }

    bool run(){

      bool fullLogging = parameters["Experiment"]["fullLogging"].as<bool>();
      std::string IDOfAgentToControl = parameters["General"]["IDOfAgentToControl"].as<std::string>();

      // create the csv file for experimental results
      YAML::Node resultsYAML;

      std::string domainName = parameters["General"]["domain"].as<std::string>();

      int numOfRepeats = parameters["Experiment"]["repeat"].as<int>();

      Domain *domainPtr = makeDomain(domainName);
      // create an agent component
      AgentComponent *agentComponentPtr = makeAgentComponent(domainPtr);
      // create an environment
      Domain::Environment *environmentPtr = domainPtr->makeEnvironment();

      // to store moving average of episodic returns
      std::vector<float> movingAverages;
      std::vector<float> secondPerAction;

      for (int i=0; i<=numOfRepeats-1; i++){
        
        resultsYAML[std::to_string(i)] = YAML::Node();

        std::string prefix = "[Episode " + std::to_string(i) + "] ";
        // run an episode
        Episode episode = Episode(i, environmentPtr, agentComponentPtr, parameters["General"]["horizon"].as<int>(), parameters["General"]["discountFactor"].as<float>(), pathToResultsFolder, parameters["Experiment"]["saveReplay"].as<bool>());
        // auto results = episode.run();
        auto results = episode.dispatch();
        // logging
        for (int j=0; j<=domainPtr->getNumberOfAgents()-1; j++){
          auto &agentID = domainPtr->getListOfAgentIDs()[j];
          float episodic_return = results["discounted_episodic_return"][agentID][0];

          if ((fullLogging == true) || (agentID == IDOfAgentToControl)) {
            resultsYAML[i][agentID]["Return"] = episodic_return;
            resultsYAML[i][agentID]["Times"] = results["time_per_action"][agentID];
            if (agentID == IDOfAgentToControl) {
              resultsYAML[i][agentID]["Num_simulations"] = results["number_of_simulations_per_step"][agentID];
              resultsYAML[i][agentID]["Num_particles"] = results["number_of_particles_before_simulation"][agentID];
            }
          }
          
          std::string returnMessage;
          returnMessage =  prefix + "Agent " + agentID + " Discounted Episodic Return: " + std::to_string(episodic_return);
            
          std::string movingAvgMessage;
          if (movingAverages.size() < domainPtr->getNumberOfAgents()) {
            movingAverages.push_back(episodic_return);
          } else {
            movingAverages[j] = (movingAverages[j] * i + episodic_return) / (i+1);
          }
          movingAvgMessage = prefix + "Agent " + agentID + " Moving Average of discounted returns: " + std::to_string(movingAverages[j]);

          std::string timeMessage; 
          timeMessage = prefix + "Agent " + agentID + " Average Decision Making Time Per Step: ";
          double totalTime = 0.0;
          for (auto &time: results["time_per_action"][agentID]) {
            totalTime += time;
          }
          timeMessage += std::to_string(totalTime / results["time_per_action"][agentID].size());

          std::string simMessage;
          simMessage = prefix + "Agent " + agentID + " Number of simulations Per Step: ";
          double totalSims = 0.0;
          for (auto &sim: results["number_of_simulations_per_step"][agentID]) {
            totalSims += sim;
          }
          simMessage += std::to_string(totalSims / results["number_of_simulations_per_step"][agentID].size());

          std::string particleMessage;
          particleMessage = prefix + "Agent " + agentID + " Number of particles before simulation Per Step: ";
          double totalParticles = 0.0;
          for (auto &particle: results["number_of_particles_before_simulation"][agentID]) {
            totalParticles += particle;
          }
          particleMessage += std::to_string(totalParticles / results["number_of_particles_before_simulation"][agentID].size());

          if (agentID == IDOfAgentToControl) {
            LOG(INFO) << returnMessage;
            LOG(INFO) << movingAvgMessage;
            LOG(INFO) << timeMessage;
            LOG(INFO) << simMessage;
            LOG(INFO) << particleMessage;
          } else {
            VLOG(1) << returnMessage;
            VLOG(1) << movingAvgMessage;
            VLOG(1) << timeMessage;
            VLOG(1) << simMessage;
            VLOG(1) << particleMessage;
          }
          
        }



      }

      std::ofstream resultsYAMLFile;
      resultsYAMLFile.open(pathToResultsFolder+"/results.yaml");
      resultsYAMLFile << resultsYAML;
      resultsYAMLFile.close();

      delete domainPtr;
      delete agentComponentPtr;
      delete environmentPtr;

      return true;
  }

};

#endif