#include <map>
#include "domains/Domain.hpp"
#include "memory"

// episode encodes how a markov multiagent environment and an agentcomponent interact with each other
class Episode {

  public:

    Episode(int episodeID, Domain::Environment *environmentPtr, AgentComponent *agentComponentPtr, int horizon, float discountFactor, std::string &pathToResultsFolder, bool saveReplay = false): _pathToResultsFolder(pathToResultsFolder) {
      this->_environmentPtr = environmentPtr;
      this->_agentComponentPtr = agentComponentPtr;
      _horizon = horizon;
      _discountFactor = discountFactor;
      _saveReplay = saveReplay;
      _episodeID = episodeID;
      _pathToReplaysFolder = _pathToResultsFolder + "/replays/";
    }

    std::map<std::string, std::map<std::string, std::vector<double>>> dispatch(){
      LOG(INFO) << "--------------------------------------------------";
      LOG(INFO) << "Episode " << std::to_string(_episodeID) << " has been dispatched.";

      // timing code for debugging
      double actTime = 0.0;
      double stepTime = 0.0;
      double observeTime = 0.0;

      std::map<std::string, std::map<std::string, std::vector<double>>> results;
      _agentComponentPtr->reset();
      _environmentPtr->reset();
      std::map<std::string, int> action;
      std::map<std::string, int> observation;
      std::map<std::string, float> reward;
      bool done = false;
      float factor = 1.0;
      for (int i=0; i<=_horizon-1; i++) {
        std::string stepID = std::to_string(i);
        
        YAML::Node agentsYAMLNode;
        if (_saveReplay == true) {
          _replay[stepID]["state"] = _environmentPtr->getState();
          agentsYAMLNode["save"] = true;
        } else {
          agentsYAMLNode["save"] = false;
        }

        auto begin = std::clock();
        _agentComponentPtr->act(action, results, agentsYAMLNode);
        actTime += (double)(std::clock()-begin)/CLOCKS_PER_SEC;
        begin = std::clock();
        // LOG(INFO) << PrintUtils::mapToTupleString(_environmentPtr->getState());
        _environmentPtr->step(action, observation, reward, done);
        stepTime += (double)(std::clock()-begin)/CLOCKS_PER_SEC;
        begin = std::clock();
        // LOG(INFO) << "true: " << PrintUtils::mapToTupleString(_environmentPtr->getState());
        _agentComponentPtr->observe(observation);
        observeTime += (double)(std::clock()-begin)/CLOCKS_PER_SEC;

        for (auto &[key, val]: reward) {
          float discounted_reward = factor * val;
          if (results["episodic_return"].count(key) == 0){
            results["episodic_return"][key].push_back(val);
            results["discounted_episodic_return"][key].push_back(discounted_reward);
          } else {
            results["episodic_return"][key][0] += val;
            results["discounted_episodic_return"][key][0] += discounted_reward;
          }
        }

        // LOG(INFO) << "[step" << std::to_string(i) << "] reward of agent 1: " << std::to_string(int(reward["1"]));

        if (_saveReplay == true) {
          _replay[stepID]["action"] = action;
          _replay[stepID]["observation"] = observation;
          _replay[stepID]["reward"] = reward;
          _replay[stepID]["agents"] = agentsYAMLNode;
        }

        if (done == true) {
          break;
        }

        factor *= _discountFactor;
        // LOG(INFO) << std::to_string(factor);
      }
      if (_saveReplay == true) {
        std::ofstream fout(_pathToReplaysFolder+"episode"+std::to_string(_episodeID)+".yaml");
        fout << _replay;
      }
      VLOG(2) << "act time in total: " << std::to_string(actTime);
      VLOG(2) << "step time in total: " << std::to_string(stepTime);
      VLOG(2) << "observe time in total: " << std::to_string(observeTime);
      return results;
    }

    virtual ~Episode(){
      
    };

  private:
    Domain::Environment *_environmentPtr;
    AgentComponent *_agentComponentPtr;
    std::string &_pathToResultsFolder;
    int _horizon;
    bool _saveReplay;
    int _episodeID;
    float _discountFactor;
    std::string _pathToReplaysFolder;
    YAML::Node _replay;
};
