#include "glog/logging.h"
#include "agents/AtomicAgent.hpp"
#include <memory>
#include "yaml-cpp/yaml.h"
#include <ctime>

// an abstract class
class AgentComponent {
  public:
    AgentComponent() {

    }
    virtual ~AgentComponent() {
      
    }
    virtual void reset() = 0;
    virtual void act(std::map<std::string, int> &action, std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentsYAMLNode) = 0;
    virtual void observe(std::map<std::string, int> &observation) = 0;
};

// an agent component which consists of only atomic agents
class SimpleAgentComponent: public AgentComponent {
  public:
    SimpleAgentComponent(std::map<std::string, AtomicAgent*> &atomicAgents): AgentComponent() {
      for (auto &[agentID, agentPtr]: atomicAgents) {
        _atomicAgents[agentID] = std::unique_ptr<AtomicAgent>(agentPtr);
      }
      LOG(INFO) << "A simple Agent component consisting of " << atomicAgents.size() << " agents has been built.";
    }
    ~SimpleAgentComponent() {
      
    }

    // the agent component resets the interal states of the atomic agents
    void reset() {
      VLOG(1) << "Resetting agent component.";
      for (const auto &[key, val]: _atomicAgents) {
        val->reset();
      }
    }

    // the agent component picks a joint action according to the interal states of the atomic agents
    void act(std::map<std::string, int> &action, std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentsYAMLNode) {
      for (const auto &[key, val]: _atomicAgents) {
        auto begin = clock();
        action[key] = val->act(results, agentsYAMLNode);
        float elapsed_secs = double(clock()-begin) / CLOCKS_PER_SEC;
        results["time_per_action"][key].push_back(elapsed_secs);
      }
    }
    
    // the agent component receives the joint observation from the environment
    void observe(std::map<std::string, int> &observation) {
      for (const auto &[key, val]: _atomicAgents) {
        val->observe(observation[key]);
      }
    }
private:
    std::map<std::string, std::unique_ptr<AtomicAgent>> _atomicAgents;
};

