#include "agents/AtomicAgent.hpp"

class FireFighterNaiveAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    // state: the current position of the agent and whether it has observed a fire (size 2)
    FireFighterNaiveAtomicAgentSimulator(): AtomicAgentSimulator() {
      LOG(INFO) << "Naive agent simulator has been built.";
    }
    int step(const std::vector<int>::iterator &it) {
      int action;
      // in the beginning, take a random action
      if (*it == 1) {
        action = 0;
        // action = std::experimental::randint(0,1);
      } else {
        // depends on the previous action and previous observation
        int prevAction = *(it + (*it-2));
        int prevObs = *(it + (*it-1));
        if (prevObs == 1) {
          action = prevAction;
        } else {
          action = 1 - prevAction;
        }
      }
      *(it + *it) = action;
      *it += 1;
      return action;
    }
};

class FireFighterNaiveAtomicAgent: public AtomicAgent, private FireFighterNaiveAtomicAgentSimulator {
  public:
    FireFighterNaiveAtomicAgent(const std::string &agentID, const int numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters): AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters), FireFighterNaiveAtomicAgentSimulator() {}
    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentsYAMLNode) {
      return FireFighterNaiveAtomicAgentSimulator::step(_AOH.begin());
    }
    void observe(int &observation) {
      FireFighterNaiveAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};