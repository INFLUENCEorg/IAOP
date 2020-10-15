#include "agents/AtomicAgent.hpp"

class GrabAChairPatternAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    GrabAChairPatternAtomicAgentSimulator(int frequency): AtomicAgentSimulator() {
      _freq = frequency;
    }
    int step(const std::vector<int>::iterator &it) {
      int action;
      if (*it == 1) {
        action = 0; // always start with left
      } else {
        if ((*it - 1) % _freq == 0) {
          action = 1 - _prevAction;
        } else {
          action = _prevAction;
        }
      }
      *(it + *it) = action;
      *it += 1;
      _prevAction = action;
      return action;
    }
  private:
    int _prevAction;
    int _freq;
};

class GrabAChairPatternAtomicAgent: public AtomicAgent, private GrabAChairPatternAtomicAgentSimulator {
  public:
    GrabAChairPatternAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters, int freq): GrabAChairPatternAtomicAgentSimulator(freq), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {

    }
    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentYAMLNode) {
      return GrabAChairPatternAtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int &observation) {
      GrabAChairPatternAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};

class GrabAChairCountBasedAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    GrabAChairCountBasedAtomicAgentSimulator(int memorySize=-1): AtomicAgentSimulator(), _memorySize(memorySize){}
    int step(const std::vector<int>::iterator &it) {

      int action;

      if (_memorySize == 0) {
        // no memory means fully random
        action = std::experimental::randint(0, 1);
      } else {
        int startingPoint;
        double sum0 = 0.0;
        double sum1 = 0.0;
        int count0 = 0;
        int count1 = 0;
        if (_memorySize == -1) {
          startingPoint = 1; // starting from the beginning
        } else {
          startingPoint = (*it) - _memorySize*2;
          if (startingPoint < 1) {
            startingPoint = 1;
          }
        }
        for (int i=startingPoint; i<(*it); i+=2) {
          int action = *(it + i);
          int obs = *(it + i + 1);
          if (action == 0) {
            count0 += 1;
            sum0 += obs;
          } else {
            count1 += 1;
            sum1 += obs;
          }
        }

        double avg0 = this->getAverage(sum0, count0);
        double avg1 = this->getAverage(sum1, count1);

        if (avg1 > avg0) {
          action = 1;
        } else if (avg1 < avg0) {
          action = 0;
        } else {
          action = std::experimental::randint(0, 1);
        }
      }

      *(it + *it) = action;
      *it += 1;
      return action;
    }
  protected:
    std::default_random_engine rg;
    int _memorySize;

    // if count = 0 return 0 otherwise return sum/count
    double getAverage(double sum, int count) {
      if (count == 0){
        return 1000000; // a large number 
      } else {
        return sum/count;
      }
    }
};

class GrabAChairCountBasedAtomicAgent: public AtomicAgent, private GrabAChairCountBasedAtomicAgentSimulator {
  public:
    GrabAChairCountBasedAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters): GrabAChairCountBasedAtomicAgentSimulator(parameters["memorySize"].as<int>()), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {

    }
    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentYAMLNode) {
      return GrabAChairCountBasedAtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int &observation) {
      GrabAChairCountBasedAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};

class GrabAChairHappyAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    GrabAChairHappyAtomicAgentSimulator(): AtomicAgentSimulator() {}
    int step(const std::vector<int>::iterator &it) {
      int action; 
      if (*it == 1) {
        action = std::experimental::randint(0,1);
      } else {
        int prevAction = *(it + (*it)-2);
        int prevObs = *(it + (*it)-1);

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

class GrabAChairHappyAtomicAgent: public AtomicAgent, private GrabAChairHappyAtomicAgentSimulator {
  public:
    GrabAChairHappyAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters): GrabAChairHappyAtomicAgentSimulator(), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {

    }
    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentYAMLNode) {
      return GrabAChairHappyAtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int &observation) {
      GrabAChairHappyAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};

class GrabAChairSadAtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    GrabAChairSadAtomicAgentSimulator(): AtomicAgentSimulator() {}
    int step(const std::vector<int>::iterator &it) {
      int action; 
      if (*it == 1) {
        action = std::experimental::randint(0,1);
      } else {
        int prevAction = *(it + (*it)-2);
        int prevObs = *(it + (*it)-1);

        if (prevObs == 1) {
          action = 1 - prevAction;
        } else {
          action = prevAction;
        }
      }
    
      *(it + *it) = action;
      *it += 1;
      return action;
    }
};

class GrabAChairSadAtomicAgent: public AtomicAgent, private GrabAChairSadAtomicAgentSimulator {
  public:
    GrabAChairSadAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters): GrabAChairSadAtomicAgentSimulator(), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {

    }
    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentYAMLNode) {
      return GrabAChairSadAtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int &observation) {
      GrabAChairSadAtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};