#include "agents/AtomicAgent.hpp"
#include <math.h>

class GridTrafficSimple2AtomicAgentSimulator: public AtomicAgentSimulator {
  public:
    GridTrafficSimple2AtomicAgentSimulator(int obsLength=2): AtomicAgentSimulator() {
      this->obsLength = obsLength;
      std::vector<std::vector<int>> binaryVectors;
      generateBinaryVectors(binaryVectors, 4*obsLength);
      for (auto &vec: binaryVectors) {
        bitMap[getExponentialSum(vec)] = vec;
      }  
    }
    int step(const std::vector<int>::iterator &it) {
      int action;
      if ((*it) == 1) {
        action = std::experimental::randint(0, 1);
      } else {
        int currentObs = *(it + (*it) - 1);
        std::vector<int> &bits = bitMap[currentObs];
        int hScore = 0;
        int vScore = 0;
        for (int i=0; i<=obsLength-1; i++) {
          hScore += bits[i];
          hScore -= bits[i+obsLength];
          vScore += bits[2*obsLength+i];
          vScore -= bits[3*obsLength+i];
        }
        if (hScore > vScore) {
          action = 0;
        } else if (vScore > hScore) {
          action = 1;
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
    std::map<int, std::vector<int>> bitMap;
    void generateBinaryVectors(std::vector<std::vector<int>> &vector, int numDigits) {
      if (numDigits == 0) {
        return;
      } else {
        if (vector.size() == 0) {
            vector.push_back(std::vector<int>({0}));
            vector.push_back(std::vector<int>({1}));
        } else {
            std::vector<std::vector<int>> newVector;
            for (auto &vec: vector) {
                std::vector<int> vec1 = vec;
                std::vector<int> vec2 = vec;
                vec1.push_back(0);
                vec2.push_back(1);
                newVector.push_back(vec1);
                newVector.push_back(vec2);
            }
            vector.clear();
            for (auto &vec: newVector) {
                vector.push_back(vec);
            }
        }
        generateBinaryVectors(vector, numDigits-1);
      }
    }
    int getExponentialSum(std::vector<int> &vec){
      int count = 0;
      int sum = 0;
      for (auto n: vec) {
        sum += n * std::pow(2, count);
        count += 1;
      }
      return sum;
    }
    int obsLength;
};

class GridTrafficSimple2AtomicAgent: public AtomicAgent, private GridTrafficSimple2AtomicAgentSimulator {
  public:
    GridTrafficSimple2AtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, const YAML::Node &parameters, int obsLength): GridTrafficSimple2AtomicAgentSimulator(obsLength), AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {

    }
    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentYAMLNode) {
      return GridTrafficSimple2AtomicAgentSimulator::step(_AOH.begin());
    }
    virtual void observe(int &observation) {
      GridTrafficSimple2AtomicAgentSimulator::observe(_AOH.begin(), observation);
    }
};
