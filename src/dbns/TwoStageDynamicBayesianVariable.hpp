#ifndef TWO_STAGE_DYNAMIC_BAYESIAN_VARIABLE_HPP_
#define TWO_STAGE_DYNAMIC_BAYESIAN_VARIABLE_HPP_

#include<cmath>
#include <experimental/random>

#define CPT 0
#define SUM 1
#define EXPSUM 2
#define NOISYEXPSUM 3

class TwoStageDynamicBayesianNetworkVariable {
  public:
    std::string name;

    TwoStageDynamicBayesianNetworkVariable(std::string name, const YAML::Node &info, std::default_random_engine *randomNumberGeneratorPtr){
      this->name = name;
      _listOfParents = info["parents"].as<std::vector<std::string>>();
      if (info["values"].IsDefined()) {
        _listOfValues = info["values"].as<std::vector<float>>();
      }
      _numberOfValues = _listOfValues.size();

      if (_listOfParents.size() != 0) {
        _mode = CPT;
        if (info["mode"].IsDefined() == true) {
          std::string modeStr = info["mode"].as<std::string>();
          if (modeStr == "SUM") {
            _mode = SUM;
          } else if (modeStr == "EXPSUM") {
            _mode = EXPSUM;
          } else if (modeStr == "NOISYEXPSUM") {
            _mode = NOISYEXPSUM;
          } else {
            _mode = CPT;
          }
        }
        if (_mode == CPT) {
          YAML::Node conditionalProbabilityTable = info["CPT"];
          for (YAML::const_iterator it = conditionalProbabilityTable.begin(); it != conditionalProbabilityTable.end(); ++it){
            std::vector<int> conditionalIndices = it->first.as<std::vector<int>>();
            std::vector<float> probabilities = it->second.as<std::vector<float>>();
            _conditionalProbabilityDistribution[conditionalIndices] = std::unique_ptr<std::discrete_distribution<int>>(new std::discrete_distribution<int>(probabilities.begin(), probabilities.end()));
          }
        } else if (_mode == EXPSUM) {
          _expSumBase = info["EXPSUM"]["base"].as<int>();
        } else if (_mode == NOISYEXPSUM) {
          _expSumBase = info["NOISYEXPSUM"]["base"].as<int>();
          _noise = info["NOISYEXPSUM"]["noise"].as<float>();
        }
      }

      if (name[0] == 'x' && StringUtils::lastBitIsPrime(name) == false) {
        this->_isStateVariable = true;
      } 

      if (info["initial_dist"].IsDefined() == true) {
        LOG(INFO) << name;
        auto initialProbabilities = info["initial_dist"].as<std::vector<float>>();
        _initialDist = std::unique_ptr<std::discrete_distribution<int>>(new std::discrete_distribution<int>(initialProbabilities.begin(), initialProbabilities.end()));
      }

      this->_randomNumberGeneratorPtr = randomNumberGeneratorPtr;
    }

    std::vector<std::string> &getListOfParents(){
      return _listOfParents;
    }

    int getNumberOfInputs(){
      return _listOfParents.size();
    }

    int getNumberOfValues(){
      return _numberOfValues;
    }

    int sampleInitialValue() {
      return (*_initialDist)(*_randomNumberGeneratorPtr);
    }

    int sample(std::vector<int> &inputs){
      int index;
      if (_mode == CPT) {
        std::discrete_distribution<int> &distribution = *(_conditionalProbabilityDistribution[inputs]);
        index = distribution(*_randomNumberGeneratorPtr);
      } else if (_mode == SUM) {
        index = 0;
        for (auto &in: inputs) {
          index += in;
        }
      } else if (_mode == EXPSUM) {
        index = 0;
        for (int i=0; i<=inputs.size()-1; i++) {
          index += std::pow(_expSumBase, i) * inputs[i];
        } 
      } else if (_mode == NOISYEXPSUM) {
        index = 0;
        int v;
        for (int i=0; i<=inputs.size()-1; i++) {
          float r = 1.0 * std::experimental::randint(0, 9) / 10;
          if (r < _noise) {
            v = 1 - inputs[i];
          } else {
            v = inputs[i];
          }
          index += std::pow(_expSumBase, i) * v;
        }
      }
      
      return index;
    }

    int sampleUniformly() {
      return std::experimental::randint(0, _numberOfValues-1);
    }

    float getValueFromIndex(const int index){
      if (_listOfValues.size() != 0) {
        return _listOfValues[index];
      } else {
        return index;
      }
    }

    // check whether the last digit is '
    bool isFromPreviousStage() {
      LOG(INFO) << name.back();
      return name.back() != 39;
    }

    bool isStateVariable() {
      return _isStateVariable;
    }

  private:
    std::vector<std::string> _listOfParents;
    std::vector<float> _listOfValues;
    std::unique_ptr<std::discrete_distribution<int>> _initialDist;
    bool _isStateVariable = false;
    int _numberOfValues;
    std::map<std::vector<int>, std::unique_ptr<std::discrete_distribution<int>>> _conditionalProbabilityDistribution;
    int _expSumBase;
    float _noise;
    std::default_random_engine *_randomNumberGeneratorPtr;
    int _mode;
};

#endif