#ifndef TWO_STAGE_DYNAMIC_BAYESIAN_NETWORK_HPP_
#define TWO_STAGE_DYNAMIC_BAYESIAN_NETWORK_HPP_

#include "TwoStageDynamicBayesianVariable.hpp"
#include <iostream>
#include <random>
#include "yaml-cpp/yaml.h"
#include "glog/logging.h"
#include <memory>
#include "Utils.hpp"
#include <algorithm>
#include <json.hpp>
#include <fstream>

// the two stage dynamic bayesian network
class TwoStageDynamicBayesianNetwork {
  public:
    std::set<std::string> getStateFactorsNextStage(std::set<std::string> factors) {
      std::set<std::string> set;
      for (auto &factor: factors) {
        if (factor[0] != 'a') {
          set.insert(StringUtils::addLastPrime(factor));
        }
      }
      return set;
    }

    void constructLocalModel(const std::string &agentID, std::vector<std::string> &localFactors, std::vector<std::string> &localStates, std::vector<std::string> &sourceFactors, std::vector<std::string> &destinationFactors, std::vector<std::string> &dSeparationSetPerStage) {
      
      std::set<std::string> _localFactors;
      std::set<std::string> _sourceFactors;
      std::set<std::string> _destinationFactors;
      std::set<std::string> _dSeparationSetPerStage; // the final d separation set is a history of this set
      
      std::string observationFactorName = "o" + agentID;
      std::string rewardFactorName = "r" + agentID;
      auto &observationFactorPtr = _twoStageDynamicBayesianNetworkVariables.at(observationFactorName);
      auto &rewardFactorPtr = _twoStageDynamicBayesianNetworkVariables.at(rewardFactorName);

      LOG(INFO) << "Identifying local state varibles.";
      // local factors - the direct parents of observation and reward factors
      for (auto &parentFactorName: observationFactorPtr->getListOfParents()) {
        _localFactors.insert(StringUtils::removeLastPrime(parentFactorName));
      }
      for (auto &parentFactorName: rewardFactorPtr->getListOfParents()) {
        _localFactors.insert(StringUtils::removeLastPrime(parentFactorName));
      }
      _localFactors.insert("a"+agentID);

      LOG(INFO) << "Identifying influence source state variables and influence destination state variables.";
      // influence sources and influence destinations
      for (auto &localFactorName: _localFactors) {
        bool hasLinkFromOutside = false;
        for (auto &localFactorParentName: _twoStageDynamicBayesianNetworkVariables.at(StringUtils::addLastPrime(localFactorName))->getListOfParents()) {
          if (_localFactors.find(localFactorParentName) == _localFactors.end() && _localFactors.find(StringUtils::addLastPrime(localFactorParentName)) == _localFactors.end()) {
            _sourceFactors.insert(StringUtils::removeLastPrime(localFactorParentName));
            hasLinkFromOutside = true;
          }
        }
        if (hasLinkFromOutside == true) {
          _destinationFactors.insert(StringUtils::addLastPrime(localFactorName));
        }
      }
      
      _dSeparationSetPerStage.insert(_localFactors.begin(), _localFactors.end()); // for now

      localFactors = ContainerUtils::setToVector(_localFactors);
      std::sort(localFactors.begin(), localFactors.end(), _factorComparator);
      LOG(INFO) << "Local state variables and actions: " << PrintUtils::vectorToTupleString(localFactors);
      sourceFactors = ContainerUtils::setToVector(_sourceFactors);
      std::sort(sourceFactors.begin(), sourceFactors.end(), _factorComparator);
      LOG(INFO) << "Influence source state variables: " << PrintUtils::vectorToTupleString(sourceFactors);
      destinationFactors = ContainerUtils::setToVector(_destinationFactors);
      std::sort(destinationFactors.begin(), destinationFactors.end(), _factorComparator);
      LOG(INFO) << "Influence destination state variables: " << PrintUtils::vectorToTupleString(destinationFactors);
      dSeparationSetPerStage = ContainerUtils::setToVector(_dSeparationSetPerStage);
      std::sort(dSeparationSetPerStage.begin(), dSeparationSetPerStage.end(), _factorComparator);
      LOG(INFO) << "D Separation Set Per Step: " << PrintUtils::vectorToTupleString(dSeparationSetPerStage);
      for (auto &factorName: localFactors) {
        if (factorName[0] != 'a' && factorName[0] != 'o') {
          localStates.push_back(factorName);
        }
      }
      std::sort(localStates.begin(), localStates.end(), _factorComparator);
      LOG(INFO) << "Local state variables: " << PrintUtils::vectorToTupleString(localStates);
      
      std::set<std::string> setIn = std::set<std::string>(_localFactors);
      for (auto &factor: sourceFactors) setIn.insert(factor);

      std::set<std::string> setOut = getStateFactorsNextStage(_localFactors);
      setOut.insert(observationFactorName);
      setOut.insert(rewardFactorName);

      LOG(INFO) << "Inputs to PGM: " << PrintUtils::setToTupleString(setIn);
      LOG(INFO) << "Outputs from PGM: " << PrintUtils::setToTupleString(setOut);

      computeSamplingOrder(setIn, setOut, "local");

      LOG(INFO) << "Local model has been constructed.";
      
    }

    TwoStageDynamicBayesianNetwork(std::string yamlFilePath){
      LOG(INFO) << "Loading " << yamlFilePath << ".";
      clock_t begin = std::clock();
      std::ifstream ifs(yamlFilePath);
      std::string content;
      content.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
      YAML::Node config = YAML::Load(content);
      double elapsed_seconds = double(std::clock()-begin) / CLOCKS_PER_SEC;
      LOG(INFO) << "Yaml file loaded after " << std::to_string(elapsed_seconds) << " seconds. Constructing the DBN ...";
      for (YAML::const_iterator it = config.begin(); it != config.end(); ++it){
        std::string key = it->first.as<std::string>();
        _twoStageDynamicBayesianNetworkVariables[key] = new TwoStageDynamicBayesianNetworkVariable(key, config[key], &_randomNumberGenerator);
        if (_twoStageDynamicBayesianNetworkVariables.at(key)->isStateVariable() == true) {
          _stateVariables.push_back(key);
        } 
      }
      _randomNumberGenerator.seed(time(0));
      LOG(INFO) << "Two stage dynamic bayesian network has been built.";
    }

    ~TwoStageDynamicBayesianNetwork() {
      for (auto &[key, val]: _twoStageDynamicBayesianNetworkVariables) {
        delete val;
      }
    }

    // sample variables sequentially acc#include <algorithm>rding to the order specified by the sampling mode
    void step(std::map<std::string, int> &state, const std::string &samplingMode){
      for (auto const &varName: _samplingOrders[samplingMode]){
        std::vector<int> indicesOfConditionalVariables;
        for (auto const& parentName: _twoStageDynamicBayesianNetworkVariables[varName]->getListOfParents()){
          try {
            indicesOfConditionalVariables.push_back(state[parentName]);
          } catch (const std::exception& e) {
            LOG(FATAL) << e.what();
          }
        }
        auto sampledValue = _twoStageDynamicBayesianNetworkVariables[varName]->sample(indicesOfConditionalVariables);
        state[varName] = sampledValue ;
      }
      // update state variables
      for (auto const &key: _samplingOrders[samplingMode]) {
        if (key[0] == 'x') {
          state.at(key.substr(0, key.size()-1)) = state.at(key);
        }
      }
    }

    // compute a sampling order given set of input variables and output variables and assign a name (sampling mode) to it
    void computeSamplingOrder(const std::set<std::string> &setOfInputVariables, const std::set<std::string> &setOfOutputVariables, const std::string &samplingMode){
      std::set<std::string> toSamplePool;
      std::set<std::string> sampledPool;
      toSamplePool.insert(setOfOutputVariables.begin(), setOfOutputVariables.end());
      sampledPool.insert(setOfInputVariables.begin(), setOfInputVariables.end());
      LOG(INFO) << "Computing sampling order with inputs: " + PrintUtils::setToTupleString<std::string>(sampledPool);
      LOG(INFO) << "and outputs: " + PrintUtils::setToTupleString<std::string>(toSamplePool);
      _samplingOrders[samplingMode].clear();
      while (toSamplePool.size() > 0){
        for (auto &var: toSamplePool){
          bool allParentsAreSampled = true;
          for (auto &parentName: _twoStageDynamicBayesianNetworkVariables[var]->getListOfParents()){
            if (sampledPool.find(parentName) == sampledPool.end()){
              allParentsAreSampled = false;
              toSamplePool.insert(parentName);
            }
          }
          if (allParentsAreSampled == true){
              _samplingOrders[samplingMode].push_back(var);
              sampledPool.insert(var);
              break;
          }
        }
        toSamplePool.erase(_samplingOrders[samplingMode].back());
      }
      LOG(INFO) << "sampling order: " + PrintUtils::vectorToString<std::string>(_samplingOrders.at(samplingMode));
    }

    TwoStageDynamicBayesianNetworkVariable *&getVariable(std::string &varName) {
      return _twoStageDynamicBayesianNetworkVariables.at(varName);
    }

    // retrieve the value of a variable from its index - this is somewhat strange design
    float getValueOfVariableFromIndex(const std::string &variableName, std::map<std::string, int> &state) {
      return _twoStageDynamicBayesianNetworkVariables.at(variableName)->getValueFromIndex(state.at(variableName));
    }

    void computeFullSamplingOrder() {
      
      std::set<std::string> setIn;
      std::set<std::string> setOut;

      for (auto const & [varName, variable]: _twoStageDynamicBayesianNetworkVariables) {
        if (varName[0] == 'a') {
          setIn.insert(varName);
        } else if (varName[0] == 'o' || varName[0] == 'r') {
          setOut.insert(varName);
        } else {
          if (StringUtils::lastBitIsPrime(varName) == true) {
            setOut.insert(varName);
          } else {
            setIn.insert(varName);
          }
        }
      }      
      computeSamplingOrder(setIn, setOut, "full");
    }

    int getNumberOfStates() {
      int count = 0;
      for (auto &[key, val]: _twoStageDynamicBayesianNetworkVariables) {
        if (key[0] == 'x' && StringUtils::lastBitIsPrime(key) == false) {
          count += 1;
        }
      }
      return count;
    }

    std::map<std::string, int> getNumberOfActions() {
      std::map<std::string, int> numberOfActions;
      for (auto &[key, var]: _twoStageDynamicBayesianNetworkVariables) {
        if (key[0] == 'a') {
          auto agentID = key.substr(1);
          int numberOfValues = var->getNumberOfValues();
          numberOfActions[agentID] = numberOfValues;
        }
      }
      return numberOfActions;
    }

    std::map<std::string, int> sampleInitialState() {
      std::map<std::string, int> initialMap;
      for (auto &key: _stateVariables) {
        initialMap[key] = _twoStageDynamicBayesianNetworkVariables.at(key)->sampleInitialValue();
      }
      return initialMap;
    }

    std::vector<std::string> &getStateVariables() {
      return _stateVariables;
    }

    std::default_random_engine &getRandomNumberGenerator() {
      return _randomNumberGenerator;
    }

  private:
    std::map<std::string, TwoStageDynamicBayesianNetworkVariable*> _twoStageDynamicBayesianNetworkVariables;
    std::vector<std::string> _stateVariables;
    std::map<std::string, std::vector<std::string>> _samplingOrders;
    std::default_random_engine _randomNumberGenerator;
    static bool _factorComparator(const std::string &a_, const std::string &b_) {
      auto a = StringUtils::removeLastPrime(a_);
      auto b = StringUtils::removeLastPrime(b_);
      if (a[0] == 'x' && b[0] != 'x') {
        return 1;
      } else if (b[0] == 'x' && a[0] != 'x') {
        return 0;
      } else if (a[0] == 'a' && b[0] != 'a') {
        return 1;
      } else if (b[0] == 'a' && a[0] != 'a') {
        return 0;
      } else if (a[0] == 'o' && b[0] != 'o') {
        return 1;
      } else if (b[0] == 'o' && a[0] != 'o') {
        return 0;
      } else {
        try {
          // find the common prefix of two strings
          int idx = 0;
          while (true) {
            if (idx > a.size()-1 || idx > b.size()-1) {
              break;
            } else if (a[idx] != b[idx]) {
              break;
            } else {
              idx += 1;
            }
          }
          int aN = std::stoi(a.substr(idx));
          int bN = std::stoi(b.substr(idx));
          if (aN < bN) {
            // means larger second
            return 1;
          } else {
            // means smaller first
            return 0;
          }
        } catch (std::invalid_argument &e) {
          int result = a.compare(b);
          if (result < 0) {
            return 1;
          } else {
            return 0;
          }
        }
      }
    }
};

#endif
