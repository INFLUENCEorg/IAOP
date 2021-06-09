#ifndef INFLUENCE_PREDICTOR_HPP_
#define INFLUENCE_PREDICTOR_HPP_

#include "dbns/TwoStageDynamicBayesianNetwork.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include "Utils.hpp"
#include "glog/logging.h"
#include <ctime>

// general influence predictor
class InfluencePredictor {
  public:
    InfluencePredictor(TwoStageDynamicBayesianNetwork *netPtr, std::vector<std::string> &localStatesAndActions, std::vector<std::string> &influenceSourceVariables): _netPtr(netPtr), _localStatesAndActions(localStatesAndActions), _influenceSourceVariables(influenceSourceVariables) {
      
    }
    virtual void sample(std::vector<int> &inputs, std::map<std::string, int> &dict) = 0;
    virtual void oneStepSample(std::vector<float> &hiddenState, std::vector<int> &inputs, bool &initial, std::map<std::string, int> &dict) {};
    virtual std::vector<float> getInitialState() { return std::vector<float>(); };
  protected:
    TwoStageDynamicBayesianNetwork *_netPtr;
    std::vector<std::string> _localStatesAndActions;
    std::vector<std::string> _influenceSourceVariables;
    int _sizeOfInputs = _localStatesAndActions.size();
};

// random influence predictor
class RandomInfluencePredictor: public InfluencePredictor  {
  public:
    RandomInfluencePredictor(TwoStageDynamicBayesianNetwork *netPtr, std::vector<std::string> &localStatesAndActions, std::vector<std::string> &influenceSourceVariables): InfluencePredictor(netPtr, localStatesAndActions, influenceSourceVariables) {
      LOG(INFO) << "Random influence predictor has been constructed.";
    }
    void oneStepSample(std::vector<float> &hiddenState, std::vector<int> &inputs, bool &initial, std::map<std::string, int> &dict) {
      for (auto &factorName: _influenceSourceVariables) {
        dict[factorName] = (_netPtr->getVariable(factorName))->sampleUniformly();
      }
    }
    void sample(std::vector<int> &inputs, std::map<std::string, int> &dict) {
      for (auto &factorName: _influenceSourceVariables) {
        dict[factorName] = (_netPtr->getVariable(factorName))->sampleUniformly();
      }
    }
};

class RecurrentInfluencePredictor: public InfluencePredictor {
  public:
    RecurrentInfluencePredictor(TwoStageDynamicBayesianNetwork *netPtr, std::vector<std::string> &localStatesAndActions, std::vector<std::string> &influenceSourceVariables, std::string modelPath, int numberOfHiddenStates): InfluencePredictor(netPtr, localStatesAndActions, influenceSourceVariables) {
      // load the trained model
      _model = torch::jit::load(modelPath);
      _model.eval();
      LOG(INFO) << "model has been loaded.";
      _numberOfHiddenStates = numberOfHiddenStates;
      _totalOutputSize = 0;
      for (auto &key: influenceSourceVariables) {
        int numberOfValues = netPtr->getVariable(key)->getNumberOfValues();
        _map[key] = numberOfValues;
        _totalOutputSize += numberOfValues;
      }
    }
    virtual void sample(std::vector<int> &inputs, std::map<std::string, int> &dict) = 0;

    virtual  void oneStepSample(std::vector<float> &hiddenState, std::vector<int> &inputs, bool &initial, std::map<std::string, int> &dict) = 0;

    std::vector<float> getInitialState() {
      std::vector<float> initialState;
      for (int i=0; i<=_numberOfHiddenStates-1; i++) {
        initialState.push_back(0.0);
      }
      return initialState;
    }

  protected:
    torch::jit::script::Module _model;
    c10::TensorOptions _options = torch::TensorOptions().dtype(torch::kFloat32);
    c10::TensorOptions _intOptions = torch::TensorOptions().dtype(torch::kInt32);
    int _numberOfHiddenStates;
    std::map<std::string, int> _map;
    int _totalOutputSize;
    std::default_random_engine *_randomNumberGeneratorPtr;
};

// RNN based influence predictor
class GRUInfluencePredictor: public RecurrentInfluencePredictor {
  public:
    GRUInfluencePredictor(TwoStageDynamicBayesianNetwork *netPtr, std::vector<std::string> &localStatesAndActions, std::vector<std::string> &influenceSourceVariables, std::string modelPath, int numberOfHiddenStates, bool fast=true): RecurrentInfluencePredictor(netPtr, localStatesAndActions, influenceSourceVariables, modelPath, numberOfHiddenStates) {

      _fast = fast;
      if (_fast == true) {
        for (const auto &pair: _model.named_parameters()) {
          if (pair.name == "gru.weight_ih_l0") {
            wxr = pair.value.index({torch::indexing::Slice(0,1*_numberOfHiddenStates)}).clone().transpose_(0,1);
            wxz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone().transpose_(0,1);
            wxn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone().transpose_(0,1);
          } else if (pair.name == "gru.weight_hh_l0") {
            whr = pair.value.index({torch::indexing::Slice(0,1*_numberOfHiddenStates)}).clone().transpose_(0,1);
            whz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone().transpose_(0,1);
            whn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone().transpose_(0,1);
          } else if (pair.name == "gru.bias_ih_l0") {
            bxr = pair.value.index({torch::indexing::Slice(0,1*_numberOfHiddenStates)}).clone();
            bxz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone();
            bxn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone();
          } else if (pair.name == "gru.bias_hh_l0") {
            bhr = pair.value.index({torch::indexing::Slice(0*_numberOfHiddenStates,1*_numberOfHiddenStates)}).clone();
            bhz = pair.value.index({torch::indexing::Slice(1*_numberOfHiddenStates,2*_numberOfHiddenStates)}).clone();
            bhn = pair.value.index({torch::indexing::Slice(2*_numberOfHiddenStates,3*_numberOfHiddenStates)}).clone();
          } else if (pair.name == "linear_layer.bias") {
            by = pair.value.clone();
          } else if (pair.name == "linear_layer.weight") {
            why = pair.value.clone().transpose_(0,1);
          }
          LOG(INFO) << "loaded: " << pair.name;
        }
      }
      LOG(INFO) << "GRU influence predictor has been constructed.";
    }
    void sample(std::vector<int> &inputs, std::map<std::string, int> &dict) {
      if ((int)inputs.size() == 0) {
        // sample from the initial belief
        for (int i=0; i <= (int)_influenceSourceVariables.size()-1; i++) {
          dict[_influenceSourceVariables[i]] = _netPtr->getVariable(_influenceSourceVariables[i])->sampleInitialValue();
        }
      } else {
        auto tensorInputs = torch::from_blob(inputs.data(), {1, (long int) inputs.size()}, _intOptions); 
        auto modelInputs = std::vector<torch::jit::IValue>({tensorInputs.view({1, -1, _sizeOfInputs}).toType(torch::kFloat32)});
        auto rawOuputs = _model.forward(modelInputs);
        c10::List<at::Tensor> modelOutputs = rawOuputs.toTensorList();
        for (int i=0; i <= (int)_influenceSourceVariables.size()-1; i++) {
          auto probs = modelOutputs.get(i)[0][-1].view(-1);
          auto sample = probs.multinomial(1).item<int64_t>();
          dict[_influenceSourceVariables[i]] = sample;
        }
      }   
    }
    void oneStepSample(std::vector<float> &hiddenState, std::vector<int> &inputs, bool &initial, std::map<std::string, int> &dict) {
      VLOG(4) << "Influence Predictor Inputs: " << PrintUtils::vectorToString(inputs);
      VLOG(4) << "Influce Predictor Hidden State: " << PrintUtils::vectorToString(hiddenState);
      if (initial == true) {
        // sample from the initial belief
        for (int i=0; i <= (int)_influenceSourceVariables.size()-1; i++) {
          dict[_influenceSourceVariables[i]] = _netPtr->getVariable(_influenceSourceVariables[i])->sampleInitialValue();
        }
      } else {
        auto begin = std::clock();
        if (_fast == true) {
          auto tensorInputs = torch::from_blob(inputs.data(), {1, _sizeOfInputs}, _intOptions).toType(torch::kFloat32);
          auto h = torch::from_blob(hiddenState.data(), {1, (long int) hiddenState.size()}, _options);
          auto r = torch::sigmoid(torch::matmul(tensorInputs, wxr) + bxr + torch::matmul(h, whr) + bhr);
          auto z = torch::sigmoid(torch::matmul(tensorInputs, wxz) + bxz + torch::matmul(h, whz) + bhz);
          auto n = torch::tanh(torch::matmul(tensorInputs, wxn) + bxn + torch::mul(r, torch::matmul(h, whn) + bhn));
          auto newHiddenState = torch::mul((1-z), n) + torch::mul(z, h);
          auto y = (torch::matmul(newHiddenState, why) + by).view(-1);
          auto expy = torch::exp(y);
          int count = 0;
          for (const std::string &key: _influenceSourceVariables) {
            auto& val = _map.at(key);
            auto probs = torch::div(expy.index({torch::indexing::Slice(count, count+val)}),torch::sum(expy.index({torch::indexing::Slice(count, count+val)}))).view(-1);
            std::discrete_distribution<int> dist (probs.data<float>(), probs.data<float>()+probs.numel());
            dict[key] = dist(_netPtr->getRandomNumberGenerator());
            count+=val;
          }
          for (int i=0; i<=(int)hiddenState.size()-1; i++){
            hiddenState[i] = *(newHiddenState.view(-1).data<float>()+i);
          }
        } else {
          auto tensorInputs = torch::from_blob(inputs.data(), {1,1,  _sizeOfInputs}, _intOptions).toType(torch::kFloat32);
          auto rawOutputs = _model.run_method("recurrentForward", torch::from_blob(hiddenState.data(), {1, 1, (long int) hiddenState.size()}, _options),  tensorInputs);
          auto TupleOfOutputs = (rawOutputs.toTuple())->elements();
          auto modelOutputs = TupleOfOutputs[0].toTensorList();
          auto newHiddenState = TupleOfOutputs[1].toTensor().view(-1);
          for (int i=0; i <= (int) _influenceSourceVariables.size()-1; i++) {
            auto probs = modelOutputs.get(i).view(-1);
            auto sample = probs.multinomial(1).item<int64_t>();
            dict[_influenceSourceVariables[i]] = sample;
          }
          for (int i=0; i<=(int)hiddenState.size()-1; i++){
            hiddenState[i] = *(newHiddenState.view(-1).data<float>()+i);
          }
        }
        VLOG(4) << "influence prediction took " << std::to_string((double)(std::clock()-begin)/CLOCKS_PER_SEC);
      }
      initial = false;
      VLOG(4) << "Update hidden to: " << PrintUtils::vectorToString(hiddenState);
    }

  private:
    torch::Tensor wxr;
    torch::Tensor bxr;
    torch::Tensor whr;
    torch::Tensor bhr;
    torch::Tensor wxz;
    torch::Tensor bxz;
    torch::Tensor whz;
    torch::Tensor bhz;
    torch::Tensor wxn;
    torch::Tensor bxn;
    torch::Tensor whn;
    torch::Tensor bhn;
    torch::Tensor by;
    torch::Tensor why;
    bool _fast;
};

class RNNInfluencePredictor: public RecurrentInfluencePredictor {
  public:
    RNNInfluencePredictor(TwoStageDynamicBayesianNetwork *netPtr, std::vector<std::string> &localStatesAndActions, std::vector<std::string> &influenceSourceVariables, std::string modelPath, int numberOfHiddenStates): RecurrentInfluencePredictor(netPtr, localStatesAndActions, influenceSourceVariables, modelPath, numberOfHiddenStates) {
      for (const auto &pair: _model.named_parameters()) {
        if (pair.name == "gru.weight_ih_l0") {
          wxh = pair.value.clone().transpose_(0,1);
        } else if (pair.name == "gru.weight_hh_l0") {
          whh = pair.value.clone().transpose_(0,1);
        } else if (pair.name == "gru.bias_ih_l0") {
          bxh = pair.value.clone();
        } else if (pair.name == "gru.bias_hh_l0") {
          bhh = pair.value.clone();
        } else if (pair.name == "linear_layer.bias") {
          by = pair.value.clone();
        } else if (pair.name == "linear_layer.weight") {
          why = pair.value.clone().transpose_(0,1);
        }
        LOG(INFO) << "loaded: " << pair.name;
      }
      LOG(INFO) << "RNN influence predictor has been constructed.";
    }
    void sample(std::vector<int> &inputs, std::map<std::string, int> &dict) {
      LOG(FATAL) << "not supported yet."; 
    }
    void oneStepSample(std::vector<float> &hiddenState, std::vector<int> &inputs, bool &initial, std::map<std::string, int> &dict) {
      VLOG(4) << "Influence Predictor Inputs: " << PrintUtils::vectorToString(inputs);
      VLOG(4) << "Influce Predictor Hidden State: " << PrintUtils::vectorToString(hiddenState);
      if (initial == true) {
        // sample from the initial belief
        for (int i=0; i <= (int)_influenceSourceVariables.size()-1; i++) {
          dict[_influenceSourceVariables[i]] = _netPtr->getVariable(_influenceSourceVariables[i])->sampleInitialValue();
        }
      } else {
        auto begin = std::clock();
        auto tensorInputs = torch::from_blob(inputs.data(), {1, _sizeOfInputs}, _intOptions).toType(torch::kFloat32);
        auto h = torch::from_blob(hiddenState.data(), {1, (long int) hiddenState.size()}, _options);
        auto newHiddenState = torch::tanh(torch::matmul(tensorInputs, wxh) + bxh + torch::matmul(h, whh) + bhh);
        auto y = (torch::matmul(newHiddenState, why) + by).view(-1);
        auto expy = torch::exp(y);
        int count = 0;
        for (const std::string &key: _influenceSourceVariables) {
          auto& val = _map.at(key);
          auto probs = torch::div(expy.index({torch::indexing::Slice(count, count+val)}),torch::sum(expy.index({torch::indexing::Slice(count, count+val)}))).view(-1);
          std::discrete_distribution<int> dist (probs.data<float>(), probs.data<float>()+probs.numel());
          dict[key] = dist(_netPtr->getRandomNumberGenerator());
          count+=val;
        }
        for (int i=0; i<=(int)hiddenState.size()-1; i++){
          hiddenState[i] = *(newHiddenState.view(-1).data<float>()+i);
        }
        VLOG(4) << "influence prediction took " << std::to_string((double)(std::clock()-begin)/CLOCKS_PER_SEC);
      }
      initial = false;
      VLOG(4) << "Update hidden to: " << PrintUtils::vectorToString(hiddenState);
    }

  private:
    torch::Tensor wxh;
    torch::Tensor bxh;
    torch::Tensor whh;
    torch::Tensor bhh;
    torch::Tensor why;
    torch::Tensor by;
};

#endif