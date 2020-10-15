#ifndef DOMAIN_HPP_
#define DOMAIN_HPP_

#include "../agents/AtomicAgent.hpp"
#include <experimental/random>
#include "dbns/TwoStageDynamicBayesianNetwork.hpp"
#include "influence/InfluencePredictor.hpp"
#include <memory>
#include <math.h>

// Assumptions & Data types:
// * an action is an integer. mutli agent action is a map of agentID->action.
// * an observation is an integer. multi agent observation is a map of agentID->observation.
// * a reward is a float. multi agent reward is a map of agentID->reward.
// * a done is a boolean. multi agent done is a boolean.
// * different simulators use different types of states.

class Domain {
  public:
    // single agent simulator for planning agents
    // using templating as different simulators will need states of data types
    template <class State> class SingleAgentSimulator {
      public:
        SingleAgentSimulator(const std::string &IDOfAgentToControl, Domain *domainPtr): _IDOfAgentToControl(IDOfAgentToControl), _domainPtr(domainPtr) {}
        virtual void step(State &state, int action, int &observation, float &reward, bool &done) = 0;
        virtual float rollout(State &initialState, int horizon, int depth, float discountHorizon) = 0;
        virtual State sampleInitialState() = 0;
      protected:
        Domain *_domainPtr;
        std::string _IDOfAgentToControl;
    };

    // the state space of global simulator
    struct SingleAgentGlobalSimulatorState {
      std::map<std::string, int> environmentState;
      std::vector<int> AOH; // the AOH of other agents // can also be map of vectors
    };

    // single agent global simulator
    class SingleAgentGlobalSimulator: public SingleAgentSimulator<SingleAgentGlobalSimulatorState> {
      public:
        SingleAgentGlobalSimulator(const std::string &IDOfAgentToControl, Domain *domainPtr, const YAML::Node &fullAgentParameters): SingleAgentSimulator<SingleAgentGlobalSimulatorState>(IDOfAgentToControl, domainPtr) {
          // build up the agent simulators
          int counter = 0;
          for (YAML::const_iterator it = fullAgentParameters.begin(); it != fullAgentParameters.end(); it++) {
            std::string agentID = it->first.as<std::string>();
            if (agentID != _IDOfAgentToControl) {
              std::string agentType = it->second["Type"].as<std::string>();
              agentSimulators[agentID] = std::unique_ptr<AtomicAgentSimulator>(_domainPtr->makeAtomicAgentSimulator(agentID, agentType));
              agentStateIndices[agentID] = counter;
              counter +=  1 + 2 * (_domainPtr->_numberOfStepsToPlan);
            }
          }
          _sizeOfAOH = counter;
          VLOG(1) << _domainPtr->_domainName << " single agent global simulator has been built.";
        }

        void updateState(SingleAgentGlobalSimulatorState &state) {
          // send observations to the corresponding agents
          for (auto &[agentID, startIndex]: agentStateIndices) {
            int agentObs = _domainPtr->_DBNPtr->getValueOfVariableFromIndex("o"+agentID, state.environmentState);
            agentSimulators[agentID]->observe(state.AOH.begin()+agentStateIndices[agentID], agentObs);
          }
        }

        void step(SingleAgentGlobalSimulatorState &state, int action, int &observation, float &reward, bool &done) {
          // simulate actions of other agents
          for (auto &[agentID, agentSimulator]: agentSimulators) {
            int simulatedAction =  agentSimulator->step(state.AOH.begin()+agentStateIndices[agentID]);
            VLOG(4) << "Agent " << agentID << " is simulated to take action " << std::to_string(simulatedAction) << ".";
            state.environmentState["a"+agentID] = simulatedAction;
          }
          state.environmentState["a"+_IDOfAgentToControl] = action;
          VLOG(4) << "Finished sampling actions of other agents.";
          _domainPtr->_DBNPtr->step(state.environmentState, "full");
          VLOG(4) << "Finished one step sampling in the DBN.";
          observation = _domainPtr->_DBNPtr->getValueOfVariableFromIndex("o"+_IDOfAgentToControl, state.environmentState);
          reward = _domainPtr->_DBNPtr->getValueOfVariableFromIndex("r"+_IDOfAgentToControl, state.environmentState);
          this->updateState(state);
          done = false;
          VLOG(4) << "Finished one step simulation in the global simulator.";
        }
        SingleAgentGlobalSimulatorState sampleInitialState() {
          SingleAgentGlobalSimulatorState sampledState;
          sampledState.environmentState = _domainPtr->sampleInitialState();
          sampledState.AOH.resize(_sizeOfAOH);
          for (auto &[agentID, startIndex]: agentStateIndices) {
            sampledState.AOH[startIndex] = 1; // 1 means writing starts from index 1
          }
          return sampledState;
        }
        float rollout(SingleAgentGlobalSimulatorState &state, int horizon, int depth, float discountHorizon) {
          auto begin = std::clock();
          float undiscounted_return = 0.0;
          float factor = 1.0;
          float tFactor = std::pow(_domainPtr->_discountFactor, depth);
          for (int step=0; step<=horizon-1; step++) {
            if (tFactor < discountHorizon) {
               
              break;
            }
            // simulate actions of agents
            for (auto &[agentID, agentSimulator]: agentSimulators) {
              int simulatedAction =  agentSimulator->step(state.AOH.begin()+agentStateIndices[agentID]);
              VLOG(4) << "Agent " << agentID << " is simulated to take action " << std::to_string(simulatedAction) << ".";
              state.environmentState["a"+agentID] = simulatedAction;
            }
            state.environmentState["a"+_IDOfAgentToControl] = std::experimental::randint(0,_domainPtr->_numberOfActions[_IDOfAgentToControl]-1);
            // one step simulation in the DBN
            _domainPtr->_DBNPtr->step(state.environmentState, "full");
            undiscounted_return += factor * _domainPtr->_DBNPtr->getValueOfVariableFromIndex("r"+_IDOfAgentToControl, state.environmentState);
            if (step != horizon-1) {
              this->updateState(state);
            }
            factor *= _domainPtr->_discountFactor;
            depth += 1;
            tFactor *= _domainPtr->_discountFactor;
          }
          VLOG(5) << "time to rollout: " << (double)(std::clock()-begin)/CLOCKS_PER_SEC;
          return undiscounted_return;
        }
      private:
        std::map<std::string, std::unique_ptr<AtomicAgentSimulator>> agentSimulators;
        int _sizeOfAOH = 0;
        std::map<std::string, int> agentStateIndices;
    };

    // single agnet influence augmented local simulator
    template <class State> class SingleAgentInfluenceAugmentedSimulator: public SingleAgentSimulator<State> {
      public:
        SingleAgentInfluenceAugmentedSimulator(const std::string &IDOfAgentToControl, Domain *domainPtr, const YAML::Node &simulatorParameters): SingleAgentSimulator<State>(IDOfAgentToControl, domainPtr) {
          // construct the local model
          this->_domainPtr->_DBNPtr->constructLocalModel(IDOfAgentToControl, _localFactors, _localStates, _sourceFactors, _destinationFactors, _dSeparationSetPerStep);
          // construct the influence predictor
          std::string influencePredictorType = simulatorParameters["InfluencePredictor"]["Type"].as<std::string>();
          if (influencePredictorType == "Random") {
            _influencePredictorPtr = std::unique_ptr<InfluencePredictor>(new RandomInfluencePredictor(this->_domainPtr->_DBNPtr, _dSeparationSetPerStep, _sourceFactors));
          } else {
            std::string modelPath = simulatorParameters["InfluencePredictor"]["modelPath"].as<std::string>();
            int numberOfHiddenStates = simulatorParameters["InfluencePredictor"]["numberOfHiddenStates"].as<int>();
            if (influencePredictorType == "RNN") {
              _influencePredictorPtr = std::unique_ptr<InfluencePredictor>(new RNNInfluencePredictor(this->_domainPtr->_DBNPtr, _dSeparationSetPerStep, _sourceFactors, modelPath, numberOfHiddenStates));
            } else if (influencePredictorType == "GRU") {
              _influencePredictorPtr = std::unique_ptr<InfluencePredictor>(new GRUInfluencePredictor(this->_domainPtr->_DBNPtr, _dSeparationSetPerStep, _sourceFactors, modelPath, numberOfHiddenStates, simulatorParameters["InfluencePredictor"]["fast"].as<bool>()));
            } else {
              LOG(FATAL) << "Influence predictor type " << influencePredictorType << " is not supported.";
            }
          }  
        }

      protected:
        // need to rethink about the namings
        std::unique_ptr<InfluencePredictor> _influencePredictorPtr;
        std::vector<std::string> _localFactors;
        std::vector<std::string> _sourceFactors;
        std::vector<std::string> _localStates;
        std::vector<std::string> _destinationFactors;
        std::vector<std::string> _dSeparationSetPerStep;

        void sampleEnvironmentState(std::map<std::string, int> &environmentState) {
          // sample a full environment state
          std::map<std::string, int> fullSampledState = this->_domainPtr->sampleInitialState();
          // take the local states out
          for (auto &varName: _localStates) {
            environmentState[varName] = fullSampledState[varName];
          }
        }
    };

    struct SingleAgentSequentialInfluenceAugmentedSimulatorState {
      std::map<std::string, int> environmentState;
      std::vector<int> influencePredictorInputs;
    };

    // singlet agent sequential influence augmented local simulator
    class SingleAgentSequentialInfluenceAugmentedSimulator: public SingleAgentInfluenceAugmentedSimulator<SingleAgentSequentialInfluenceAugmentedSimulatorState> {
      public:
        SingleAgentSequentialInfluenceAugmentedSimulator(const std::string &IDOfAgentToControl, Domain *domainPtr, const YAML::Node &simulatorParameters): SingleAgentInfluenceAugmentedSimulator<SingleAgentSequentialInfluenceAugmentedSimulatorState>(IDOfAgentToControl, domainPtr, simulatorParameters) {
          VLOG(1) << "Single agent sequential influence augmented simulator has been built.";
        }

        // append local states + action + observation to the influence predictor inputs for the next stage
        void updateState(SingleAgentSequentialInfluenceAugmentedSimulatorState &state, int action) {
          for (auto &key: _localStates) {
            state.influencePredictorInputs.push_back(state.environmentState[key]);
          }
          state.influencePredictorInputs.push_back(action);
        }

        void step(SingleAgentSequentialInfluenceAugmentedSimulatorState &state, int action, int &observation, float &reward, bool &done) {
          state.environmentState["a"+this->_IDOfAgentToControl] = action;
          _influencePredictorPtr->sample(state.influencePredictorInputs, state.environmentState);
          this->_domainPtr->_DBNPtr->step(state.environmentState, "local");
          reward = this->_domainPtr->_DBNPtr->getValueOfVariableFromIndex("r"+this->_IDOfAgentToControl, state.environmentState);
          observation = this->_domainPtr->_DBNPtr->getValueOfVariableFromIndex("o"+this->_IDOfAgentToControl, state.environmentState);
          this->updateState(state, action);
          done =false;
        }

        float rollout(SingleAgentSequentialInfluenceAugmentedSimulatorState &state, int horizon, int depth, float discountHorizon) {
          if (horizon <= 0) return 0.0;
          float undiscounted_return = 0.0;
          float factor = 1.0;
          float tFactor = std::pow(_domainPtr->_discountFactor, depth);
          for (int step=0; step<=horizon-1; step++){
            
            if (tFactor < discountHorizon) {
              VLOG(4) << "rollout terminated because discount horizon reached.";
              break;
            }

            int action = std::experimental::randint(0,_domainPtr->_numberOfActions[_IDOfAgentToControl]-1);
            _influencePredictorPtr->sample(state.influencePredictorInputs, state.environmentState);
            state.environmentState["a"+_IDOfAgentToControl] = action;
            _domainPtr->_DBNPtr->step(state.environmentState, "local");
            undiscounted_return += factor * _domainPtr->_DBNPtr->getValueOfVariableFromIndex("r"+_IDOfAgentToControl, state.environmentState);
            if (step != horizon-1) {
              int observation = _domainPtr->_DBNPtr->getValueOfVariableFromIndex("o"+_IDOfAgentToControl, state.environmentState);
              this->updateState(state, action);
            }

            depth += 1;
            factor *= _domainPtr->_discountFactor;
            tFactor *= _domainPtr->_discountFactor;
          }
          return undiscounted_return;
        }

        SingleAgentSequentialInfluenceAugmentedSimulatorState sampleInitialState() {
          SingleAgentSequentialInfluenceAugmentedSimulatorState sampledState;
          this->sampleEnvironmentState(sampledState.environmentState);
          return sampledState;
        }
    };

    struct SingleAgentRecurrentInfluenceAugmentedSimulatorState {
      std::map<std::string, int> environmentState;
      bool initial; // whether this is an initial state
      std::vector<int> influencePredictorInputs;
      std::vector<float> influencePredictorState; // the hidden state of the influence predictor
    };

    class SingleAgentRecurrentInfluenceAugmentedSimulator: public SingleAgentInfluenceAugmentedSimulator<SingleAgentRecurrentInfluenceAugmentedSimulatorState> {
      public:
        SingleAgentRecurrentInfluenceAugmentedSimulator(const std::string &IDOfAgentToControl, Domain *domainPtr, const YAML::Node &simulatorParameters): SingleAgentInfluenceAugmentedSimulator<SingleAgentRecurrentInfluenceAugmentedSimulatorState>(IDOfAgentToControl, domainPtr, simulatorParameters) {
          VLOG(1) << "Single agent recurrent influence augmented simulator has been built.";
        }

        // notice that in the previous simulator, the inputs include the entire history of local states, actions and observations
        // but here since we have hidden states, the inputs only need to include local state, action, and observation of last stage
        void updateState(SingleAgentRecurrentInfluenceAugmentedSimulatorState &state, int action) {
          int count = 0;
          for (auto &key: _localStates){
            state.influencePredictorInputs[count] = state.environmentState[key];
            count += 1;
          }
          state.influencePredictorInputs[count] = action;
          state.initial = false;
        }

        void step(SingleAgentRecurrentInfluenceAugmentedSimulatorState &state, int action, int &observation, float &reward, bool &done) {
          state.environmentState["a"+this->_IDOfAgentToControl] = action;
          _influencePredictorPtr->oneStepSample(state.influencePredictorState, state.influencePredictorInputs, state.initial, state.environmentState);
          this->_domainPtr->_DBNPtr->step(state.environmentState, "local");
          reward = this->_domainPtr->_DBNPtr->getValueOfVariableFromIndex("r"+this->_IDOfAgentToControl, state.environmentState);
          observation = this->_domainPtr->_DBNPtr->getValueOfVariableFromIndex("o"+this->_IDOfAgentToControl, state.environmentState);
          this->updateState(state, action);
          done =false;
        }

        float rollout(SingleAgentRecurrentInfluenceAugmentedSimulatorState &state, int horizon, int depth, float discountHorizon) {
          float undiscounted_return = 0.0;
          float factor = 1.0;
          float tFactor = std::pow(_domainPtr->_discountFactor, depth);
          for (int step=0; step<=horizon-1; step++){
            
            if (tFactor < discountHorizon) {
              VLOG(4) << "rollout terminated because discount horizon reached.";
              break;
            }

            int action = std::experimental::randint(0, _domainPtr->_numberOfActions[_IDOfAgentToControl]-1);
            _influencePredictorPtr->oneStepSample(state.influencePredictorState, state.influencePredictorInputs, state.initial, state.environmentState);
            state.environmentState["a"+_IDOfAgentToControl] = action;
            _domainPtr->_DBNPtr->step(state.environmentState, "local");
            undiscounted_return += factor * _domainPtr->_DBNPtr->getValueOfVariableFromIndex("r"+_IDOfAgentToControl, state.environmentState);
            if (step != horizon-1) {
              int observation = _domainPtr->_DBNPtr->getValueOfVariableFromIndex("o"+_IDOfAgentToControl, state.environmentState);
              this->updateState(state, action);
            }
            depth += 1;
            tFactor *= _domainPtr->_discountFactor;
            factor *= _domainPtr->_discountFactor;
          }
          return undiscounted_return;
        }

        // this should work as our local state space only includes the fire levels of the houses
        SingleAgentRecurrentInfluenceAugmentedSimulatorState sampleInitialState() {
          SingleAgentRecurrentInfluenceAugmentedSimulatorState sampledState;
          sampledState.initial = true;
          sampledState.influencePredictorState = _influencePredictorPtr->getInitialState();
          this->sampleEnvironmentState(sampledState.environmentState);
          // placeholders
          for (int i=0; i<=(int)_dSeparationSetPerStep.size()-1; i++){
            sampledState.influencePredictorInputs.push_back(0);
          }
          return sampledState;
        }
    };

    class Environment {
      public:
        Environment(Domain &domain): _domain(domain){};
        virtual void step(std::map<std::string, int> &action, std::map<std::string, int> &observation, std::map<std::string, float> &reward, bool &done) {
          _domain.step(state, action, observation, reward, done, "full");
          std::map<std::string, int> newState = state;
          state.clear();
          for (auto &[key, val]: newState) {
            if (key[0] == 'x' && key.back() != 39) {
              state[key] = val;
            }
          }
        }
        virtual void reset() {
          VLOG(1) << "Resetting environment.";
          state = _domain.sampleInitialState();
        }
        virtual ~Environment(){};
        virtual std::map<std::string, int> &getState() {
          return state;
        }
      private:
        Domain &_domain;
        std::map<std::string, int> state;
    };

    Domain(const YAML::Node &parameters){
      this->parameters = parameters;

      // name of domain
      _domainName = parameters["General"]["domain"].as<std::string>();

      // loading agent IDs
      for(YAML::const_iterator it=parameters["AgentComponent"].begin(); it!=parameters["AgentComponent"].end(); ++it) {
        std::string agentID = it->first.as<std::string>();
        _listOfAgentIDs.push_back(agentID);
      }
      VLOG(1) << "list of agents: " + PrintUtils::vectorToString(_listOfAgentIDs);

      // building the DBN
      std::string yamlFilePath = parameters[_domainName]["2SDBNYamlFilePath"].as<std::string>();
      _DBNPtr = new TwoStageDynamicBayesianNetwork(yamlFilePath);
      _DBNPtr->computeFullSamplingOrder(); 

      _numberOfActions = _DBNPtr->getNumberOfActions();
      _numberOfAgents = _listOfAgentIDs.size();
      _numberOfStepsToPlan = parameters["General"]["horizon"].as<int>();
      _numberOfEnvironmentStates = _DBNPtr->getNumberOfStates();
      _discountFactor = parameters["General"]["discountFactor"].as<float>();

      LOG(INFO) << _domainName << " has been initialized";
    }

    virtual AtomicAgentSimulator *makeAtomicAgentSimulator(const std::string &agentID, const std::string &agentType) = 0;

    virtual AtomicAgent *makeAtomicAgent(const std::string &agentID, const std::string &agentType) = 0;

    Environment *makeEnvironment() {
      return new Environment(*this);
    }

    virtual ~Domain(){
      delete _DBNPtr;
      VLOG(1) << "Domain deleted.";
    }

    int getNumberOfAgents(){ return _listOfAgentIDs.size(); }

    std::vector<std::string> &getListOfAgentIDs() {return _listOfAgentIDs;} 

    std::map<std::string, int> &getAgentsSpecification() {return _numberOfActions;}

    TwoStageDynamicBayesianNetwork *getDBNPtr() {
      return _DBNPtr;
    }

protected:
  TwoStageDynamicBayesianNetwork *_DBNPtr;
  YAML::Node parameters;
  std::string _domainName;
  int _numberOfAgents;
  std::vector<std::string> _listOfAgentIDs;
  std::map<std::string, int> _numberOfActions;
  int _numberOfEnvironmentStates;
  int _numberOfStepsToPlan;
  float _discountFactor;

  virtual std::map<std::string, int> sampleInitialState() {
    return _DBNPtr->sampleInitialState();
  }

  virtual void step(std::map<std::string, int> &state, std::map<std::string, int> &action, std::map<std::string, int> &observation, std::map<std::string, float> &reward, bool &done, const std::string &samplingMode) {

    // read actions
    for (const auto &agentID: _listOfAgentIDs) {
      state["a"+agentID] = action[agentID];
    }
    
    // perform one step sampling in the two stage dynamic bayesian network
    _DBNPtr->step(state, samplingMode); 

    VLOG(4) << "Finished one step in the bayesian network.";

    // update state, observation, reward and done
    for (const auto &agentID: _listOfAgentIDs) {
      reward[agentID] = _DBNPtr->getValueOfVariableFromIndex("r" + agentID, state);
      observation[agentID] = _DBNPtr->getValueOfVariableFromIndex("o" + agentID, state);
    }

    done = false;
  }
};

#endif
