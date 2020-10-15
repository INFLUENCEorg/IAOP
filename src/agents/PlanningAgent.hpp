#ifndef PLANNING_AGENT_HPP_
#define PLANNING_AGENT_HPP_

#include "agents/AtomicAgent.hpp"
#include "domains/Domain.hpp"
#include <math.h>
#include <ctime>

// forget about things above
template <class State> 
class POMCPAtomicAgent: public AtomicAgent {
  public:
  
    POMCPAtomicAgent(const std::string &agentID, const int &numberOfActions, int numberOfStepsToPlan, float discountFactor, const YAML::Node &parameters, Domain::SingleAgentSimulator<State> *simulatorPtr): AtomicAgent(agentID, numberOfActions, numberOfStepsToPlan, parameters) {
      _numberOfParticles = _parameters["Rollout"]["numberOfParticles"].as<int>();
      _rootObservationNodePtr = new POMCPObservationNode(this);
      _simulatorPtr = simulatorPtr;
      _planningHorizon = _numberOfStepsToPlan;
      _discountFactor = discountFactor;
      _discountHorizon = parameters["Rollout"]["discountHorizon"].as<float>();
      _particleReinvigoration = parameters["Rollout"]["particleReinvigoration"].as<bool>();
      _particleReinvigorationRate = parameters["Rollout"]["particleReinvigorationRate"].as<float>();
      _explorationConstant = _parameters["Rollout"]["explorationConstant"].as<float>();
      if (_parameters["Rollout"]["numberOfSimulationsPerStep"].IsDefined()) {
        _numberOfSimulationsPerStep = _parameters["Rollout"]["numberOfSimulationsPerStep"].as<int>();
      }
      if (_parameters["Rollout"]["numberOfSecondsPerStep"].IsDefined()) {
        _numberOfSecondsPerStep = _parameters["Rollout"]["numberOfSecondsPerStep"].as<double>();
      }
      
      LOG(INFO) << "A POMCP Agent has been created.";
    }

    ~POMCPAtomicAgent() {
      delete _rootObservationNodePtr;
    }

    void observe(int &observation) {
      VLOG(2) << "[Agent " + _agentID + "]: observed " + std::to_string(observation) + ".";
      // detect particle depletion
      if (_particleDepleted == false) {
        // prune the tree
        POMCPObservationNode *newRootNodePtr = _rootObservationNodePtr->pop(_previousActionTaken, observation);
        VLOG(3) << "[Agent " + _agentID + "]: new root node has been extracted from the previous search tree.";
        delete _rootObservationNodePtr;
        _rootObservationNodePtr = newRootNodePtr;
        VLOG(3) << "Search tree has been pruned."; 
        VLOG(3) << "the number of particles left after particle filtering: " << std::to_string(_rootObservationNodePtr->particles.size());
        if (_rootObservationNodePtr->particles.size() == 0) {
          _particleDepleted = true;
          LOG(INFO) << "Particle depleted with " << std::to_string(_planningHorizon) << " steps to go!";
        } else if (_particleReinvigoration == true) {
          int numberOfNewParticles = (int)(_rootObservationNodePtr->particles.size()*_particleReinvigorationRate);
          for (int i=0; i<=numberOfNewParticles-1; i++) {
            _rootObservationNodePtr->particles.push_back(_simulatorPtr->sampleInitialState());
          }
          VLOG(3) << std::to_string(numberOfNewParticles) + " new particles have been added.";
        }
      }
      VLOG(2) << "--------------------------------------------------";
    }

    void reset() {
      _AOH.clear();
      _AOH.reserve(1 + 2 * _numberOfStepsToPlan);
      _AOH[0] = 1;
      _particleDepleted = false;
      // reset planning horizon
      _planningHorizon = _numberOfStepsToPlan;

      delete _rootObservationNodePtr;
      // destroy the previous search tree and build a new one
      _rootObservationNodePtr = new POMCPObservationNode(this);

      // resample particles from initial state distributions
      _rootObservationNodePtr->sampleParticles();
    }

    int act(std::map<std::string, std::map<std::string, std::vector<double>>> &results, YAML::Node &agentsYAMLNode){
      
      int selectedAction;
      results["number_of_particles_before_simulation"][_agentID].push_back(_rootObservationNodePtr->particles.size());
      if (_particleDepleted == true) {
        VLOG(3) << "[Agent " + _agentID + "]: taking random action because of particle depletion";
        selectedAction = std::experimental::randint(0, _numberOfActions-1);
      } else {
        VLOG(3) << "[Agent " + _agentID + "]: started to do planning with horizon " + std::to_string(_planningHorizon) + ".";
        double elapsedTime = 0.0;
        int simulationID = 0;
        while (true) {
          // simulation stoping condition
          if (_numberOfSecondsPerStep > 0.0 && elapsedTime >= _numberOfSecondsPerStep) {
            VLOG(3) << "[Agent " + _agentID + "]: reached planning time.";
            break;
          } else if (_numberOfSimulationsPerStep > 0 && simulationID >= _numberOfSimulationsPerStep) {
            VLOG(3) << "[Agent " + _agentID + "]: reached number of simulations.";
            break;
          } else {
            // do the simulation and accumulate used time
            auto begin = std::clock();
            VLOG(4) << "Simulation " << std::to_string(simulationID) << " started.";
            _rootObservationNodePtr->rootSimulate(_planningHorizon);
            elapsedTime += double(std::clock()-begin)/CLOCKS_PER_SEC;
            simulationID += 1;
          }
        }
        VLOG(3) << "number of simulations performed: " << std::to_string(simulationID);
        // record the number of simulations per step
        results["number_of_simulations_per_step"][_agentID].push_back((double)simulationID);
        // pick the greedy action to take
        selectedAction = _rootObservationNodePtr->getBestAction(false);
      }

      _previousActionTaken = selectedAction;
      // decrease planning horizon by one
      _planningHorizon -= 1;
      // save planning information for replay

      if (agentsYAMLNode["save"].as<bool>() == true) {
        agentsYAMLNode[_agentID] = _rootObservationNodePtr->convertToYAMLNode();
      }
      VLOG(2) << "[Agent " + _agentID + "]: selected action " + std::to_string(selectedAction) + ".";
      VLOG(2) << "--------------------------------------------------";

      return selectedAction;
    }
    Domain::SingleAgentSimulator<State> *_simulatorPtr;
    int _numberOfSimulationsPerStep = -1;
    double _numberOfSecondsPerStep = -1.0;
    int _planningHorizon;
    float _explorationConstant;

    class POMCPTreeNode {
      public:
        POMCPTreeNode(POMCPAtomicAgent *POMCPAgentPtr):_POMCPAtomicAgentPtr(POMCPAgentPtr) {
            
        }
        void update(float Return) {
          _N += 1;
          _Q = _Q + (Return - _Q) / _N;
        }
        float &getQ() {
          return _Q;
        }
        int &getN() {
          return _N;
        }
      protected:
        int _N = 0;
        float _Q = 0.0;
        POMCPAtomicAgent *_POMCPAtomicAgentPtr;
    };

    class POMCPObservationNode;
    class POMCPActionNode: public POMCPTreeNode {
      public:
        POMCPActionNode(POMCPAtomicAgent *POMCPAgentPtr):POMCPTreeNode(POMCPAgentPtr) {
          
        }
        bool getObservationNode(int &observation, POMCPObservationNode *&node) {
          if (_childrenNodes.find(observation) == _childrenNodes.end()) {
            return false;
          } else {
            node = _childrenNodes[observation];
            return true;
          }
        }
        void attachObservationNode(int &observation, POMCPObservationNode *&node){
          _childrenNodes[observation] = node;
        }
        ~POMCPActionNode() {
          for (auto &[key, val]: _childrenNodes){
            delete val;
          }
        }
        std::map<int, POMCPObservationNode*> &getChildrenNodes() {
          return _childrenNodes;
        } 
        POMCPObservationNode *pop(int &observation){
          if (_childrenNodes.find(observation) == _childrenNodes.end()) {
            return nullptr;
          } else {
            POMCPObservationNode *nodePtr = _childrenNodes[observation];
            _childrenNodes.erase(observation);
            return nodePtr;   
          }
        }
      private:
        std::map<int, POMCPObservationNode*> _childrenNodes;
    };

    class POMCPObservationNode: public POMCPTreeNode {
      public:
        POMCPObservationNode(POMCPAtomicAgent *POMCPAgentPtr): POMCPTreeNode(POMCPAgentPtr) {
          for (int actionID=0; actionID<=_numberOfActions-1; actionID++){
            _actionsThatHaveNotBeenTaken.push(actionID);
            _childrenNodes[actionID] = new POMCPActionNode(this->_POMCPAtomicAgentPtr);
          }
        }
        ~POMCPObservationNode() {
          for (auto &[key, val]: _childrenNodes){
            delete val;
          }
        }
        void sampleParticles() {
          particles.clear();
          for (int particleID=0; particleID <= this->_POMCPAtomicAgentPtr->_numberOfParticles-1; particleID++){
            particles.push_back(this->_POMCPAtomicAgentPtr->_simulatorPtr->sampleInitialState());
          }
        }

        float rootSimulate(int horizon) {
          auto sampledState = sampleOneParticle();
          return simulate(sampledState, horizon, 0);
        }

        float simulate(State &sampledState, int horizon, int depth) {
          if (horizon == 0 || std::pow(this->_POMCPAtomicAgentPtr->_discountFactor, depth) < this->_POMCPAtomicAgentPtr->_discountHorizon){
            VLOG(4) << "simulation terminated with horizon " << std::to_string(horizon);
            return 0.0;
          } else {
            if (depth != 0) {
              particles.push_back(sampledState);
            }
            // one step simulation
            int action = getBestAction(true);
            int observation;
            float reward;
            bool done;
            VLOG(4) << "Doing one step simulation in the simulator";
            this->_POMCPAtomicAgentPtr->getSimulator()->step(sampledState, action, observation, reward, done);
            VLOG(4) << "Finished one step simulation in the simulator";

            POMCPObservationNode *observationNodePtr;
            bool exist = _childrenNodes[action]->getObservationNode(observation, observationNodePtr);
            float Return = reward;
            if (exist == true) {
                Return += this->_POMCPAtomicAgentPtr->_discountFactor * observationNodePtr->simulate(sampledState, horizon-1, depth+1);
            } else {

                
                POMCPObservationNode *newObservationNode = new POMCPObservationNode(this->_POMCPAtomicAgentPtr);
            
                float rolloutReturn = this->_POMCPAtomicAgentPtr->_simulatorPtr->rollout(sampledState, horizon-1, depth+1, this->_POMCPAtomicAgentPtr->_discountHorizon);
                
                newObservationNode->update(rolloutReturn);

                Return += this->_POMCPAtomicAgentPtr->_discountFactor * rolloutReturn;
                
                _childrenNodes[action]->attachObservationNode(observation, newObservationNode);
                
            }
            this->update(Return);
            _childrenNodes[action]->update(Return);
            return Return;
          }
        }

        float computeExplorationBonus(float &explorationConstant, int &Ntotal, int &N) {
            return explorationConstant * sqrtf(log(Ntotal)/N);
        }

        State sampleOneParticle() {
          int index = std::experimental::randint(0, (int)particles.size()-1);
          return particles[index];
        }

        int getBestAction(bool UCB=false) {
            if (_actionsThatHaveNotBeenTaken.empty() == false) {
                int action = _actionsThatHaveNotBeenTaken.front();
                _actionsThatHaveNotBeenTaken.pop();
                return action;
            } else {
                int bestAction = -1;
                float bestValue;
                for (int actionID=0; actionID<=_numberOfActions-1; actionID++){
                    float value = _childrenNodes[actionID]->getQ();
                    if (UCB==true) {
                        value += this->_POMCPAtomicAgentPtr->computeExplorationBonus(this->_N, _childrenNodes[actionID]->getN());
                    }
                    if (bestAction == -1 || value >= bestValue) {
                        bestAction = actionID;
                        bestValue = value;
                    }
                }
                return bestAction;
            }
        }

        POMCPObservationNode *pop(int &realActionTaken, int &realObservation) {
            auto nodePtr = _childrenNodes[realActionTaken]->pop(realObservation);
            if (nodePtr == nullptr) {
                nodePtr = new POMCPObservationNode(this->_POMCPAtomicAgentPtr);
            } 
            return nodePtr; 
        }

        YAML::Node convertToYAMLNode() {
            YAML::Node node;
            node["N"] = this->_N;
            node["Q"] = this->_Q;
            for (auto &[key, actionNode]: _childrenNodes) {
                node["Actions"][std::to_string(key)]["N"] = actionNode->getN();
                node["Actions"][std::to_string(key)]["Q"] = actionNode->getQ();
                for (auto &[obs, observationNode]: _childrenNodes[key]->getChildrenNodes()){
                    node["Actions"][std::to_string(key)]["Observations"][std::to_string(obs)] = observationNode->convertToYAMLNode();
                }
            }
            return node;
        }
        std::vector<State> particles;
      private:
        std::map<int, POMCPActionNode*> _childrenNodes;
        std::queue<int> _actionsThatHaveNotBeenTaken;
        int _numberOfActions = this->_POMCPAtomicAgentPtr->getNumberOfActions();
    };
    POMCPObservationNode *_rootObservationNodePtr;
    float computeExplorationBonus(int NTtotal, int N) {
      return _explorationConstant * sqrtf(log(NTtotal)/N);
    }
    int _previousActionTaken;
    int _numberOfParticles;
    float _discountFactor;
    float _discountHorizon;
    bool _particleDepleted = false;
    bool _particleReinvigoration = false;
    float _particleReinvigorationRate;
    int &getNumberOfActions(){
      return _numberOfActions;
    }
    float &getExplorationConstant() {
      return _explorationConstant;
    }
    Domain::SingleAgentSimulator<State> *&getSimulator() {
      return _simulatorPtr;
    }
    
};

#endif