#ifndef Grid_Traffic_DOMAIN_HPP_
#define Grid_Traffic_DOMAIN_HPP_

#include "domains/Domain.hpp"
#include "agents/PlanningAgent.hpp"
#include "Utils.hpp"
#include "GridTrafficAtomicAgent.hpp"

class GridTrafficDomain: public Domain {

  private:
    int _obsLength;

  public:
    
    GridTrafficDomain(const YAML::Node &parameters) : Domain(parameters) {
      _obsLength = parameters["GridTraffic"]["obsLength"].as<int>();
    }
    
    AtomicAgentSimulator *makeAtomicAgentSimulator(const std::string &agentID, const std::string &agentType) {
      AtomicAgentSimulator *agentSimulatorPtr;
      if (agentType == "Random") {
        agentSimulatorPtr = new RandomAtomicAgentSimulator(_numberOfActions[agentID]);
      } else if (agentType == "Simple2") {
        agentSimulatorPtr = new GridTrafficSimple2AtomicAgentSimulator(_obsLength);
      } else {
        LOG(FATAL) << "Agent type " << agentType << " not supported.";
      }
      return agentSimulatorPtr;
    }

    AtomicAgent *makeAtomicAgent(const std::string &agentID, const std::string &agentType) {
      AtomicAgent *atomicAgentPtr;
      YAML::Node fullAgentParameters = parameters["AgentComponent"];
      YAML::Node thisAgentParameters =  parameters["AgentComponent"][agentID];

      atomicAgentPtr = new RandomAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters);

      if (agentType == "Random") {
        atomicAgentPtr = new RandomAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters);
      } else if (agentType == "Simple2") {
        atomicAgentPtr = new GridTrafficSimple2AtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters, _obsLength);
      } else if (agentType == "POMCP") {
        std::string simulatorType = thisAgentParameters["Simulator"]["Type"].as<std::string>();
        if (simulatorType == "Global") {
          atomicAgentPtr = new POMCPAtomicAgent<Domain::SingleAgentGlobalSimulatorState>(
            agentID, 
            _numberOfActions[agentID], 
            _numberOfStepsToPlan, 
            _discountFactor,
            thisAgentParameters,
            new Domain::SingleAgentGlobalSimulator(
              agentID, 
              this, 
              fullAgentParameters
            )
          );
        } else if (simulatorType == "Local") {
          bool recurrent = thisAgentParameters["Simulator"]["InfluencePredictor"]["recurrent"].as<bool>();
          if (recurrent == false) {
            atomicAgentPtr = new POMCPAtomicAgent<Domain::SingleAgentSequentialInfluenceAugmentedSimulatorState>(
              agentID, 
              _numberOfActions[agentID], 
              _numberOfStepsToPlan, 
              _discountFactor,
              thisAgentParameters,
              new Domain::SingleAgentSequentialInfluenceAugmentedSimulator(
                agentID, 
                this, 
                thisAgentParameters["Simulator"]
              )
            );
          } else {
            atomicAgentPtr = new POMCPAtomicAgent<Domain::SingleAgentRecurrentInfluenceAugmentedSimulatorState>(
              agentID, 
              _numberOfActions[agentID], 
              _numberOfStepsToPlan, 
              _discountFactor,
              thisAgentParameters,
              new Domain::SingleAgentRecurrentInfluenceAugmentedSimulator(
                agentID, 
                this, 
                thisAgentParameters["Simulator"]
              )
            );
          }
        } else {
          std::string message = "Unsupported simulator type: " + simulatorType + ".";
          LOG(FATAL) << message;
          return nullptr;
        }
      } else {
        std::string message = "Agent Type " + agentType + " is not supported.";
        LOG(FATAL) << message;
        atomicAgentPtr == nullptr;
      }
      return atomicAgentPtr;
    }
};

#endif