#ifndef GRAB_A_CHAIR_DOMAIN_HPP_
#define GRAB_A_CHAIR_DOMAIN_HPP_

#include "domains/Domain.hpp"
#include "agents/PlanningAgent.hpp"
#include "Utils.hpp"
#include "GrabAChairAtomicAgent.hpp"

class GrabAChairDomain: public Domain {

  public:
    
    GrabAChairDomain(const YAML::Node &parameters) : Domain(parameters) {}
    
    AtomicAgentSimulator *makeAtomicAgentSimulator(const std::string &agentID, const std::string &agentType) {
      AtomicAgentSimulator *agentSimulatorPtr;
      if (agentType == "Random") {
        agentSimulatorPtr = new RandomAtomicAgentSimulator(_numberOfActions[agentID]);
      } else if (agentType[0] == 'A') {
        agentSimulatorPtr = new DeterministicAtomicAgentSimulator(_numberOfActions[agentID], (int)(agentType[1] - (int)'0'));
      } else if (agentType.rfind("Pattern", 0) == 0) {
        agentSimulatorPtr = new GrabAChairPatternAtomicAgentSimulator((int)(agentType[7]-(int)'0'));
      } else if (agentType == "Count") {
        agentSimulatorPtr = new GrabAChairCountBasedAtomicAgentSimulator();
      } else if (agentType == "Happy") {
        agentSimulatorPtr = new GrabAChairHappyAtomicAgentSimulator();
      } else if (agentType == "Sad") {
        agentSimulatorPtr = new GrabAChairSadAtomicAgentSimulator();
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
      } else if (agentType == "A0") {
        atomicAgentPtr = new DeterministicAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters, 0);
      } else if (agentType == "A1") {
        atomicAgentPtr = new DeterministicAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters, 1);
      } else if (agentType.rfind("Pattern", 0) == 0) {
        atomicAgentPtr = new GrabAChairPatternAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters, (int)(agentType[7]-'0'));
      } else if (agentType == "Count") {
        atomicAgentPtr = new GrabAChairCountBasedAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters);
      } else if (agentType == "Happy") {
        atomicAgentPtr = new GrabAChairHappyAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters);
      } else if (agentType == "Sad") {
        atomicAgentPtr = new GrabAChairSadAtomicAgent(agentID, _numberOfActions[agentID], _numberOfStepsToPlan, thisAgentParameters);
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