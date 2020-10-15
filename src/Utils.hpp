#ifndef UTILS_HPP_
#define UTILS_HPP_

#define PRIME_CHAR 39

#include <iostream>
#include "yaml-cpp/yaml.h"
#include <string.h>
#include <type_traits>

namespace StringUtils {

  template <typename T>std::string toString(T &a) {
    std::ostringstream s;
    s << a;
    return s.str();
  }

  // remove the last n digits from the string
  std::string removeTheLastNDigits(std::string &str, int n) {
    return std::string(str.begin(), str.end()-n);
  }

  // remove the prime in the end if there is one
  std::string removeLastPrime(std::string str) {
    if (str.back() == PRIME_CHAR) {
      str = std::string(str.begin(), str.end()-1);
    }
    return str;
  }

  // add a prime to the end if there is none
  std::string addLastPrime(std::string str) {
    if (str[0] != 'a' && str.back() != PRIME_CHAR) {
      str = str + "'";
    }
    return str;
  }

  bool lastBitIsPrime(const std::string &str) {
    if (str.back() == 39) {
      return true;
    } else {
      return false;
    }
  }

}

namespace ContainerUtils {
  template <class T> std::vector<T> setToVector(std::set<T> &set) {
    return std::vector<T>(set.begin(), set.end());
  }
}

namespace PrintUtils {

  // convert vector to string  
  template <typename T> std::string vectorToString(std::vector<T> &vec, std::string connector=", ") {
    std::string str = "";
    for (auto &element: vec) {
      str += StringUtils::toString(element) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());;
  }

  // convert vector to tuple string
  template <class T> std::string vectorToTupleString(std::vector<T> &vec, std::string connector=", ") {
    return "(" + vectorToString(vec, connector) + ")";
  }

  // convert set to string
  template <typename T> std::string setToString(std::set<T> &set, std::string connector=", ") {
    std::string str = "";
    for (auto &element: set) {
      str += StringUtils::toString(element) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());;
  }

  // convert set to tuple string
  template <class T> std::string setToTupleString(std::set<T> &set, std::string connector=", ") {
    return "(" + setToString(set, connector) + ")";
  }

  template <class T> std::string mapToString(std::map<std::string, T> &map, std::string connector=", ") {
    std::string str = "";
    for (auto &[key, val]: map) {
      str += key + ": " + StringUtils::toString(val) + connector;
    }
    return StringUtils::removeTheLastNDigits(str, connector.length());
  } 

  template <class T> std::string mapToTupleString(std::map<std::string, T> &map, std::string connector=", ") {
    return "(" + mapToString(map, connector) + ")"; 
  }

}

namespace YAMLUtils {
    
}

namespace FireFighterUtils {
  std::string environmentStateToString(std::vector<int> &environmentState) {
    std::string str = "";
    for (int i=0; i<=(int)environmentState.size()-1; i++) {
      str += "House " + std::to_string(i+1) + ": Level " + std::to_string(environmentState[i]) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }
  std::string jointActionToString(std::map<std::string, int> &jointAction) {
    std::string str = "";
    for (auto &[key, val]: jointAction) {
      int agentID = std::stoi(key);
      int houseID;
      if (val == 0) {
        houseID = agentID;
      } else {
        houseID = agentID + 1;
      }
      str += "Agent " + key + " goes to House " + std::to_string(houseID) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }
  std::string jointObservationToString(std::map<std::string, int> &jointObservation) {
    std::string str = "";
    for (auto &[key, val]: jointObservation) {
      str += "Agent " + key + ": ";
      if (val == true) {
        str += "Fire!";
      } else {
        str += "No Fire.";
      }
      str += " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }
  std::string jointRewardToString(std::map<std::string, float> &jointReward) {
    std::string str = "";
    for (auto &[key, val]: jointReward) {
      str += "Agent " + key + ": " + std::to_string(val) + " | "; 
    }
    str = str.substr(0, str.size()-2);
    return str;
  }
}

namespace GrabAChairUtils {
  std::string environmentStateToString(std::vector<int> &environmentState) {
    std::string str = "";
    for (int i=0; i<=(int)environmentState.size()-1; i++) {
      str += "House " + std::to_string(i+1) + ": Level " + std::to_string(environmentState[i]) + " | ";
    }
    str = str.substr(0, str.size()-2);
    return str;
  }
}

#endif