#ifndef TWO_STAGE_DYNAMIC_BAYESIAN_STATE_HPP_
#define TWO_STAGE_DYNAMIC_BAYESIAN_STATE_HPP_

// the state of two stage dynamic bayesian network
class TwoStageDynamicBayesianNetworkState {
  public:
    
    TwoStageDynamicBayesianNetworkState(std::map<std::string, int> &dictionary):_valuesOfVariables(dictionary){}
    
    std::string toString(){
      std::string str = "\n";
      for (auto const& [key, val] : _valuesOfVariables) {
          str = str + key + ": " + std::to_string(val) + "\n";
      }
      str = str.substr(0, str.size()-1);
      return str;
    }
    auto operator[](std::string const& variableName){
      return _valuesOfVariables[variableName];
    }
    auto size(){
      return _valuesOfVariables.size();
    }
    auto count(std::string const& variableName){
      return _valuesOfVariables.count(variableName);
    }
    void set(std::string const &key, int const &value){
      _valuesOfVariables[key] = value;
    }
    ~TwoStageDynamicBayesianNetworkState(){
    }
  private:
    std::map<std::string, int> &_valuesOfVariables;
};

#endif