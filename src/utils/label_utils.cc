#include "label_utils.h"

#include <fstream>

#include <map>
#include <regex>
#include <string>

using std::getline;
using std::ifstream;
using std::istringstream;
using std::map;
using std::regex;
using std::regex_replace;
using std::string;

namespace edge {

map<int, std::string> ParseLabel(const string& label_path) {
  map<int, string> ret;
  std::ifstream label_file(label_path);
  if (!label_file.good()) return ret;
  for (std::string line; std::getline(label_file, line);) {
    std::istringstream ss(line);
    int id;
    ss >> id;
    line = std::regex_replace(line, std::regex("^ +[0-9]+ +"), "");
    ret.emplace(id,line);
  }
  return ret;
}

}  // namespace edge
