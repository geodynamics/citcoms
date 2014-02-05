#ifndef __citcoms__py2c__hpp__
#define __citcoms__py2c__hpp__
//------------------------------------------------------------------------------
// Py2C.cpp : convert Python based CitcomS config files to C-based versions
//
// parsing rules for .cfg files
// [0] Blank lines are ignored
// [1] If line starts with a #, ignore the rest of the line
// [2] If line starts with [, look for the closing ] and everything in between 
//     is the name of a section.
// [3] If a line contains a ';', ignore everything after the ';'
// [4] Parse name = value pairs
//------------------------------------------------------------------------------
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <list>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

struct Parameter
{
  Parameter();
  Parameter(const char* val, const char* sec, bool cReq=false,bool isDef=true); 

  std::string value;      // string representation of the Parameter value
  bool isDefault;         // true if the value has not been set from the .cfg file
  bool cRequired;         // values whose presence is mandatory in C config files
  std::string section;    // the section in which this Parameter is to be listed
};


class Py2CConverter
{
public:
  Py2CConverter(const char* pycfgfile, const char* ccfgfile);
  ~Py2CConverter();
  
  void convert();
private:
  void initialize_parameters();
  bool parse(const std::string& name, const std::string& value);
  void check_and_fix_errors();
  void save();
  void load();
  
  std::map<std::string, Parameter> parameters;
  std::set<std::string> section_names;
  std::list<std::string> log_messages;
  
  std::ifstream fin;
  std::ofstream fout;
};

#endif // __citcoms__py2c__hpp__
