#include "Py2C.hpp"

const char* usage = "Usage: Py2C infile outfile save_all\n\n"
  "Where\n\n"
  "infile is the name of Python version of the configuration file\n\n"
  "outfile is the name of the C version of the configuration file\n\n"
  "save_all=False means outfile will only have parameters that were set in infile\n\n"
  "save_all=True means outfile will have all CitcomS parameters, grouped by the Python section names\n";

int main(int argc, char* argv[])
{
  if( argc != 4)
  {
    std::cerr << usage << std::endl;
    exit(-1);
  }

  Py2CConverter py2c(argv[1], argv[2]);

  std::string saveallstr(argv[3]);
  std::transform(saveallstr.begin(), saveallstr.end(), saveallstr.begin(), ::tolower);

  bool saveall = ((saveallstr == "true") ? true : false);
  py2c.convert(saveall);
}
