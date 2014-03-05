#include "Py2C.hpp"

int main(int argc, char* argv[])
{
  if( argc != 4)
  {
    std::cerr << "Usage: Py2C infile outfile true|false" << std::endl;
    exit(-1);
  }

  Py2CConverter py2c(argv[1], argv[2]);

  bool save_all = (std::string(argv[3])=="true"?true:false);
  py2c.convert(save_all);
}
