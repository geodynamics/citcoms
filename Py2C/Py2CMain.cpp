#include "Py2C.hpp"

int main(int argc, char* argv[])
{
  if( argc != 3)
  {
    std::cerr << "Usage: Py2C infile outfile" << std::endl;
    exit(-1);
  }

  Py2CConverter py2c(argv[1], argv[2]);
  py2c.convert();
}
