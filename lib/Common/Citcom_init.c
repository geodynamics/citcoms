
#include "global_defs.h"
#include "citcom_init.h"

struct All_variables *E;

void Citcom_Init()
{

  E = (struct All_variables*) malloc(sizeof(struct All_variables));
  fprintf(stderr,"Citcom_Init: address of E is %p\n",E);

  return;
}
