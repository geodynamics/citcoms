
#include "global_defs.h"
#include "citcom_init.h"

struct All_variables *E;

void Citcom_Init()
{

  E = (struct All_variables*) malloc(sizeof(struct All_variables));
  fprintf(stderr,"Citcom_Init: address of E is %p\n",E);

  E->parallel.me = 0;
  E->parallel.nproc = 1;
  E->parallel.me_loc[1] = 0;
  E->parallel.me_loc[2] = 0;
  E->parallel.me_loc[3] = 0;

  E->monitor.solution_cycles=0;

  return;
}
