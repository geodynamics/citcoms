
#include "global_defs.h"
#include "citcom_init.h"

struct All_variables *E;

void Citcom_Init(int nproc, int rank)
{

  E = (struct All_variables*) malloc(sizeof(struct All_variables));
  //fprintf(stderr,"Citcom_Init: address of E is %p\n",E);

  E->parallel.nproc = nproc;
  E->parallel.me = rank;

  //fprintf(stderr,"%d in %d processpors\n", rank, nproc);

  E->monitor.solution_cycles=0;

  return;
}
