
#include "global_defs.h"
#include "citcom_init.h"

struct All_variables *E;

void Citcom_Init(MPI_Comm *world)
{
  int rank, nproc;

  E = (struct All_variables*) malloc(sizeof(struct All_variables));
  //fprintf(stderr,"Citcom_Init: address of E is %p\n",E);

  MPI_Comm_rank(*world, &rank);
  MPI_Comm_size(*world, &nproc);

  E->parallel.world = *world;
  E->parallel.nproc = nproc;
  E->parallel.me = rank;

  fprintf(stderr,"%d in %d processpors\n", rank, nproc);

  E->monitor.solution_cycles=0;
  E->control.keep_going=1;

  return;
}
