
/*************************************************************************/
/* from Process_buoyancy.c                                               */
/*************************************************************************/


void process_temp_field(E,ii)
 struct All_variables *E;
    int ii;
{
    void heat_flux();
    void output_temp();
    void parallel_process_sync();
    void process_output_field();
    int record_h;

    record_h = E->control.record_every;

    if ( (ii == 0) || ((ii % record_h) == 0) || E->control.DIRECTII)    {
      heat_flux(E);
      parallel_process_sync();
/*      output_temp(E,ii);  */
    }

    if ( ((ii == 0) || ((ii % E->control.record_every) == 0))
	 || E->control.DIRECTII)     {
       process_output_field(E,ii);
    }

    return;
}



/*************************************************************************/
/* from Global_operations.c                                              */
/*************************************************************************/

void sum_across_depth_sph1(E,sphc,sphs)
struct All_variables *E;
float *sphc,*sphs;
{
 int jumpp,total,j,d;

 static float *sphcs,*temp;
 static int been_here=0;
 static int *processors,nproc;

 static MPI_Comm world, horizon_p;
 static MPI_Group world_g, horizon_g;

if (been_here==0)  {
 processors = (int *)malloc((E->parallel.nprocz+2)*sizeof(int));
 temp = (float *) malloc((E->sphere.hindice*2+3)*sizeof(float));
 sphcs = (float *) malloc((E->sphere.hindice*2+3)*sizeof(float));

 nproc = 0;
 for (j=0;j<E->parallel.nprocz;j++) {
   d =E->parallel.me_sph*E->parallel.nprocz+E->parallel.nprocz-1-j; 
   processors[nproc] =  d;
   nproc ++;
   }

 if (nproc>0)  {
    world = E->parallel.world;
    MPI_Comm_group(world, &world_g);
    MPI_Group_incl(world_g, nproc, processors, &horizon_g);
    MPI_Comm_create(world, horizon_g, &horizon_p);
    }

 been_here++;
 }

 total = E->sphere.hindice*2+3;
  jumpp = E->sphere.hindice;
  for (j=0;j<E->sphere.hindice;j++)   {
      sphcs[j] = sphc[j];
      sphcs[j+jumpp] = sphs[j];
     }


 if (nproc>0)  {

    MPI_Allreduce(sphcs,temp,total,MPI_FLOAT,MPI_SUM,horizon_p);

    for (j=0;j<E->sphere.hindice;j++)   {
      sphc[j] = temp[j];
      sphs[j] = temp[j+jumpp];
     }

    }

return;
}


/*************************************************************************/
/* from                                                                  */
/*************************************************************************/

