
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
/* from                                                                  */
/*************************************************************************/


