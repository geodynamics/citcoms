void general_stokes_solver_init(struct All_variables*);
void general_stokes_solver_fini(struct All_variables*);
void general_stokes_solver_update_velo(struct All_variables*);
void general_stokes_solver_Unorm(struct All_variables*, double*, double*);
void general_stokes_solver_log(struct All_variables*, float, float, int);
