void parallel_process_termination();
void parallel_process_sync(struct All_variables *E);
double CPU_time0();
void parallel_processor_setup(struct All_variables *E);
void parallel_domain_decomp0(struct All_variables *E);
void parallel_domain_boundary_nodes(struct All_variables *E);
void parallel_communication_routs_v(struct All_variables *E);
void parallel_communication_routs_s(struct All_variables *E);
void set_communication_sphereh(struct All_variables *E);
void exchange_id_d(struct All_variables *E, double **U, int lev);
void exchange_node_d(struct All_variables *E, double **U, int lev);
void exchange_node_f(struct All_variables *E, float **U, int lev);
void exchange_snode_f(struct All_variables *E, float **U1, float **U2, int lev);

