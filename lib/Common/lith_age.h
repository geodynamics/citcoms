void lith_age_input(struct All_variables *E);
void lith_age_init(struct All_variables *E);
void lith_age_restart_tic(struct All_variables *E);
void lith_age_construct_tic(struct All_variables *E) ;
void lith_age_read_files(struct All_variables *E, int output);
void lith_age_temperature_bound_adj(struct All_variables *E, int lv);
void lith_age_conform_tbc(struct All_variables *E);
