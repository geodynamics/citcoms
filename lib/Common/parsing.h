void setup_parser(struct All_variables *E, char *filename);
void shutdown_parser(struct All_variables *E);

int input_string(char *name, char *value, char *Default, int m);
int input_boolean(char *name, int *value, char *interpret, int m);
int input_int(char *name, int *value, char *interpret, int m);
int input_float(char *name, float *value, char *interpret, int m);
int input_double(char *name, double *value, char *interpret, int m);
int input_int_vector(char *name, int number, int *value, int m);
int input_char_vector(char *name, int number, char *value, int m);
int input_float_vector(char *name,int number, float *value, int m);
int input_double_vector(char *name, int number, double *value, int m);
