
#include <cstdio>
#include <cmath>
#include "fann/floatfann.c"
#include <typeinfo>
#include <iostream>
const int DATA_NUMBER_ROWS = 100;
//const int BATCH_SIZE = 10000;
const int ITERATIONS = 500;
const int HIDDEN_LAYERS = 2;

float f(float x){
    float y;
     x>5 ? y = x*x : y =x +5;
    return y;
}

fann_train_data* createData()
{

	fann_train_data *trainingSet;
	trainingSet = fann_create_train ( DATA_NUMBER_ROWS, 1, 1 );

	for ( int i = 0; i<DATA_NUMBER_ROWS; i++ ) {
        float x = rand()%10;
		trainingSet->input[i][0] = x;
		trainingSet->output[i][0] = f ( x );
	}
	return trainingSet;
}

void testData(fann* ann){
	fann_type *calc_out;
	float x = 5;
	fann_reset_MSE(ann);
	fann_scale_input( ann, &x );
	calc_out = fann_run( ann, &x );
	fann_descale_output( ann, calc_out );
	printf("Result %f input %d\n",	calc_out[0], 5);
}

int main()
{
   // srand(time(0));
	const unsigned int num_input = 1;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = HIDDEN_LAYERS;

	struct fann *ann;

	ann = fann_create_standard ( num_layers, num_input, num_neurons_hidden, num_output );

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);

	fann_train_data *data = createData();

     /* CPU UTILIZATION INIT*/;
    static clock_t lastCPU, lastSysCPU, lastUserCPU;
    static int numProcessors;

    void init(){
        FILE* file;
        struct tms timeSample;
        char line[128];

        lastCPU = times(&timeSample);
        lastSysCPU = timeSample.tms_stime;
        lastUserCPU = timeSample.tms_utime;

        file = fopen("/proc/cpuinfo", "r");
        numProcessors = 0;
        while(fgets(line, 128, file) != NULL){
            if (strncmp(line, "processor", 9) == 0) numProcessors++;
        }
        fclose(file);
    }

    init();

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_set_scaling_params(  ann,	data, -1, 1, -1, 1);	/* New output maximum */

	fann_scale_train( ann, data );
	fann_train_on_data ( ann, data, ITERATIONS, 10, 1e-8f ); // epochs, epochs between reports, desired error

	  /* Total CPU used */

    double getCurrentValue(){
        struct tms timeSample;
        clock_t now;
        float percent;

        now = times(&timeSample);
        if (now <= lastCPU || timeSample.tms_stime < lastSysCPU ||
            timeSample.tms_utime < lastUserCPU){
            //Overflow detection. Just skip this value.
            percent = -1.0;
        }
        else{
            percent = (timeSample.tms_stime - lastSysCPU) +
                (timeSample.tms_utime - lastUserCPU);
            percent /= (now - lastCPU);
            percent /= numProcessors;
            percent *= 100;
        }
        lastCPU = now;
        lastSysCPU = timeSample.tms_stime;
        lastUserCPU = timeSample.tms_utime;

        return percent;
    }

    /* Total RAM memory */

    int parseLine(char* line){
        // This assumes that a digit will be found and the line ends in " Kb".
        int i = strlen(line);
        const char* p = line;
        while (*p <'0' || *p > '9') p++;
        line[i-3] = '\0';
        i = atoi(p);
        return i;
    }

    int getValue() { //Note: this value is in KB!
        FILE* file = fopen("/proc/self/status", "r");
        int result = -1;
        char line[128];

        while (fgets(line, 128, file) != NULL){
            if (strncmp(line, "VmSize:", 7) == 0){
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
        return result;
    }

    printf("%f\n", getCurrentValue());
    printf("%i\n", getValue());
    getValue();


    std::cout<< "Testing Data" <<std::endl;
    testData(ann);

	fann_destroy ( ann );
}
