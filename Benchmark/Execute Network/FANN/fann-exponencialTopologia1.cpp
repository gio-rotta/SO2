
#include <cstdio>
#include <cmath>
#include "../../../fann/src/floatfann.c"
#include <typeinfo>
#include <iostream>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
#include "sys/vtimes.h"
#include "sys/types.h"
#include "sys/sysinfo.h"

const int DATA_NUMBER_ROWS = 1000;
//const int BATCH_SIZE = 10000;
const int ITERATIONS = 200;
const int HIDDEN_LAYERS = 2;


/* CPU UTILIZATION INIT*/;
static clock_t lastCPU, lastSysCPU, lastUserCPU;
static int numProcessors;

void init() {
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

int main()
{
   

    struct fann *ann;

    ann = fann_create_from_file("fann");
   
    init();
    clock_t begin = clock();

    fann_type *calc_out;
    float x = 5;
    fann_reset_MSE(ann);
    fann_scale_input( ann, &x );
    calc_out = fann_run( ann, &x );
    fann_descale_output( ann, calc_out );

    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("%f\n", time_spent);
    printf("%f\n", getCurrentValue());
    printf("%i\n", getValue());
    getValue();

    fann_save(ann, "fann");

	fann_destroy ( ann );

	return 0;
}
