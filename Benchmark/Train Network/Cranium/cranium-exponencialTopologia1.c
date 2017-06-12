#include "../../../Cranium/src/cranium.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
#include "sys/vtimes.h"
#include "sys/types.h"
#include "sys/sysinfo.h"

void generateResults(Network* networkReg) {

    FILE *f = fopen("craniumExp.txt", "w");
    if (f == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i <= 20; ++i)
    {
        float** oneEx = (float**)malloc(sizeof(float*));
        oneEx[0] = (float*)malloc(sizeof(float));
        oneEx[0][0] = i;
        DataSet* oneExData = createDataSet(1, 1, oneEx);

        forwardPassDataSet(networkReg, oneExData);   /* code */

        float e = i * i + 15;

        fprintf(f, "%f %f %f\n", oneEx[0][0], e, networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
        destroyDataSet(oneExData);
    }
    fclose(f);
}

int main(){

    srand(time(NULL));
    const int DATA_NUMBER_ROWS = 1000;
    const int BATCH_SIZE = 4;
    const int ITERATIONS = 200;

    int i;

    // Criação dos data set
    float** dataReg = (float**)malloc(sizeof(float*) * DATA_NUMBER_ROWS);
    float** classesReg = (float**)malloc(sizeof(float*) * DATA_NUMBER_ROWS);
    for (i = 0; i < DATA_NUMBER_ROWS; i++){
        dataReg[i] = (float*)malloc(sizeof(float) * 1);
        dataReg[i][0] = rand() % 20;
    }

    for (i = 0; i < DATA_NUMBER_ROWS; i++){
        classesReg[i] = (float*)malloc(sizeof(float) * 1);
        classesReg[i][0] = dataReg[i][0] * dataReg[i][0] + 15;
    }

    DataSet* trainingDataReg = createDataSet(DATA_NUMBER_ROWS, 1, dataReg);
    DataSet* trainingClassesReg = createDataSet(DATA_NUMBER_ROWS, 1, classesReg);

    // Criação da rede neural
    size_t hiddenSizeReg[] = {2}; 
    void (*hiddenActivationsReg[])(Matrix*) = {relu};
    Network* networkReg = createNetwork(1, 1, hiddenSizeReg, hiddenActivationsReg, 1, linear);

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

    // Treinamento da rede
    // printf("Initially maps 5 to %f but should map to 10\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
    batchGradientDescent(networkReg, trainingDataReg, trainingClassesReg, MEAN_SQUARED_ERROR, BATCH_SIZE, .01, 0, .001, .9, ITERATIONS, 1, 0);
    // printf("After training maps 5 to %f but should map to 10\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
    // generateResults(networkReg);  
    
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
    // Salvar rede
    saveNetwork(networkReg, "Cranium");

    // free network and data
    destroyNetwork(networkReg);
    destroyDataSet(trainingDataReg);
    destroyDataSet(trainingClassesReg);

    // load previous network from file
    // Network* previousNet = readNetwork("network");
    // destroyNetwork(previousNet);

    return 0;
    //gcc -std=c99 -Wall -Wno-unused-function -O3 -o example exponencialTopologia1.c -lm
};
