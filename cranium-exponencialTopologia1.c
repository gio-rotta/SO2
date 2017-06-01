#include "src/cranium.h"

int main(){

    srand(time(NULL));
    const int DATA_NUMBER_ROWS = 100;
    const int BATCH_SIZE = 4;
    const int ITERATIONS = 200;

    int i;

    // Criação dos data set
    float** dataReg = (float**)malloc(sizeof(float*) * DATA_NUMBER_ROWS);
    float** classesReg = (float**)malloc(sizeof(float*) * DATA_NUMBER_ROWS);
    for (i = 0; i < DATA_NUMBER_ROWS; i++){
        dataReg[i] = (float*)malloc(sizeof(float) * 1);
        dataReg[i][0] = rand() % 10;
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

    // Criação de um dado experimental
    float** oneEx = (float**)malloc(sizeof(float*));
    oneEx[0] = (float*)malloc(sizeof(float));
    oneEx[0][0] = 5.0;
    DataSet* oneExData = createDataSet(1, 1, oneEx);

    // Treinamento da rede
    forwardPassDataSet(networkReg, oneExData);
    printf("Initially maps 5 to %f but should map to 40\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
    batchGradientDescent(networkReg, trainingDataReg, trainingClassesReg, MEAN_SQUARED_ERROR, BATCH_SIZE, .01, 0, .001, .9, ITERATIONS, 1, 0);
    forwardPassDataSet(networkReg, oneExData);
    printf("After training maps 5 to %f but should map to 40\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
    
    // Salvar rede
    saveNetwork(networkReg, "REGRESSAO_PARABOLA_NETWORK");

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

/*
    ParameterSet params;
    params.network = networkReg;
    params.data = trainingDataReg;
    params.classes = trainingClassesReg;
    params.lossFunction = MEAN_SQUARED_ERROR;
    params.batchSize = BATCH_SIZE;
    params.learningRate = .01;
    params.searchTime = 0;
    params.regularizationStrength = .001;
    params.momentumFactor = .9;
    params.maxIters = ITERATIONS;
    params.shuffle = 1;
    params.verbose = 0;
*/