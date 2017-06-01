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
        dataReg[i] = (float*)malloc(sizeof(float) * 2);
        dataReg[i][0] = rand() % 10;
        dataReg[i][1] = rand() % 10;
    }

    for (i = 0; i < DATA_NUMBER_ROWS; i++){
        classesReg[i] = (float*)malloc(sizeof(float) * 1);
        classesReg[i][0] = (dataReg[i][0] * dataReg[i][0]) + dataReg[i][1];
    }

    DataSet* trainingDataReg = createDataSet(DATA_NUMBER_ROWS, 2, dataReg);
    DataSet* trainingClassesReg = createDataSet(DATA_NUMBER_ROWS, 1, classesReg);

    // Criação da rede neural
    size_t hiddenSizeReg[] = {2};
    void (*hiddenActivationsReg[])(Matrix*) = {relu};
    Network* networkReg = createNetwork(2, 1, hiddenSizeReg, hiddenActivationsReg, 1, linear);

    // Criação de um dado experimental
    float** oneEx = (float**)malloc(sizeof(float*));
    oneEx[0] = (float*)malloc(sizeof(float) * 2);
    oneEx[0][0] = 5.0;
    oneEx[0][1] = 5.0;
    DataSet* oneExData = createDataSet(1, 2, oneEx);

    // Treinamento da rede
    forwardPassDataSet(networkReg, oneExData);
    printf("Initially maps 5 to x and 5 to y, %f but should map to 30\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
    batchGradientDescent(networkReg, trainingDataReg, trainingClassesReg, MEAN_SQUARED_ERROR, BATCH_SIZE, .01, 0, .001, .9, ITERATIONS, 1, 0);
    forwardPassDataSet(networkReg, oneExData);
    printf("After training maps 5 to x and 5 to y, %f but should map to 30\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0]);
    
    // Salvar rede
    saveNetwork(networkReg, "REGRESSAO_PARABOLA_NETWORK2");

    // free network and data
    destroyNetwork(networkReg);
    destroyDataSet(trainingDataReg);
    destroyDataSet(trainingClassesReg);

    // load previous network from file
    // Network* previousNet = readNetwork("network");
    // destroyNetwork(previousNet);

    return 0;
};
