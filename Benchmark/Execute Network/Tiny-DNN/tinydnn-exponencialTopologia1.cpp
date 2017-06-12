#include <iostream>
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "sys/times.h"
#include "sys/vtimes.h"
#include "sys/types.h"
#include "sys/sysinfo.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace std;

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

void testData(network<sequential> net){
     FILE *fp = fopen("TinyExp.txt", "w");
    if (fp == NULL)
    {
        printf("Error opening file!\n");
        exit(1);
    }

    for (int i = 0; i <= 20; ++i)
    {
        vec_t in = { i };  

        int d = (i * i) + 15;
        
        fprintf(fp, "%d %d %f\n", i, d, net.predict(in)[0]);
    }
    fclose(fp);
}

int main() {
	using namespace tiny_dnn;

	srand(time(NULL));
    const int DATA_NUMBER_ROWS = 1000;
    const int BATCH_SIZE = 4;
    const int ITERATIONS = 200;

	int i;

	// Criação dos data set
    std::vector<vec_t> data;
    std::vector<vec_t> target;
    for (i = 0; i < DATA_NUMBER_ROWS; i++){
    	float r = rand() % 20;
    	data.push_back( { r } );	
    }

    for (i = 0; i < DATA_NUMBER_ROWS; i++){
    	vec_t d = { ( data.at(i)[0] * data.at(i)[0] ) + 15 };
    	target.push_back(d);
    }
   
    // Criação da rede neural
    network<sequential> net;
    net << layers::fc(1, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 100) << activation::relu() 
    << layers::fc(100, 1);


	// Treinamento da rede
    size_t batch_size = BATCH_SIZE;
    size_t epochs = ITERATIONS;


    // Treinamento da rede
    adagrad opt;
    // std::cout << "Initially maps 5 to " << net.predict(in)[0] << " but should map to 10" << std::endl;
    net.fit<mse>(opt, data, target, batch_size, epochs);
    // std::cout << "After training maps 5 to " << net.predict(in)[0] << " but should map to 10" << std::endl;
    // Criação de um dado experimental
    vec_t in = { 5.0 };     

    init();
    
    float test =  net.predict(in)[0];

    printf("%f\n", getCurrentValue());
    printf("%i\n", getValue());
    getValue();

    // testData(net);
    net.save("tinydnn");

	return 0;
}

//g++ -std=c++11 -I/home/giovanni/Documents/UFSC\ -\ CCO/SO2/tiny-dnn/ -O3 -pthread exponencialTopologia1.cpp -o example
