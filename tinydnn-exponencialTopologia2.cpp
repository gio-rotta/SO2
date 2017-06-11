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

int main() {
	using namespace tiny_dnn;

	srand(time(NULL));
    const int DATA_NUMBER_ROWS = 100;
    const int BATCH_SIZE = 4;
    const int ITERATIONS = 200;

	int i;

	// Criação dos data set
    std::vector<vec_t> data;
    std::vector<vec_t> target;
    for (i = 0; i < DATA_NUMBER_ROWS; i++){
    	float a = rand() % 10;
    	float b = rand() % 10;
    	data.push_back( { a , b } );	
    }

    for (i = 0; i < DATA_NUMBER_ROWS; i++){
        float c = ( data.at(i)[0] * data.at(i)[0] ) + data.at(i)[1];
    	vec_t d = { c };
    	target.push_back(d);
    }

	network<sequential> net;
    net << layers::fc(2, 3) << activation::relu()
        << layers::fc(3, 1);
	
    // Criação de um dado experimental
    vec_t in = { 5 , 5 };     

	// Treinamento da rede
	size_t batch_size = BATCH_SIZE;
	size_t epochs = ITERATIONS;
	
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
    adagrad opt;
    // std::cout << "Initially maps 5 to " << net.predict(in)[0] << " but should map to 10" << std::endl;
    net.fit<mse>(opt, data, target, batch_size, epochs);
    // std::cout << "After training maps 5 to " << net.predict(in)[0] << " but should map to 10" << std::endl;

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

	// Salvar rede
	net.save("net");

	return 0;
}

//g++ -std=c++11 -I/home/giovanni/Documents/UFSC\ -\ CCO/SO2/tiny-dnn/ -O3 -pthread exponencialTopologia2.cpp -o example
