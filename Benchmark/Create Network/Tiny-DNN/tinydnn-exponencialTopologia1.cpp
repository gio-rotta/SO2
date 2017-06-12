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

int main() {
	using namespace tiny_dnn;

    init();
	
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

    printf("%f\n", getCurrentValue());
    printf("%i\n", getValue());
    getValue();

    // Salvar rede
    net.save("net");

    return 0;

}

//g++ -std=c++11 -I/home/giovanni/Documents/UFSC\ -\ CCO/SO2/tiny-dnn/ -O3 -pthread exponencialTopologia1.cpp -o example
