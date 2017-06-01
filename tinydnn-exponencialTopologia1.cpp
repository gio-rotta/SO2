#include <iostream>

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
    	float r = rand() % 10;
    	data.push_back( { r } );	
    }

    for (i = 0; i < DATA_NUMBER_ROWS; i++){
    	vec_t d = { ( data.at(i)[0] * data.at(i)[0] ) + 15 };
    	target.push_back(d);
    }

	// Criação da rede neural
    network<sequential> net;
	net << layers::fc(1, 2) << activation::relu() 
    	<< layers::fc(2, 1);

	// Criação de um dado experimental
    vec_t in = { 5.0 };     

	// Treinamento da rede
    size_t batch_size = BATCH_SIZE;
    size_t epochs = ITERATIONS;
	adagrad opt;
	std::cout << "Initially maps 5 to " << net.predict(in)[0] << " but should map to 40" << std::endl;
    net.fit<mse>(opt, data, target, batch_size, epochs);
	std::cout << "After training maps 5 to " << net.predict(in)[0] << " but should map to 40" << std::endl;

	// Salvar rede
	net.save("net");

	return 0;
}

//g++ -std=c++11 -I/home/giovanni/Documents/UFSC\ -\ CCO/SO2/tiny-dnn/ -O3 -pthread exponencialTopologia1.cpp -o example
