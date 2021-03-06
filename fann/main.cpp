#include <cstdio>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <fann.h>
#include <floatfann.h>
const int DATA_NUMBER_ROWS = 10;
const int BATCH_SIZE = 10000;
const int ITERATIONS = 200;
const int HIDDEN_LAYERS = 2;

#define fann_sigmoid_symmetric_real(sum) (2.0f/(1.0f + exp(-2.0f * sum)) - 1.0f)
//const float max = (15*15)+15;

float f(float x){
    return (x*x +15);
}

fann_train_data* createData()
{
    const float angleRange = 3.0f;
	const float angleStep = 0.1;

	fann_train_data *trainingSet;
	trainingSet = fann_create_train ( DATA_NUMBER_ROWS, 1, 1 );

	for ( int i = 0; i<DATA_NUMBER_ROWS; i++ ) {
        float x = rand()%10;
		trainingSet->input[i][0] = x;
		trainingSet->output[i][0] = f ( x );
	}
	return trainingSet;
}

void testData(fann* ann)
{
    fann_type a = 5;
    std::cout<< typeid(a).name() <<std::endl;

    for ( int i = -5; i<15; i++ ) {
        float x = i;
		float *o = fann_run(ann, &x);
		std::cout<< "f("<<x<<")="<<f(x)<<"  received:"<<*o<< "  error:"<< abs( f(x)- *o ) <<std::endl;
	}

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

	fann_set_activation_function_hidden ( ann, FANN_SIGMOID_SYMMETRIC );
	fann_set_activation_function_output ( ann, FANN_SIGMOID_SYMMETRIC );

	fann_set_train_stop_function ( ann, FANN_STOPFUNC_BIT );
	fann_set_bit_fail_limit ( ann, 0.02f );

	fann_set_training_algorithm ( ann, FANN_TRAIN_BATCH );

	fann_randomize_weights ( ann, 0, 1 );

	fann_train_data *trainingSet = createData();

	fann_train_on_data ( ann, trainingSet, ITERATIONS, 10, 1e-8f ); // epochs, epochs between reports, desired error

    std::cout<< "Testing Data" <<std::endl;
    testData(ann);


	fann_destroy ( ann );

	return 0;
}
