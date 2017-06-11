#include <stdio.h>
#include <cmath>
#include <typeinfo>
#include <iostream>
#include <floatfann.h>
const int DATA_NUMBER_ROWS = 10;
const int BATCH_SIZE = 10000;
const int ITERATIONS = 200;
const int HIDDEN_LAYERS = 2;


float f(float x){
    return (x*x + 15);
}

fann_train_data* createData()
{
    const float angleRange = 3.0f;
	const float angleStep = 0.1;

	fann_train_data *trainingSet;
	trainingSet = fann_create_train ( DATA_NUMBER_ROWS, 1, 1 );

	for ( int i = 0; i<DATA_NUMBER_ROWS; i++ ) {
        float x = rand() % 10;
        printf("Input %f Output %f\n", x, f(x));
		trainingSet->input[i][0] = x;
		trainingSet->output[i][0] = f ( x );
	}
	return trainingSet;
}

void testData(fann* ann)
{
	fann_type *calc_out;
	unsigned int i;
	for(i = 0; i < 10; i++)
	{
		float x = i;
		fann_reset_MSE(ann);
    	fann_scale_input( ann, &x );
		calc_out = fann_run( ann, &x );
		fann_descale_output( ann, calc_out );
		printf("Result %f input %d\n",	calc_out[0], i);
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

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);

	fann_train_data *data = createData();
	fann_train_data *data2 = data;

	unsigned int i;
	printf("Length %d \n", fann_length_train_data(data));
	for(i = 0; i < fann_length_train_data(data); i++) {
		printf("Input %f Output %f\n", data->input[i][0], data->output[i][0]);
	}

	fann_set_scaling_params(  ann,	data, -1, 1, -1, 1);	/* New output maximum */

	fann_scale_train( ann, data );
	fann_train_on_data ( ann, data, ITERATIONS, 10, 1e-8f ); // epochs, epochs between reports, desired error

    std::cout<< "Testing Data" <<std::endl;
    testData(ann);

	fann_destroy ( ann );

	return 0;
}
