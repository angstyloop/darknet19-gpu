# darknet-c-api-example

This project uses a shared library libdarknet.so built from a clone of https://github.com/pjreddie/darknet.git, after making the following changes. 

- I removed some printf statements from parser.c, softmax_layer.c, avgpool_layer.c, and maxpool_layer.c. 
- I copied examples/predict_classifier.c and modified it, so that it no longer prints, and returns a dynamically allocated array of result_t, wheree result_t has the form { float accuracy, char* name }.

# darknet19-gpu
