#include "include/darknet.h"
#include <assert.h>

void predict_classifier_multi(char *datacfg, char *cfgfile, char *weightfile, char **filenames, int n_files, int top) {

    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);

    list *options = read_data_cfg(datacfg);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", "data/labels.list");
    if(top==0) top = option_find_int(options, "top", 1);

    char **names = get_labels(name_list);
    int *indexes = calloc(top, sizeof(int));
    char buff[256];
    char *input = buff;

    for (int i=0; i<n_files; ++i) {
        char* filename = filenames[i];
        if(filename){
            strncpy(input, filename, 256);
            printf("%s\n", input);
            image im = load_image_color(input, 0, 0);
            image r = letterbox_image(im, net->w, net->h);
            float  *X = r.data;
            float *predictions = network_predict(net, X);
            if(net->hierarchy) hierarchy_predictions(predictions, net->outputs, net->hierarchy, 1, 1);
            top_k(predictions, net->outputs, top, indexes);
            int j=0;
            for (j=0; j<top; ++j) {
                int index = indexes[j];
                printf("%.2f %s\n", predictions[index]*100, names[index]);
            }
            if(r.data!=im.data) free_image(r);
            free_image(im);
        }
    }
}

/* drives classification */
int main(int argc, char** argv) { 
    char* data = argv[1];
    char* cfg = argv[2];
    char* weights = argv[3];
    int top = atoi(argv[4]);
    int n_images = argc - 5;
    char** images = malloc(n_images*sizeof(char*));
    for (int i=0; i<n_images; ++i) {
        images[i] = malloc(256*sizeof(char));
    }
    // copy trailing arguments to array of image paths
    memcpy(images, argv+5, n_images*sizeof(char*));
    predict_classifier_multi(data, cfg, weights, images, n_images, top);
    return 0; 
}
