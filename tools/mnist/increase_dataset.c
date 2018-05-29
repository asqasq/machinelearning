#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BUFFERSIZE (4 * 1024*1024*1024UL)
int main(int argc, char **argv)
{
    if (argc != 4) {
        printf("\nUsage: %s <nr_copies> <trainingimages> <traininglabels>\n", argv[0]);
        return 2;
    }

    int copies = atoi(argv[1]);
    
    printf("Copies = %d\n", copies);
    
    if (copies < 0) {
        printf("\n copies must be >= 0\n");
        return 1;
    }

    unsigned char *buffer = (unsigned char*)malloc(BUFFERSIZE);
    
/******************************************************************************/    
/* training images                                                            */
/******************************************************************************/    
    FILE *f = fopen(argv[2], "r");
    fread(buffer, BUFFERSIZE, 1, f);
    
    int idx = 0;
    
    unsigned int header = 0;
    for (int i = 0; i < 4; i++) {
        header = header << 8;
        header |= buffer[idx++];
    }
    unsigned int nr_images = 0;
    for (int i = 0; i < 4; i++) {
        nr_images = nr_images << 8;
        nr_images |= buffer[idx++];
    }
    
    unsigned int nr_rows = 0;
    for (int i = 0; i < 4; i++) {
        nr_rows = nr_rows << 8;
        nr_rows |= buffer[idx++];
    }
    unsigned int nr_cols = 0;
    for (int i = 0; i < 4; i++) {
        nr_cols = nr_cols << 8;
        nr_cols |= buffer[idx++];
    }
    
    printf("\nheader = 0x%08x, nr images = %u, nr rows = %u, nr columns = %u\n",
        header, nr_images, nr_rows, nr_cols);
    
    unsigned int dataoffset = idx;
    unsigned int singledatasize = nr_rows * nr_cols * nr_images;
    printf("\nsingle data size = %u\n", singledatasize);
    for (int i = 0; i < (copies - 1); i++) {
        memcpy(&(buffer[(i + 1) * singledatasize + dataoffset]), &(buffer[dataoffset]), singledatasize);
    }
    nr_images *= copies;
    
    printf("\n Data copies in total: %u\n", nr_images);
    
    for (int i = 3; i >= 0; i--) {
        buffer[i + 4] = (unsigned char)(nr_images & 0xff);
        nr_images >>= 8;
    }
    
    nr_images = 0;
    for (int i = 0; i < 4; i++) {
        nr_images = nr_images << 8;
        nr_images |= buffer[i + 4];
    }
    printf("nr images new = %u\n", nr_images);
    
    
    char filenamenew[1024];
    snprintf(filenamenew, 1024, "%s-%d",argv[2], copies);
    f = fopen(filenamenew, "w");
    
    fwrite(buffer, singledatasize * copies + dataoffset, 1, f);
    fclose(f);
    
/******************************************************************************/    
/* training labels                                                            */
/******************************************************************************/    
    f = fopen(argv[3], "r");
    fread(buffer, BUFFERSIZE, 1, f);
    
    idx = 0;
    
    header = 0;
    for (int i = 0; i < 4; i++) {
        header = header << 8;
        header |= buffer[idx++];
    }
    unsigned int nr_labels = 0;
    for (int i = 0; i < 4; i++) {
        nr_labels = nr_labels << 8;
        nr_labels |= buffer[idx++];
    }
    
    printf("\nheader = 0x%08x, nr labels = %u\n",
        header, nr_labels);
    
    dataoffset = idx;
    singledatasize = 1 * nr_labels;
    printf("\nsingle data size = %u\n", singledatasize);
    for (int i = 0; i < (copies - 1); i++) {
        memcpy(&(buffer[(i + 1) * singledatasize + dataoffset]), &(buffer[dataoffset]), singledatasize);
    }
    nr_labels *= copies;
    
    printf("\n Data copies in total: %u\n", nr_labels);
    
    for (int i = 3; i >= 0; i--) {
        buffer[i + 4] = (unsigned char)(nr_labels & 0xff);
        nr_labels >>= 8;
    }
    
    nr_labels = 0;
    for (int i = 0; i < 4; i++) {
        nr_labels = nr_labels << 8;
        nr_labels |= buffer[i + 4];
    }
    printf("nr labels new = %u\n", nr_labels);
    
    
    snprintf(filenamenew, 1024, "%s-%d",argv[3], copies);
    f = fopen(filenamenew, "w");
    
    fwrite(buffer, singledatasize * copies + dataoffset, 1, f);
    fclose(f);
    
    
    return 0;
}

