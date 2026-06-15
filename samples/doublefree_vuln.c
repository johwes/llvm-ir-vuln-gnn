/* CWE-415: Double free */
#include <stdlib.h>

int main(void) {
    int *p = (int *)malloc(sizeof(int));
    *p = 42;
    free(p);
    free(p);  /* double free */
    return 0;
}
