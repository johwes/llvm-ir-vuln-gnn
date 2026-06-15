/* CWE-415: Double free — fixed */
#include <stdlib.h>

int main(void) {
    int *p = (int *)malloc(sizeof(int));
    *p = 42;
    free(p);
    p = NULL;       /* nullify before second free — free(NULL) is a no-op */
    free(p);
    return 0;
}
