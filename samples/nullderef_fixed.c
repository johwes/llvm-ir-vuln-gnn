/* CWE-476: NULL pointer dereference — fixed */
#include <stdio.h>

int main(void) {
    int *p = NULL;
    if (p != NULL)
        printf("%d\n", *p);
    return 0;
}
