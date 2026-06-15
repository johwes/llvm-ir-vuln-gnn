/* CWE-476: NULL pointer dereference */
#include <stdio.h>

int main(void) {
    int *p = NULL;
    printf("%d\n", *p);  /* unconditional null dereference */
    return 0;
}
