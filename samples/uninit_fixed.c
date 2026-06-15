/* CWE-457: Use of uninitialized variable — fixed */
#include <stdio.h>

int main(void) {
    int x = 0;
    printf("%d\n", x);
    return 0;
}
