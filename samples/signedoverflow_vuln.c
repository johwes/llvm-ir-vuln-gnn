/* CWE-190: Signed integer overflow */
#include <stdio.h>
#include <limits.h>

int main(void) {
    int a = INT_MAX;
    int b = a + 1;  /* signed overflow — undefined behaviour in C */
    printf("%d\n", b);
    return 0;
}
