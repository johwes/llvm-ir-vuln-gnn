/* CWE-369: Divide by zero */
#include <stdio.h>

int divide(int a, int b) {
    return a / b;  /* b may be zero */
}

int main(void) {
    printf("%d\n", divide(10, 0));
    return 0;
}
