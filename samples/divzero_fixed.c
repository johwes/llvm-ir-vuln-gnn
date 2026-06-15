/* CWE-369: Divide by zero — fixed */
#include <stdio.h>
#include <stdlib.h>

int divide(int a, int b) {
    if (b == 0) {
        fprintf(stderr, "divide by zero\n");
        exit(1);
    }
    return a / b;
}

int main(void) {
    printf("%d\n", divide(10, 0));
    return 0;
}
