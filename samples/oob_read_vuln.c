/* CWE-125: Out-of-bounds read */
#include <stdio.h>

int main(void) {
    int arr[8] = {0};
    int idx = 10;
    printf("%d\n", arr[idx]);  /* index out of bounds */
    return 0;
}
