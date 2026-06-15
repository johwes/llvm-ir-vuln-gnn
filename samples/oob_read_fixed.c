/* CWE-125: Out-of-bounds read — fixed */
#include <stdio.h>

int main(void) {
    int arr[8] = {0};
    int idx = 10;
    if (idx >= 0 && idx < 8)
        printf("%d\n", arr[idx]);
    return 0;
}
