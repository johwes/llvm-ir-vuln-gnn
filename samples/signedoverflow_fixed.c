/* CWE-190: Signed integer overflow — fixed */
#include <stdio.h>
#include <limits.h>

int main(void) {
    int a = INT_MAX;
    int b = (a == INT_MAX) ? INT_MAX : a + 1;
    printf("%d\n", b);
    return 0;
}
