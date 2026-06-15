/* CWE-121: Stack-based buffer overflow — fixed */
#include <string.h>

void copy_input(const char *input) {
    char buf[16];
    strncpy(buf, input, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
}

int main(void) {
    copy_input("this string is longer than sixteen bytes!!");
    return 0;
}
