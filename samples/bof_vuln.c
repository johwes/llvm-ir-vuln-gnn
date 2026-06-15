/* CWE-121: Stack-based buffer overflow */
#include <string.h>

void copy_input(const char *input) {
    char buf[16];
    strcpy(buf, input);  /* no bounds check */
}

int main(void) {
    copy_input("this string is longer than sixteen bytes!!");
    return 0;
}
