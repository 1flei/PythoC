#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

struct Pad { char pad[0]; };
struct Tail { char tag; int32_t pad[0]; };
struct Head { int32_t pad[0]; uint64_t value; };
struct End { uint64_t value; int32_t pad[0]; };
struct Middle { char tag; int32_t pad[0]; char tail; };

#define PRINT_LAYOUT(T) printf("%zu %zu\n", sizeof(T), _Alignof(T))

int main(void)
{
    PRINT_LAYOUT(struct Pad);
    PRINT_LAYOUT(struct Tail);
    PRINT_LAYOUT(struct Head);
    PRINT_LAYOUT(struct End);
    PRINT_LAYOUT(struct Middle);
    PRINT_LAYOUT(int32_t[3][0]);
    printf("%zu %zu\n", offsetof(struct Middle, pad),
           offsetof(struct Middle, tail));
    return 0;
}
