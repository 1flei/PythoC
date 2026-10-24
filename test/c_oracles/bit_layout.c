#include <stdalign.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct AllShort {
    unsigned short a : 3;
    unsigned short b : 5;
    unsigned short c : 6;
};

struct Mixed {
    unsigned short a : 3;
    unsigned int b : 5;
    unsigned short c : 6;
};

struct Boundary {
    unsigned short a : 3;
    unsigned int : 0;
    unsigned short b : 3;
};

struct LargeBits {
    uint64_t a : 64;
    uint64_t b : 64;
    uint64_t c : 64;
};

struct AllShort pc_bit_short(struct AllShort value) {
    value.a += 1;
    value.b += 2;
    value.c += 3;
    return value;
}

struct Mixed pc_bit_mixed(struct Mixed value) {
    value.a += 1;
    value.b += 2;
    value.c += 3;
    return value;
}

struct Boundary pc_bit_boundary(struct Boundary value) {
    value.a += 1;
    value.b += 2;
    return value;
}

struct LargeBits pc_bit_large(struct LargeBits value) {
    value.a += 1;
    value.b += 2;
    value.c += 3;
    return value;
}

static void print_bytes(const void *value, size_t size) {
    const unsigned char *bytes = value;
    for (size_t index = 0; index < size; ++index) {
        printf("%02x", bytes[index]);
    }
}

int main(void) {
    struct AllShort all_short;
    struct Mixed mixed;
    struct Boundary boundary;

    memset(&all_short, 0, sizeof(all_short));
    memset(&mixed, 0, sizeof(mixed));
    memset(&boundary, 0, sizeof(boundary));
    all_short.a = 5;
    all_short.b = 17;
    all_short.c = 33;
    mixed.a = 5;
    mixed.b = 17;
    mixed.c = 33;
    boundary.a = 5;
    boundary.b = 3;

    printf("all_short %zu %zu ", sizeof(all_short), alignof(struct AllShort));
    print_bytes(&all_short, sizeof(all_short));
    printf("\n");

    printf("mixed %zu %zu ", sizeof(mixed), alignof(struct Mixed));
    print_bytes(&mixed, sizeof(mixed));
    printf("\n");

    printf("boundary %zu %zu ", sizeof(boundary), alignof(struct Boundary));
    print_bytes(&boundary, sizeof(boundary));
    printf("\n");
    return 0;
}
