#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

union ScalarUnion {
    int32_t integer;
    double floating;
};

struct NaturalRecord {
    int8_t tag;
    union ScalarUnion value;
    int32_t items[2][3];
    int8_t tail;
};

typedef int32_t Vector4 __attribute__((vector_size(16)));

union VectorUnion {
    Vector4 vector;
    int8_t byte;
};

struct VectorRecord {
    int8_t tag;
    union VectorUnion value;
    int64_t tail;
};

int main(void)
{
    printf("%zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu %zu\n",
           sizeof(union ScalarUnion), _Alignof(union ScalarUnion),
           sizeof(struct NaturalRecord), _Alignof(struct NaturalRecord),
           offsetof(struct NaturalRecord, value),
           offsetof(struct NaturalRecord, items),
           offsetof(struct NaturalRecord, tail),
           sizeof(union VectorUnion), _Alignof(union VectorUnion),
           sizeof(struct VectorRecord), _Alignof(struct VectorRecord),
           offsetof(struct VectorRecord, tail));
    return 0;
}
