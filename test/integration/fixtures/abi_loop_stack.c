#include <stdint.h>
struct Record { int64_t fields[8]; };
struct Record make_stack_record(int32_t value) {
    struct Record result;
    for (int i = 0; i < 8; ++i) result.fields[i] = value + i;
    return result;
}
int64_t sum_stack_record(struct Record value) {
    return value.fields[0] + value.fields[7];
}
