#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t tag;
    uint32_t value;
} Natural;

#pragma pack(push, 1)
typedef struct {
    uint8_t tag;
    uint64_t value;
} Packed1;

typedef union {
    uint64_t value;
    uint8_t bytes[8];
} PackedUnion;

typedef struct {
    uint8_t tag;
    PackedUnion data;
} WithUnion;

typedef struct {
    float value;
} FloatBox;
#pragma pack(pop)

#pragma pack(push, 2)
typedef struct {
    uint8_t tag;
    uint32_t value;
} Packed2;
#pragma pack(pop)

typedef struct __attribute__((aligned(16))) {
    uint32_t value;
} Aligned;

typedef struct __attribute__((packed, aligned(4))) {
    uint8_t tag;
    uint32_t value;
} PackedAligned;

typedef struct {
    uint8_t tag;
    Packed2 values[2];
    uint8_t tail;
} Nested;

typedef struct {
    uint8_t tag;
    Aligned value;
    uint8_t tail;
} NestedAligned;

typedef struct __attribute__((aligned(8))) {
    uint32_t value;
} AlignedMember;

typedef struct __attribute__((packed)) {
    uint32_t tag;
    AlignedMember data;
} PackedOuter;

#define LAYOUT(T, member) { sizeof(T), _Alignof(T), offsetof(T, member) }
static const uint64_t layout[][3] = {
    LAYOUT(Natural, value),
    LAYOUT(Packed1, value),
    LAYOUT(Packed2, value),
    LAYOUT(Aligned, value),
    LAYOUT(PackedAligned, value),
    LAYOUT(Nested, tail),
    LAYOUT(NestedAligned, tail),
    LAYOUT(PackedUnion, value),
    LAYOUT(WithUnion, data.value),
    LAYOUT(FloatBox, value),
    LAYOUT(PackedOuter, data.value),
};

uint64_t pc_record_metric(int record, int metric) {
    return layout[record][metric];
}

#define STEP(T, member, delta)                             \
    T pc_step_##T(T value) {                                \
        value.member += delta;                             \
        return value;                                      \
    }                                                      \
    T pc_callback_##T(T (*callback)(T), T value) {           \
        return callback(pc_step_##T(value));                \
    }

STEP(Natural, value, 1)
STEP(Packed1, value, 2)
STEP(Packed2, value, 3)
STEP(WithUnion, data.value, 4)
STEP(Aligned, value, 5)
STEP(PackedAligned, value, 6)
STEP(FloatBox, value, 0.5f)
STEP(PackedOuter, data.value, 12)

typedef union { double value; } FloatUnion;
typedef union { uint64_t value; } IntegerUnion;

STEP(FloatUnion, value, 0.5)
STEP(IntegerUnion, value, 2)

typedef struct {
    uint64_t left, right;
} IntegerLanes;

typedef struct {
    uint64_t left;
    double right;
} MixedLanes;

typedef struct {
    uint64_t left, right, tail;
} IndirectResult;

uint64_t pc_stack_integer(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                          uint64_t e, IntegerLanes value, uint64_t tail) {
    return 10000 * value.left + 100 * value.right + tail;
}

uint64_t pc_callback_integer(
    uint64_t (*callback)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                         IntegerLanes, uint64_t)) {
    return callback(1, 2, 3, 4, 5, (IntegerLanes){10, 20}, 12);
}

uint64_t pc_stack_mixed(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                        uint64_t e, uint64_t f, MixedLanes value, double tail) {
    return 10000 * value.left + 100 * (uint64_t)value.right + (uint64_t)tail;
}

uint64_t pc_callback_mixed(
    uint64_t (*callback)(uint64_t, uint64_t, uint64_t, uint64_t, uint64_t,
                         uint64_t, MixedLanes, double)) {
    return callback(1, 2, 3, 4, 5, 6, (MixedLanes){10, 20.0}, 12.0);
}

IndirectResult pc_stack_sret(uint64_t a, uint64_t b, uint64_t c, uint64_t d,
                             IntegerLanes value, uint64_t tail) {
    return (IndirectResult){value.left, value.right, tail};
}

IndirectResult pc_callback_sret(
    IndirectResult (*callback)(uint64_t, uint64_t, uint64_t, uint64_t,
                               IntegerLanes, uint64_t)) {
    return callback(1, 2, 3, 4, (IntegerLanes){10, 20}, 12);
}

typedef float Vector128 __attribute__((vector_size(16)));
typedef float Vector64 __attribute__((vector_size(8)));
typedef struct { Vector128 lanes; } VectorRecord;
typedef struct { Vector64 first, second; } VectorPair;
typedef union { Vector128 lanes; uint64_t integer; } VectorUnion;

VectorRecord pc_step_vector(VectorRecord value, float tail) {
    value.lanes += (Vector128){tail, tail + 1, tail + 2, tail + 3};
    return value;
}

VectorRecord pc_callback_vector(VectorRecord (*callback)(VectorRecord, float),
                                 VectorRecord value, float tail) {
    return callback(pc_step_vector(value, tail), tail + 1);
}

VectorPair pc_step_vector_pair(VectorPair value, float tail) {
    value.first[1] += tail;
    value.second[0] += 2 * tail;
    return value;
}

int pc_has_vector_pair_abi(void) {
#if defined(__x86_64__) && !defined(_WIN32) && defined(__GNUC__) && \
    !defined(__clang__) && __GNUC__ < 11
    /* GCC 11.1 changed the SysV ABI of aggregates of 64-bit float vectors. */
    return 0;
#else
    return 1;
#endif
}

VectorUnion pc_step_vector_union(VectorUnion value, float tail) {
    value.lanes[3] += tail;
    return value;
}

static float score_vector(VectorRecord value, float tail) {
    return 1000 * value.lanes[0] + 10 * value.lanes[2] + value.lanes[3] + tail;
}

float pc_last_vector_register(double a, double b, double c, double d, double e,
                               double f, double g, VectorRecord value, float tail) {
    return score_vector(value, tail);
}

float pc_spilled_vector(double a, double b, double c, double d, double e, double f,
                        double g, double h, VectorRecord value, float tail) {
    return score_vector(value, tail);
}
