#include <math.h>
#include <stdint.h>
#include <string.h>

int finite32(uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return !!isfinite(value);
}

int finite64(uint64_t bits)
{
    double value;
    memcpy(&value, &bits, sizeof(value));
    return !!isfinite(value);
}
