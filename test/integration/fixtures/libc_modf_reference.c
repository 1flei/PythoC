#include <math.h>

double split_reference(double value, double *integral)
{
    return modf(value, integral);
}
