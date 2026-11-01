#include <stdint.h>

struct Node { int64_t value; };
struct Holder { struct Node *next; };

int64_t read_second(struct Holder holder)
{
    return holder.next[1].value;
}

void set_holder(struct Holder *holder, struct Node *node)
{
    holder->next = node;
}
