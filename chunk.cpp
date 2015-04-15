// Per http://stackoverflow.com/a/21661547/591483 , we can't use C++11 stuff in .cu files. Therefore, we separate this out.

#include <atomic>

std::atomic<unsigned> chunk_id;

void chunk_reset(void)
{
	chunk_id = 0;
}

unsigned chunk_get(void)
{
	return chunk_id.fetch_add(1);
}

