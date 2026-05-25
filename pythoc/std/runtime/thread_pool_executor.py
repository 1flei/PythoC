"""Thread-pool executor backend for Future."""
from __future__ import annotations

from .policy import bind_mem
bind_mem()

from types import SimpleNamespace
from pythoc import (
    compile, effect, ptr, void, func, u64, i32, i64, nullptr, sizeof, static,
)
from pythoc.libc.string import memset

from .executor_effect import (
    ExecutorHandle,
    executor_handle_new,
    executor_handle_consume,
)
from .platform import (
    ThreadHandle,
    Mutex,
    CondVar,
    thread_create,
    thread_join,
    mutex_init,
    mutex_lock,
    mutex_unlock,
    mutex_destroy,
    condvar_init,
    condvar_wait,
    condvar_signal,
    condvar_broadcast,
    condvar_destroy,
)


DEFAULT_THREAD_POOL_WORKERS = u64(4)


@compile
class _ThreadJob:
    entry: func[ptr[void], ptr[void]]
    arg: ptr[void]
    result: ptr[void]
    done: i32
    lock: Mutex
    done_cv: CondVar
    next: ptr[_ThreadJob]


@compile
class _ThreadPool:
    workers: ptr[ThreadHandle]
    worker_count: u64
    head: ptr[_ThreadJob]
    tail: ptr[_ThreadJob]
    lock: Mutex
    ready_cv: CondVar
    shutdown: i32
    initialized: i32


@compile
def _thread_pool_state() -> ptr[_ThreadPool]:
    pool: static[ptr[_ThreadPool]] = nullptr
    if pool == nullptr:
        pool = ptr[_ThreadPool](effect.mem.malloc(u64(sizeof(_ThreadPool))))
        memset(ptr[void](pool), 0, i64(sizeof(_ThreadPool)))
    return pool


@compile
def _thread_pool_pop(pool: ptr[_ThreadPool]) -> ptr[_ThreadJob]:
    job: ptr[_ThreadJob] = pool.head
    if job != nullptr:
        pool.head = job.next
        if pool.head == nullptr:
            pool.tail = nullptr
        job.next = nullptr
    return job


@compile
def _thread_pool_worker(raw: ptr[void]) -> ptr[void]:
    pool: ptr[_ThreadPool] = ptr[_ThreadPool](raw)
    while i32(1):
        mutex_lock(ptr[Mutex](ptr[void](ptr(pool.lock))))
        while pool.head == nullptr and pool.shutdown == i32(0):
            condvar_wait(
                ptr[CondVar](ptr[void](ptr(pool.ready_cv))),
                ptr[Mutex](ptr[void](ptr(pool.lock))),
            )
        if pool.shutdown != i32(0) and pool.head == nullptr:
            mutex_unlock(ptr[Mutex](ptr[void](ptr(pool.lock))))
            return nullptr
        job: ptr[_ThreadJob] = _thread_pool_pop(pool)
        mutex_unlock(ptr[Mutex](ptr[void](ptr(pool.lock))))

        job.result = job.entry(job.arg)

        mutex_lock(ptr[Mutex](ptr[void](ptr(job.lock))))
        job.done = i32(1)
        condvar_signal(ptr[CondVar](ptr[void](ptr(job.done_cv))))
        mutex_unlock(ptr[Mutex](ptr[void](ptr(job.lock))))
    return nullptr


@compile
def thread_pool_start(worker_count: u64) -> void:
    pool: ptr[_ThreadPool] = _thread_pool_state()
    if pool.initialized != i32(0):
        return

    mutex_init(ptr[Mutex](ptr[void](ptr(pool.lock))))
    condvar_init(ptr[CondVar](ptr[void](ptr(pool.ready_cv))))
    pool.head = nullptr
    pool.tail = nullptr
    pool.shutdown = i32(0)
    pool.worker_count = worker_count
    pool.workers = ptr[ThreadHandle](
        effect.mem.malloc(u64(sizeof(ThreadHandle)) * worker_count)
    )

    i: u64 = u64(0)
    while i < worker_count:
        pool.workers[i] = thread_create(
            ptr[void](_thread_pool_worker),
            ptr[void](pool),
        )
        i = i + u64(1)
    pool.initialized = i32(1)


@compile
def thread_pool_shutdown() -> void:
    pool: ptr[_ThreadPool] = _thread_pool_state()
    if pool.initialized == i32(0):
        return

    mutex_lock(ptr[Mutex](ptr[void](ptr(pool.lock))))
    pool.shutdown = i32(1)
    condvar_broadcast(ptr[CondVar](ptr[void](ptr(pool.ready_cv))))
    mutex_unlock(ptr[Mutex](ptr[void](ptr(pool.lock))))

    i: u64 = u64(0)
    while i < pool.worker_count:
        thread_join(pool.workers[i])
        i = i + u64(1)

    effect.mem.free(ptr[void](pool.workers))
    condvar_destroy(ptr[CondVar](ptr[void](ptr(pool.ready_cv))))
    mutex_destroy(ptr[Mutex](ptr[void](ptr(pool.lock))))
    memset(ptr[void](pool), 0, i64(sizeof(_ThreadPool)))


@compile
def _thread_job_free(job: ptr[_ThreadJob]) -> void:
    condvar_destroy(ptr[CondVar](ptr[void](ptr(job.done_cv))))
    mutex_destroy(ptr[Mutex](ptr[void](ptr(job.lock))))
    effect.mem.free(ptr[void](job))


@compile
def _thread_job_wait(job: ptr[_ThreadJob]) -> ptr[void]:
    mutex_lock(ptr[Mutex](ptr[void](ptr(job.lock))))
    while job.done == i32(0):
        condvar_wait(
            ptr[CondVar](ptr[void](ptr(job.done_cv))),
            ptr[Mutex](ptr[void](ptr(job.lock))),
        )
    result: ptr[void] = job.result
    mutex_unlock(ptr[Mutex](ptr[void](ptr(job.lock))))
    return result


@compile
def _thread_spawn(
    entry: func[ptr[void], ptr[void]],
    arg: ptr[void],
    stack_size: u64,
) -> ExecutorHandle:
    thread_pool_start(DEFAULT_THREAD_POOL_WORKERS)
    pool: ptr[_ThreadPool] = _thread_pool_state()
    job: ptr[_ThreadJob] = ptr[_ThreadJob](
        effect.mem.malloc(u64(sizeof(_ThreadJob)))
    )
    memset(ptr[void](job), 0, i64(sizeof(_ThreadJob)))
    job.entry = entry
    job.arg = arg
    job.result = nullptr
    job.done = i32(0)
    job.next = nullptr
    mutex_init(ptr[Mutex](ptr[void](ptr(job.lock))))
    condvar_init(ptr[CondVar](ptr[void](ptr(job.done_cv))))

    mutex_lock(ptr[Mutex](ptr[void](ptr(pool.lock))))
    if pool.tail == nullptr:
        pool.head = job
        pool.tail = job
    else:
        pool.tail.next = job
        pool.tail = job
    condvar_signal(ptr[CondVar](ptr[void](ptr(pool.ready_cv))))
    mutex_unlock(ptr[Mutex](ptr[void](ptr(pool.lock))))
    return executor_handle_new(ptr[void](job))


@compile
def _thread_join(handle: ExecutorHandle) -> ptr[void]:
    job: ptr[_ThreadJob] = ptr[_ThreadJob](
        executor_handle_consume(handle)
    )
    result: ptr[void] = _thread_job_wait(job)
    _thread_job_free(job)
    return result


@compile
def _thread_detach(handle: ExecutorHandle) -> void:
    job: ptr[_ThreadJob] = ptr[_ThreadJob](
        executor_handle_consume(handle)
    )
    _thread_job_wait(job)
    _thread_job_free(job)


@compile
def _thread_yield() -> void:
    pass


ThreadPoolExecutor = SimpleNamespace(
    spawn=_thread_spawn,
    join=_thread_join,
    yield_now=_thread_yield,
    detach=_thread_detach,
    start=thread_pool_start,
    shutdown=thread_pool_shutdown,
)


__all__ = ["ThreadPoolExecutor", "thread_pool_start", "thread_pool_shutdown"]
