"""Native POSIX calls on disposable files, mappings and thread state."""

import errno
import os
from pathlib import Path
import sys
import tempfile
import unittest

from pythoc import array, compile, i8, i32, nullptr, ptr, u64, void


IS_POSIX = sys.platform == 'darwin' or sys.platform.startswith('linux')

if IS_POSIX:
    from pythoc.libc.errno import errno_slot
    from pythoc.libc.pthread import pthread_create, pthread_join, pthread_t
    from pythoc.libc.sys_stat import fchmod, timespec
    from pythoc.libc.time import nanosleep
    from pythoc.libc.unistd import (
        fchown, ftruncate, fsync, pread, pwrite, rmdir,
    )

    @compile
    def file_operations(fd: i32) -> i32:
        source: array[i8, 3] = (120, 121, 122)
        target: array[i8, 3] = (0, 0, 0)
        wrote = pwrite(fd, source, 3, 1)
        read = pread(fd, target, 3, 1)
        truncated = ftruncate(fd, 6)
        chmod = fchmod(fd, 384)
        synced = fsync(fd)
        return (i32(wrote == 3) + i32(read == 3)
                + i32(target[0] == 120) + i32(target[1] == 121)
                + i32(target[2] == 122) + i32(truncated == 0)
                + i32(chmod == 0) + i32(synced == 0))

    @compile
    def read_at_eof(fd: i32) -> i32:
        target: array[i8, 1] = (77,)
        result = pread(fd, target, 1, 100)
        return i32(result == 0) + i32(target[0] == 77)

    @compile
    def invalid_owner_change() -> i32:
        result = fchown(-1, 0, 0)
        if result != -1:
            return -1
        return errno_slot()[0]

    @compile
    def thread_start(argument: ptr[void]) -> ptr[void]:
        value = ptr[i32](argument)
        value[0] = value[0] + 1
        return argument

    @compile
    def thread_result() -> i32:
        tid: pthread_t
        local: i32 = 73
        result: ptr[void] = nullptr
        rc = pthread_create(ptr(tid), nullptr, thread_start,
                            ptr[void](ptr(local)))
        if rc != 0:
            return -1
        joined = pthread_join(tid, ptr(result))
        return (i32(joined == 0) + i32(u64(result) == u64(ptr(local)))
                + i32(local == 74))

    @compile
    def remove_directory(path: ptr[i8]) -> i32:
        return rmdir(path)

    @compile
    def sleep_zero() -> i32:
        request: timespec
        request.tv_sec = 0
        request.tv_nsec = 0
        return nanosleep(ptr(request), nullptr)

    @compile
    def sleep_invalid() -> i32:
        request: timespec
        request.tv_sec = 0
        request.tv_nsec = 1000000000
        if nanosleep(ptr(request), nullptr) != -1:
            return -1
        return errno_slot()[0]

if sys.platform.startswith('linux'):
    from pythoc.libc.sys_mman import mmap, mremap, munmap

    @compile
    def remap_roundtrip(page_size: u64) -> i32:
        # Linux MAP_PRIVATE | MAP_ANONYMOUS and MREMAP_MAYMOVE.
        address = mmap(nullptr, page_size, 3, 34, -1, 0)
        if u64(address) == u64(0xffffffffffffffff):
            return -1
        ptr[i8](address)[0] = 73
        expanded = mremap(address, page_size, page_size * 2, 1)
        if u64(expanded) == u64(0xffffffffffffffff):
            munmap(address, page_size)
            return -2
        result = i32(ptr[i8](expanded)[0] == 73)
        ptr[i8](expanded)[page_size] = 41
        result = result + i32(ptr[i8](expanded)[page_size] == 41)
        return result + i32(munmap(expanded, page_size * 2) == 0)


@unittest.skipUnless(IS_POSIX, 'POSIX calls require macOS or Linux')
class TestPosixIO(unittest.TestCase):
    def test_file_contents_offsets_and_metadata(self):
        with tempfile.TemporaryDirectory(prefix='pc-posix-io-') as temp:
            filename = Path(temp) / 'data.bin'
            fd = os.open(filename, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                os.lseek(fd, 9, os.SEEK_SET)
                self.assertEqual(file_operations(fd), 8)
                self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 9)
                self.assertEqual(os.fstat(fd).st_size, 6)
                self.assertEqual(os.pread(fd, 3, 1), b'xyz')
                self.assertEqual(os.fstat(fd).st_mode & 0o777, 0o600)
                self.assertEqual(read_at_eof(fd), 2)
            finally:
                os.close(fd)

    def test_fchown_invalid_fd(self):
        self.assertEqual(invalid_owner_change(), errno.EBADF)

    def test_thread_argument_return_and_side_effect(self):
        self.assertEqual(thread_result(), 3)

    def test_remove_directory(self):
        with tempfile.TemporaryDirectory(prefix='pc-posix-dir-') as temp:
            child = Path(temp) / 'empty'
            child.mkdir()
            self.assertEqual(remove_directory(os.fsencode(child)), 0)
            self.assertFalse(child.exists())

    def test_nanosleep_success_and_invalid_request(self):
        self.assertEqual(sleep_zero(), 0)
        self.assertEqual(sleep_invalid(), errno.EINVAL)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Linux mremap')
    def test_mremap_preserves_data_and_grows_mapping(self):
        self.assertEqual(remap_roundtrip(os.sysconf('SC_PAGE_SIZE')), 3)


if __name__ == '__main__':
    unittest.main()
