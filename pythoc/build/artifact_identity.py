"""Verify the full specialization identity behind a shortened artifact path."""

import json
import os
import threading
import uuid

from ..type_descriptor import specialization_key


_local_lock = threading.RLock()


def verify_artifact_identity(obj_file, so_file, *, scope, suffix, effect_suffix):
    """Bind a compact filename to its full specialization key.

    A missing sidecar makes previous artifacts unverifiable; preserve them
    under unique backup names and force compilation. A mismatched sidecar is
    a real collision and must fail before any cache lookup or publication.
    """
    if scope is None and suffix is None and effect_suffix is None:
        return
    key = json.dumps(specialization_key((scope, suffix, effect_suffix)), ensure_ascii=False,
                     separators=(',', ':'))
    sidecar = obj_file + '.typekey'
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    with _local_lock:
        with open(sidecar + '.lock', 'a+b') as lock_file:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except ImportError:
                import msvcrt
                lock_file.seek(0)
                if not lock_file.read(1):
                    lock_file.write(b'\0')
                    lock_file.flush()
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            if os.path.exists(sidecar):
                with open(sidecar, encoding='utf-8') as stream:
                    stored = stream.read()
                if stored != key:
                    raise TypeError(
                        f'Artifact filename collision for {obj_file}; '
                        'the full specialization identities differ')
                return
            for artifact in (obj_file, so_file):
                if os.path.exists(artifact):
                    os.replace(artifact, artifact + '.unverified-' + uuid.uuid4().hex)
            temporary = sidecar + '.tmp-' + uuid.uuid4().hex
            try:
                with open(temporary, 'w', encoding='utf-8') as stream:
                    stream.write(key)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, sidecar)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
