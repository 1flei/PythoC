"""Build a proof type carrying a fixed acquire/release pair."""

from pythoc import compile, linear, struct, consume, refined, assume, typeof, meta


def linearize(acquire_func, release_func, name=None):
    """Return a proof type with acquire and release methods.

    acquire returns struct[resource, ProofType]. release accepts the original
    release arguments followed by that proof. The proof records the selected
    callable pair; a display name alone never establishes protocol identity.
    """
    acquire_type, release_type = typeof(acquire_func), typeof(release_func)
    if (getattr(acquire_type, 'param_types', None) is None
            or getattr(release_type, 'param_types', None) is None):
        raise TypeError('linearize requires concrete callable signatures')
    acquire_param_types = list(acquire_type.param_types)
    release_param_types = list(release_type.param_types)
    acquire_return_type = acquire_type.return_type
    identity = meta.normalize_factory_key(acquire_func, release_func, name=name)
    proof_tag = 'linearize:' + identity
    ProofType = refined[linear, proof_tag]
    ProofType.__name__ = name or 'ResourceProof'
    suffix = ('linearize', identity)

    @compile(suffix=(suffix, 'make_proof'))
    def make_proof() -> ProofType:
        return assume(linear(), proof_tag)

    ReturnStruct = struct[acquire_return_type, ProofType]
    AcquireParams = struct[tuple(acquire_param_types)]

    @compile(suffix=(suffix, 'acquire'))
    def acquire(*args: AcquireParams) -> ReturnStruct:
        result: ReturnStruct
        result[0] = acquire_func(*args)
        result[1] = make_proof()
        return result

    ReleaseParams = struct[tuple(release_param_types)]
    ReleaseWithProof = struct[tuple(release_param_types + [ProofType])]
    count = len(release_param_types)

    @compile(suffix=(suffix, 'release'))
    def release(*args: ReleaseWithProof):
        params: ReleaseParams
        for index in range(count):
            params[index] = args[index]
        proof: ProofType = args[count]
        release_func(*params)
        consume(proof)

    ProofType.acquire = acquire
    ProofType.release = release
    return ProofType
