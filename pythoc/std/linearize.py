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
        return (acquire_func(*args), make_proof())

    ReleaseWithProof = struct[tuple(release_param_types + [ProofType])]
    params = [(f'arg{i}', ty) for i, ty in enumerate(release_param_types)]

    @meta.quote
    def release_body(proof, /, *values):
        release_func(*values)
        consume(proof)

    release_resource = meta.inline_generated(meta.func(
        'release_resource', [*params, ('proof', ProofType)], None,
        release_body('proof', *(name for name, _ in params))))

    @compile(suffix=(suffix, 'release'))
    def release(*args: ReleaseWithProof):
        release_resource(*args)

    ProofType.acquire = acquire
    ProofType.release = release
    return ProofType
