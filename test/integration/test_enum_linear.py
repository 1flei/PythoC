"""
Integration tests for enum with linear payload (sum of linear type).

These tests verify whole-enum linearity:
- If any variant has a linear payload, the whole enum type is linear.
- Matching the enum consumes the scrutinee.
- The linear payload's ownership transfers to the bound variable.
- Linear variants must be explicitly matched; wildcards cannot swallow them.

Coverage emphasizes *complex payloads*: structs with multiple linear fields,
enums whose payload is itself another enum/struct, deeply nested compositions,
and real-world-style state machines (Connection/Request/Response).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import unittest

from pythoc import i32, i8, enum, compile, struct, linear, consume, void
from pythoc.builtin_entities import move
from pythoc.build.output_manager import flush_all_pending_outputs, clear_failed_group
from pythoc.logger import set_raise_on_error

set_raise_on_error(True)


# =============================================================================
# Test types
# =============================================================================

@enum(i32)
class Resource:
    Open: linear
    Closed: i32


@struct
class FileHandle:
    fd: i32
    token: linear


@enum(i32)
class ResourceWithStruct:
    Open: FileHandle
    Closed: i32


@struct
class Container:
    data: Resource


@enum(i32)
class Token:
    A: linear
    B: linear


@enum(i32)
class Result:
    Ok: i32
    Err: linear


@struct
class Buffer:
    size: i32
    token: linear


@enum(i32)
class ConnectionState:
    Open: FileHandle
    Closed: i32
    Error: linear


@enum(i32)
class Request:
    Get: linear
    Post: Buffer
    Empty: None


@enum(i32)
class Response:
    Ok: Buffer
    Err: linear
    NotFound: None


@enum(i32)
class MaybeResource:
    Some: Resource
    None_: None


@struct
class Wrapper:
    payload: Resource


@struct
class DoubleWrapper:
    inner: Wrapper


@struct
class ResourcePair:
    first: Resource
    second: Resource


@enum(i32)
class TreeNode:
    Leaf: linear
    Node: ResourcePair


@enum(i32)
class OptionalLinear:
    Some: linear
    None_: None


@enum(i32)
class MultiLinear:
    X: linear
    Y: linear
    Z: linear


@struct
class Pair:
    first: Resource
    second: Resource


# A linear file representation: the token represents the open-file obligation.
# The only way to satisfy it is to match the file out and consume the token.
@struct
class File:
    fd: i32
    token: linear


@enum(i32)
class OptionalFile:
    Some: File
    None_: None


@enum(i32)
class FileResult:
    Ok: File
    Err: i32


@enum(i32)
class ReadResult:
    Data: Buffer
    Eof: None


# =============================================================================
# Helpers
# =============================================================================

@compile
def consume_resource_enum(r: Resource) -> i32:
    match r:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def consume_result_enum(r: Result) -> i32:
    match r:
        case (Result.Ok, val):
            return val
        case (Result.Err, token):
            consume(token)
            return -1
    return -2


@compile
def consume_connection_state(c: ConnectionState) -> i32:
    match c:
        case (ConnectionState.Open, fh):
            consume(fh.token)
            return fh.fd
        case (ConnectionState.Closed, code):
            return code
        case (ConnectionState.Error, token):
            consume(token)
            return -1
    return -2


@compile
def consume_response(r: Response) -> i32:
    match r:
        case (Response.Ok, buf):
            consume(buf.token)
            return buf.size
        case (Response.Err, token):
            consume(token)
            return -1
        case (Response.NotFound):
            return 0
    return -2


@compile
def make_open_resource() -> Resource:
    return Resource(Resource.Open, linear())


@compile
def make_err_result() -> Result:
    return Result(Result.Err, linear())


@compile
def make_open_connection(fd: i32) -> ConnectionState:
    fh: FileHandle
    fh.fd = fd
    fh.token = linear()
    return ConnectionState(ConnectionState.Open, fh)


@compile
def make_ok_response(size: i32) -> Response:
    buf: Buffer
    buf.size = size
    buf.token = linear()
    return Response(Response.Ok, buf)


@compile
def open_file(fd: i32) -> OptionalFile:
    if fd >= 0:
        f: File
        f.fd = fd
        f.token = linear()
        return OptionalFile(OptionalFile.Some, f)
    else:
        return OptionalFile(OptionalFile.None_)


@compile
def close_file(of: OptionalFile) -> i32:
    match of:
        case (OptionalFile.Some, f):
            consume(f.token)
            return f.fd
        case (OptionalFile.None_):
            return -1
    return -2


@compile
def open_file_result(fd: i32) -> FileResult:
    if fd >= 0:
        f: File
        f.fd = fd
        f.token = linear()
        return FileResult(FileResult.Ok, f)
    else:
        return FileResult(FileResult.Err, -1)


@compile
def close_file_result(r: FileResult) -> i32:
    match r:
        case (FileResult.Ok, f):
            consume(f.token)
            return f.fd
        case (FileResult.Err, code):
            return code
    return -2


@compile
def read_chunk(size: i32) -> ReadResult:
    if size > 0:
        buf: Buffer
        buf.size = size
        buf.token = linear()
        return ReadResult(ReadResult.Data, buf)
    else:
        return ReadResult(ReadResult.Eof)


@compile
def consume_read_result(r: ReadResult) -> i32:
    match r:
        case (ReadResult.Data, b):
            consume(b.token)
            return b.size
        case (ReadResult.Eof):
            return 0
    return -1


# =============================================================================
# Valid cases - basic enum linearity
# =============================================================================

@compile
def test_basic_enum_linear() -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_merge_different_variants(cond: i32) -> i32:
    r: Resource
    if cond:
        r = Resource(Resource.Open, linear())
    else:
        r = Resource(Resource.Closed, 42)

    match r:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_move_payload() -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, _):
            return 0


@compile
def test_struct_enum_field() -> i32:
    c: Container
    c.data = Resource(Resource.Open, linear())
    match c.data:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_nested_linear_struct_in_enum() -> i32:
    fh: FileHandle
    fh.fd = 3
    fh.token = linear()
    r = ResourceWithStruct(ResourceWithStruct.Open, fh)
    match r:
        case (ResourceWithStruct.Open, fh2):
            consume(fh2.token)
            return fh2.fd
        case (ResourceWithStruct.Closed, val):
            return val
    return -1


@compile
def test_two_linear_variants(tag: i32) -> i32:
    t = Token(Token.A, linear())
    match t:
        case (Token.A, a):
            consume(a)
            return 1
        case (Token.B, b):
            consume(b)
            return 2
    return -1


@compile
def test_non_linear_variant_must_be_consumed() -> i32:
    r = Resource(Resource.Closed, 42)
    match r:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_enum_return_transfers_ownership() -> Resource:
    return Resource(Resource.Open, linear())


@compile
def test_enum_move_transfers_ownership() -> i32:
    r1 = Resource(Resource.Open, linear())
    r2 = move(r1)
    match r2:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_match_with_guard() -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token) if True:
            consume(token)
            return 1
        case (Resource.Open, token):
            # Satisfy exhaustiveness checker which treats guards as potentially False.
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


# =============================================================================
# Valid cases - complex struct payloads
# =============================================================================

@compile
def test_filehandle_payload_consume_token() -> i32:
    fh: FileHandle
    fh.fd = 10
    fh.token = linear()
    c = ConnectionState(ConnectionState.Open, fh)
    match c:
        case (ConnectionState.Open, h):
            consume(h.token)
            return h.fd
        case (ConnectionState.Closed, code):
            return code
        case (ConnectionState.Error, token):
            consume(token)
            return -1
    return -2


@compile
def test_buffer_payload_consume_token() -> i32:
    buf: Buffer
    buf.size = 20
    buf.token = linear()
    req = Request(Request.Post, buf)
    match req:
        case (Request.Get, token):
            consume(token)
            return 1
        case (Request.Post, b):
            consume(b.token)
            return b.size
        case (Request.Empty):
            return 0
    return -1


@compile
def test_connection_state_all_branches(cond: i32) -> i32:
    c: ConnectionState
    if cond == 0:
        fh: FileHandle
        fh.fd = 5
        fh.token = linear()
        c = ConnectionState(ConnectionState.Open, fh)
    elif cond == 1:
        c = ConnectionState(ConnectionState.Closed, 99)
    else:
        c = ConnectionState(ConnectionState.Error, linear())
    return consume_connection_state(c)


@compile
def test_request_get_token() -> i32:
    req = Request(Request.Get, linear())
    match req:
        case (Request.Get, token):
            consume(token)
            return 1
        case (Request.Post, b):
            consume(b.token)
            return b.size
        case (Request.Empty):
            return 0
    return -1


@compile
def test_request_empty() -> i32:
    req = Request(Request.Empty)
    match req:
        case (Request.Get, token):
            consume(token)
            return 1
        case (Request.Post, b):
            consume(b.token)
            return b.size
        case (Request.Empty):
            return 0
    return -1


@compile
def test_response_ok_return_buffer() -> i32:
    r = make_ok_response(33)
    return consume_response(r)


@compile
def test_response_err_consume_token() -> i32:
    r = Response(Response.Err, linear())
    match r:
        case (Response.Ok, b):
            consume(b.token)
            return b.size
        case (Response.Err, token):
            consume(token)
            return -1
        case (Response.NotFound):
            return 0
    return -2


@compile
def test_response_not_found() -> i32:
    r = Response(Response.NotFound)
    match r:
        case (Response.Ok, b):
            consume(b.token)
            return b.size
        case (Response.Err, token):
            consume(token)
            return -1
        case (Response.NotFound):
            return 0
    return -2


# =============================================================================
# Valid cases - enum payload is another enum / nested struct
# =============================================================================

@compile
def test_maybe_resource_some_open() -> i32:
    inner = Resource(Resource.Open, linear())
    m = MaybeResource(MaybeResource.Some, inner)
    match m:
        case (MaybeResource.Some, r):
            return consume_resource_enum(r)
        case (MaybeResource.None_):
            return 0
    return -1


@compile
def test_maybe_resource_some_closed() -> i32:
    inner = Resource(Resource.Closed, 77)
    m = MaybeResource(MaybeResource.Some, inner)
    match m:
        case (MaybeResource.Some, r):
            return consume_resource_enum(r)
        case (MaybeResource.None_):
            return 0
    return -1


@compile
def test_wrapper_struct_contains_enum() -> i32:
    w: Wrapper
    w.payload = Resource(Resource.Open, linear())
    match w.payload:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_double_wrapper_nested_struct_enum() -> i32:
    dw: DoubleWrapper
    dw.inner.payload = Resource(Resource.Closed, 55)
    match dw.inner.payload:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_resource_pair_node_consume_both() -> i32:
    rp: ResourcePair
    rp.first = Resource(Resource.Open, linear())
    rp.second = Resource(Resource.Closed, 8)
    n = TreeNode(TreeNode.Node, rp)
    match n:
        case (TreeNode.Leaf, token):
            consume(token)
            return 1
        case (TreeNode.Node, pair):
            match pair.first:
                case (Resource.Open, t1):
                    consume(t1)
                case (Resource.Closed, _):
                    pass
            match pair.second:
                case (Resource.Open, t2):
                    consume(t2)
                case (Resource.Closed, _):
                    pass
            return 10
    return -1


@compile
def test_tree_node_leaf_consume() -> i32:
    n = TreeNode(TreeNode.Leaf, linear())
    match n:
        case (TreeNode.Leaf, token):
            consume(token)
            return 1
        case (TreeNode.Node, pair):
            match pair.first:
                case (Resource.Open, t1):
                    consume(t1)
                case (Resource.Closed, _):
                    pass
            match pair.second:
                case (Resource.Open, t2):
                    consume(t2)
                case (Resource.Closed, _):
                    pass
            return 2
    return -1


@compile
def test_tree_node_node_consume_second() -> i32:
    rp: ResourcePair
    rp.first = Resource(Resource.Closed, 3)
    rp.second = Resource(Resource.Open, linear())
    n = TreeNode(TreeNode.Node, rp)
    match n:
        case (TreeNode.Leaf, token):
            consume(token)
            return 1
        case (TreeNode.Node, pair):
            match pair.first:
                case (Resource.Open, t1):
                    consume(t1)
                case (Resource.Closed, _):
                    pass
            match pair.second:
                case (Resource.Open, t2):
                    consume(t2)
                    return 20
                case (Resource.Closed, v2):
                    return v2
    return -1


# =============================================================================
# Valid cases - move/return complex payloads
# =============================================================================

@compile
def test_return_connection_state() -> ConnectionState:
    fh: FileHandle
    fh.fd = 42
    fh.token = linear()
    return ConnectionState(ConnectionState.Open, fh)


@compile
def test_move_connection_state() -> i32:
    c = make_open_connection(7)
    return consume_connection_state(move(c))


@compile
def test_move_response() -> i32:
    r = make_ok_response(99)
    return consume_response(move(r))


@compile
def test_return_tree_node() -> TreeNode:
    rp: ResourcePair
    rp.first = Resource(Resource.Open, linear())
    rp.second = Resource(Resource.Closed, 0)
    return TreeNode(TreeNode.Node, rp)


@compile
def test_consume_returned_tree_node() -> i32:
    n = test_return_tree_node()
    match n:
        case (TreeNode.Leaf, token):
            consume(token)
            return 1
        case (TreeNode.Node, pair):
            match pair.first:
                case (Resource.Open, t1):
                    consume(t1)
                case (Resource.Closed, _):
                    pass
            match pair.second:
                case (Resource.Open, t2):
                    consume(t2)
                case (Resource.Closed, v2):
                    return v2
    return -1


# =============================================================================
# Valid cases - practical optional/result/file resources
# =============================================================================

@compile
def test_optional_file_some() -> i32:
    f = open_file(42)
    return close_file(f)


@compile
def test_optional_file_none() -> i32:
    f = open_file(-1)
    return close_file(f)


@compile
def test_file_result_ok() -> i32:
    r = open_file_result(7)
    return close_file_result(r)


@compile
def test_file_result_err() -> i32:
    r = open_file_result(-1)
    return close_file_result(r)


@compile
def test_read_result_data() -> i32:
    r = read_chunk(100)
    return consume_read_result(r)


@compile
def test_read_result_eof() -> i32:
    r = read_chunk(0)
    return consume_read_result(r)


@compile
def test_optional_file_match_only() -> i32:
    # Spelled inline to emphasize: an optional linear file can only be consumed
    # by matching the Some variant and closing (consuming) the file.
    f = open_file(3)
    match f:
        case (OptionalFile.Some, file):
            consume(file.token)
            return file.fd
        case (OptionalFile.None_):
            return -1
    return -2


# =============================================================================
# Valid cases - helpers and control flow with complex payloads
# =============================================================================

@compile
def test_helper_consumes_filehandle_enum() -> i32:
    fh: FileHandle
    fh.fd = 11
    fh.token = linear()
    c = ConnectionState(ConnectionState.Open, fh)
    return consume_connection_state(move(c))


@compile
def test_helper_returns_filehandle_enum() -> i32:
    c = make_open_connection(22)
    match c:
        case (ConnectionState.Open, h):
            consume(h.token)
            return h.fd
        case (ConnectionState.Closed, code):
            return code
        case (ConnectionState.Error, token):
            consume(token)
            return -1
    return -2


@compile
def test_if_inside_match_case_with_complex(cond: i32) -> i32:
    fh: FileHandle
    fh.fd = 10
    fh.token = linear()
    c = ConnectionState(ConnectionState.Open, fh)
    match c:
        case (ConnectionState.Open, h):
            if cond:
                consume(h.token)
                return h.fd
            else:
                consume(h.token)
                return -h.fd
        case (ConnectionState.Closed, code):
            return code
        case (ConnectionState.Error, token):
            consume(token)
            return -1
    return -2


@compile
def test_nested_match_complex_payload() -> i32:
    inner = Resource(Resource.Open, linear())
    m = MaybeResource(MaybeResource.Some, inner)
    match m:
        case (MaybeResource.Some, r):
            match r:
                case (Resource.Open, token):
                    consume(token)
                    return 1
                case (Resource.Closed, val):
                    return val
        case (MaybeResource.None_):
            return 0
    return -1


@compile
def test_reassign_complex_enum_after_consume() -> i32:
    c = ConnectionState(ConnectionState.Error, linear())
    match c:
        case (ConnectionState.Open, h):
            consume(h.token)
        case (ConnectionState.Closed, code):
            pass
        case (ConnectionState.Error, token):
            consume(token)

    fh2: FileHandle
    fh2.fd = 0
    fh2.token = linear()
    c = ConnectionState(ConnectionState.Open, fh2)
    match c:
        case (ConnectionState.Open, h2):
            consume(h2.token)
            return h2.fd
        case (ConnectionState.Closed, code):
            return code
        case (ConnectionState.Error, token):
            consume(token)
            return -1
    return -2


@compile
def test_loop_complex_enum(n: i32) -> i32:
    total: i32 = 0
    i: i32 = 0
    while i < n:
        fh: FileHandle
        fh.fd = i
        fh.token = linear()
        c = ConnectionState(ConnectionState.Open, fh)
        match c:
            case (ConnectionState.Open, h):
                consume(h.token)
                total = total + h.fd
            case (ConnectionState.Closed, code):
                total = total + code
            case (ConnectionState.Error, token):
                consume(token)
                total = total - 1
        i = i + 1
    return total


@compile
def test_complex_enum_in_if_branches(cond: i32) -> i32:
    r: Response
    if cond == 0:
        buf: Buffer
        buf.size = 1
        buf.token = linear()
        r = Response(Response.Ok, buf)
    elif cond == 1:
        r = Response(Response.Err, linear())
    else:
        r = Response(Response.NotFound)
    return consume_response(r)


# =============================================================================
# Valid cases - previously added broad coverage
# =============================================================================

@compile
def test_match_result_variant(cond: i32) -> i32:
    r: Result
    if cond:
        r = Result(Result.Ok, 7)
    else:
        r = Result(Result.Err, linear())
    return consume_result_enum(r)


@compile
def test_optional_linear_some() -> i32:
    opt = OptionalLinear(OptionalLinear.Some, linear())
    match opt:
        case (OptionalLinear.Some, token):
            consume(token)
            return 1
        case (OptionalLinear.None_):
            return 0
    return -1


@compile
def test_optional_linear_none() -> i32:
    opt = OptionalLinear(OptionalLinear.None_)
    match opt:
        case (OptionalLinear.Some, token):
            consume(token)
            return 1
        case (OptionalLinear.None_):
            return 0
    return -1


@compile
def test_all_linear_variants_exhaustive() -> i32:
    m = MultiLinear(MultiLinear.X, linear())
    match m:
        case (MultiLinear.X, x):
            consume(x)
            return 1
        case (MultiLinear.Y, y):
            consume(y)
            return 2
        case (MultiLinear.Z, z):
            consume(z)
            return 3
    return -1


@compile
def test_multiple_linear_enums() -> i32:
    r = Resource(Resource.Open, linear())
    t = Token(Token.A, linear())
    match r:
        case (Resource.Open, rt):
            consume(rt)
            match t:
                case (Token.A, ta):
                    consume(ta)
                    return 1
                case (Token.B, tb):
                    consume(tb)
                    return 2
        case (Resource.Closed, val):
            match t:
                case (Token.A, ta):
                    consume(ta)
                    return val + 10
                case (Token.B, tb):
                    consume(tb)
                    return val + 20
    return -1


@compile
def test_enum_in_if_branches(cond: i32) -> i32:
    r: Resource
    if cond == 0:
        r = Resource(Resource.Open, linear())
    elif cond == 1:
        r = Resource(Resource.Closed, 10)
    else:
        r = Resource(Resource.Open, linear())
    return consume_resource_enum(r)


@compile
def test_if_inside_match_case(cond: i32) -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token):
            if cond:
                consume(token)
                return 1
            else:
                consume(token)
                return 2
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_nested_match_enum() -> i32:
    outer = Resource(Resource.Open, linear())
    match outer:
        case (Resource.Open, token):
            inner = Resource(Resource.Open, token)
            match inner:
                case (Resource.Open, t2):
                    consume(t2)
                    return 1
                case (Resource.Closed, val):
                    return val
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_loop_create_consume_each_iteration(n: i32) -> i32:
    total: i32 = 0
    i: i32 = 0
    while i < n:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token):
                consume(token)
                total = total + 1
            case (Resource.Closed, val):
                total = total + val
        i = i + 1
    return total


@compile
def test_while_true_break_after_consume() -> i32:
    r = Resource(Resource.Open, linear())
    while True:
        match r:
            case (Resource.Open, token):
                consume(token)
                return 1
            case (Resource.Closed, val):
                return val


@compile
def test_reassign_enum_after_consume() -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token):
            consume(token)
        case (Resource.Closed, val):
            pass
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token2):
            consume(token2)
            return 1
        case (Resource.Closed, val2):
            return val2
    return -1


@compile
def test_reassign_in_branches(cond: i32) -> i32:
    r = Resource(Resource.Open, linear())
    if cond:
        match r:
            case (Resource.Open, token):
                consume(token)
            case (Resource.Closed, val):
                pass
        r = Resource(Resource.Closed, 5)
    else:
        match r:
            case (Resource.Open, token):
                consume(token)
            case (Resource.Closed, val):
                pass
        r = Resource(Resource.Open, linear())
    return consume_resource_enum(r)


@compile
def test_move_enum_to_helper() -> i32:
    r = Resource(Resource.Open, linear())
    return consume_resource_enum(move(r))


@compile
def test_helper_returns_enum() -> i32:
    r = make_open_resource()
    return consume_resource_enum(r)


@compile
def test_return_enum_from_branches(cond: i32) -> Resource:
    if cond:
        return Resource(Resource.Open, linear())
    else:
        return Resource(Resource.Closed, 99)


@compile
def test_early_return_in_match_case(cond: i32) -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token):
            if cond:
                consume(token)
                return 1
            consume(token)
            return 2
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_enum_in_struct_pair() -> i32:
    p: Pair
    p.first = Resource(Resource.Open, linear())
    p.second = Resource(Resource.Closed, 7)
    match p.first:
        case (Resource.Open, token):
            consume(token)
            match p.second:
                case (Resource.Closed, val):
                    return val
                case (Resource.Open, t2):
                    consume(t2)
                    return -1
        case (Resource.Closed, val):
            match p.second:
                case (Resource.Closed, val2):
                    return val2
                case (Resource.Open, t2):
                    consume(t2)
                    return -1
    return -2


@compile
def test_match_then_ignore_nonlinear_payload() -> i32:
    r = Resource(Resource.Closed, 42)
    match r:
        case (Resource.Open, token):
            consume(token)
            return 1
        case (Resource.Closed, _):
            return 0


@compile
def test_different_linear_variants_returned(tag: i32) -> Token:
    if tag == 0:
        return Token(Token.A, linear())
    else:
        return Token(Token.B, linear())


@compile
def test_match_enum_returned_by_helper() -> i32:
    r = make_err_result()
    return consume_result_enum(r)


@compile
def test_multiple_consumes_in_same_case() -> i32:
    r = Resource(Resource.Open, linear())
    match r:
        case (Resource.Open, token):
            consume(token)
            return 42
        case (Resource.Closed, val):
            return val
    return -1


@compile
def test_nested_struct_enum_reassignment() -> i32:
    fh: FileHandle
    fh.fd = 5
    fh.token = linear()
    r = ResourceWithStruct(ResourceWithStruct.Open, fh)
    match r:
        case (ResourceWithStruct.Open, h):
            consume(h.token)
        case (ResourceWithStruct.Closed, val):
            pass
    r = ResourceWithStruct(ResourceWithStruct.Closed, 8)
    match r:
        case (ResourceWithStruct.Open, h2):
            consume(h2.token)
            return 1
        case (ResourceWithStruct.Closed, val):
            return val
    return -1


@compile
def test_linear_enum_in_loop_with_break(n: i32) -> i32:
    i: i32 = 0
    while i < n:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token):
                consume(token)
                if i == 2:
                    break
            case (Resource.Closed, val):
                if i == 2:
                    break
        i = i + 1
    return i


# =============================================================================
# Error cases
# =============================================================================

def _compile_bad_function(suffix, func):
    source_file = os.path.abspath(__file__)
    group_key = (source_file, 'module', suffix)
    try:
        wrapped = compile(suffix=suffix)(func)
        flush_all_pending_outputs()
        return False, "should have raised error"
    except (RuntimeError, ValueError, SystemExit, TypeError) as e:
        return True, str(e)
    finally:
        clear_failed_group(group_key)


def test_error_payload_not_consumed():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token):
                pass  # token not consumed
            case (Resource.Closed, val):
                return val
        return -1

    return _compile_bad_function('bad_payload_not_consumed', bad)


def test_error_wildcard_swallows_linear():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Closed, val):
                return val
            case _:
                return -1  # wildcard swallows Open

    return _compile_bad_function('bad_wildcard_swallows_linear', bad)


def test_error_only_wildcard_on_linear_enum():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case _:
                return -1

    return _compile_bad_function('bad_only_wildcard', bad)


def test_error_non_linear_variant_not_consumed():
    def bad() -> i32:
        r = Resource(Resource.Closed, 42)
        # Closed is non-linear payload, but the enum type is linear.
        # Must still consume it via match/return/move.
        return r[1]

    return _compile_bad_function('bad_non_linear_not_consumed', bad)


def test_error_consume_payload_twice():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token):
                consume(token)
                consume(token)  # double consume
                return 1
            case (Resource.Closed, val):
                return val
        return -1

    return _compile_bad_function('bad_consume_payload_twice', bad)


def test_error_use_payload_after_consume():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token):
                consume(token)
                consume(token)  # use after consume
                return 1
            case (Resource.Closed, val):
                return val
        return -1

    return _compile_bad_function('bad_use_payload_after_consume', bad)


def test_error_one_of_two_linear_variants_not_consumed():
    def bad() -> i32:
        t = Token(Token.A, linear())
        match t:
            case (Token.A, a):
                consume(a)
                return 1
            case (Token.B, b):
                # b not consumed
                return 2
        return -1

    return _compile_bad_function('bad_one_linear_variant_not_consumed', bad)


def test_error_nested_match_payload_not_consumed():
    def bad() -> i32:
        outer = Resource(Resource.Open, linear())
        match outer:
            case (Resource.Open, token):
                inner = Resource(Resource.Open, token)
                match inner:
                    case (Resource.Open, t2):
                        pass  # t2 not consumed
                        return 1
                    case (Resource.Closed, val):
                        return val
            case (Resource.Closed, val):
                return val
        return -1

    return _compile_bad_function('bad_nested_match_payload_not_consumed', bad)


def test_error_enum_variable_never_consumed():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        # r is never matched, returned, or moved
        return 0

    return _compile_bad_function('bad_enum_variable_never_consumed', bad)


def test_error_partial_wildcard_on_linear_enum():
    def bad() -> i32:
        m = MultiLinear(MultiLinear.X, linear())
        match m:
            case (MultiLinear.X, x):
                consume(x)
                return 1
            case _:
                # wildcard swallows Y and Z, both linear
                return -1

    return _compile_bad_function('bad_partial_wildcard_linear', bad)


def test_error_move_enum_then_use_original():
    def bad() -> i32:
        r1 = Resource(Resource.Open, linear())
        r2 = move(r1)
        consume_resource_enum(r1)  # use after move
        return consume_resource_enum(r2)

    return _compile_bad_function('bad_move_then_use_original', bad)


def test_error_payload_consumed_only_when_guard_true():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token) if False:
                consume(token)
                return 1
            case (Resource.Closed, val):
                return val
        return -1

    return _compile_bad_function('bad_payload_guard_only', bad)


def test_error_payload_not_consumed_in_one_branch():
    def bad() -> i32:
        r = Resource(Resource.Open, linear())
        match r:
            case (Resource.Open, token):
                if False:
                    consume(token)
                    return 1
                else:
                    pass  # token not consumed in else
            case (Resource.Closed, val):
                return val
        return -1

    return _compile_bad_function('bad_payload_branch_not_consumed', bad)


def test_error_enum_created_in_loop_not_consumed():
    def bad(n: i32) -> i32:
        i: i32 = 0
        while i < n:
            r = Resource(Resource.Open, linear())
            # r never consumed
            i = i + 1
        return i

    return _compile_bad_function('bad_enum_loop_not_consumed', bad)


def test_error_return_nonlinear_payload_directly():
    def bad() -> i32:
        r = Resource(Resource.Closed, 42)
        return r[1]

    return _compile_bad_function('bad_return_nonlinear_payload_directly', bad)


# --- Complex payload error cases ---

def test_error_filehandle_token_not_consumed():
    def bad() -> i32:
        fh: FileHandle
        fh.fd = 1
        fh.token = linear()
        c = ConnectionState(ConnectionState.Open, fh)
        match c:
            case (ConnectionState.Open, h):
                # h.token not consumed
                return h.fd
            case (ConnectionState.Closed, code):
                return code
            case (ConnectionState.Error, token):
                consume(token)
                return -1
        return -2

    return _compile_bad_function('bad_filehandle_token_not_consumed', bad)


def test_error_buffer_token_not_consumed():
    def bad() -> i32:
        buf: Buffer
        buf.size = 5
        buf.token = linear()
        req = Request(Request.Post, buf)
        match req:
            case (Request.Get, token):
                consume(token)
                return 1
            case (Request.Post, b):
                # b.token not consumed
                return b.size
            case (Request.Empty):
                return 0
        return -1

    return _compile_bad_function('bad_buffer_token_not_consumed', bad)


def test_error_nested_enum_payload_not_consumed():
    def bad() -> i32:
        inner = Resource(Resource.Open, linear())
        m = MaybeResource(MaybeResource.Some, inner)
        match m:
            case (MaybeResource.Some, r):
                # inner enum r not consumed
                return 0
            case (MaybeResource.None_):
                return 0
        return -1

    return _compile_bad_function('bad_nested_enum_payload_not_consumed', bad)


def test_error_wrapper_payload_not_consumed():
    def bad() -> i32:
        w: Wrapper
        w.payload = Resource(Resource.Open, linear())
        # w.payload never consumed
        return 0

    return _compile_bad_function('bad_wrapper_payload_not_consumed', bad)


def test_error_double_wrapper_not_consumed():
    def bad() -> i32:
        dw: DoubleWrapper
        dw.inner.payload = Resource(Resource.Open, linear())
        # deeply nested enum never consumed
        return 0

    return _compile_bad_function('bad_double_wrapper_not_consumed', bad)


def test_error_resource_pair_not_fully_consumed():
    def bad() -> i32:
        rp: ResourcePair
        rp.first = Resource(Resource.Open, linear())
        rp.second = Resource(Resource.Open, linear())
        n = TreeNode(TreeNode.Node, rp)
        match n:
            case (TreeNode.Leaf, token):
                consume(token)
                return 1
            case (TreeNode.Node, pair):
                match pair.first:
                    case (Resource.Open, t1):
                        consume(t1)
                    case (Resource.Closed, _):
                        pass
                # pair.second not consumed
                return 2
        return -1

    return _compile_bad_function('bad_resource_pair_not_fully_consumed', bad)


def test_error_tree_node_leaf_not_consumed():
    def bad() -> i32:
        n = TreeNode(TreeNode.Leaf, linear())
        match n:
            case (TreeNode.Leaf, token):
                # token not consumed
                return 1
            case (TreeNode.Node, pair):
                match pair.first:
                    case (Resource.Open, t1):
                        consume(t1)
                    case (Resource.Closed, _):
                        pass
                match pair.second:
                    case (Resource.Open, t2):
                        consume(t2)
                    case (Resource.Closed, _):
                        pass
                return 2
        return -1

    return _compile_bad_function('bad_tree_node_leaf_not_consumed', bad)


def test_error_tree_node_node_partial_consumed():
    def bad() -> i32:
        rp: ResourcePair
        rp.first = Resource(Resource.Open, linear())
        rp.second = Resource(Resource.Closed, 0)
        n = TreeNode(TreeNode.Node, rp)
        match n:
            case (TreeNode.Leaf, token):
                consume(token)
                return 1
            case (TreeNode.Node, pair):
                # neither pair.first nor pair.second consumed
                return 2
        return -1

    return _compile_bad_function('bad_tree_node_node_partial_consumed', bad)


def test_error_outer_consumed_inner_not():
    def bad() -> i32:
        inner = Resource(Resource.Open, linear())
        m = MaybeResource(MaybeResource.Some, inner)
        match m:
            case (MaybeResource.Some, r):
                # consume outer enum semantics are satisfied by the match,
                # but the inner linear payload of r is not consumed.
                match r:
                    case (Resource.Open, token):
                        pass  # token not consumed
                    case (Resource.Closed, val):
                        pass
                return 0
            case (MaybeResource.None_):
                return 0
        return -1

    return _compile_bad_function('bad_outer_consumed_inner_not', bad)


def test_error_move_complex_enum_then_use_original():
    def bad() -> i32:
        fh: FileHandle
        fh.fd = 1
        fh.token = linear()
        c = ConnectionState(ConnectionState.Open, fh)
        c2 = move(c)
        consume_connection_state(c)  # use after move
        return consume_connection_state(c2)

    return _compile_bad_function('bad_move_complex_then_use_original', bad)


# --- Practical optional/result/read error cases ---

def test_error_optional_file_not_closed():
    def bad() -> i32:
        f = open_file(1)
        match f:
            case (OptionalFile.Some, file):
                return file.fd  # file.token not consumed
            case (OptionalFile.None_):
                return -1
        return -2

    return _compile_bad_function('bad_optional_file_not_closed', bad)


def test_error_optional_file_never_consumed():
    def bad() -> i32:
        f = open_file(1)
        # OptionalFile is linear because Some carries a linear File;
        # not matching/returning/moving it is a leak.
        return 0

    return _compile_bad_function('bad_optional_file_never_consumed', bad)


def test_error_copy_optional_file():
    def bad() -> i32:
        f1 = open_file(1)
        f2 = f1  # linear enum cannot be copied, must use move()
        close_file(f1)
        return close_file(f2)

    return _compile_bad_function('bad_copy_optional_file', bad)


def test_error_file_result_ok_not_closed():
    def bad() -> i32:
        r = open_file_result(1)
        match r:
            case (FileResult.Ok, file):
                return file.fd  # file.token not consumed
            case (FileResult.Err, code):
                return code
        return -2

    return _compile_bad_function('bad_file_result_ok_not_closed', bad)


def test_error_read_result_data_not_consumed():
    def bad() -> i32:
        r = read_chunk(10)
        match r:
            case (ReadResult.Data, buf):
                return buf.size  # buf.token not consumed
            case (ReadResult.Eof):
                return 0
        return -1

    return _compile_bad_function('bad_read_result_data_not_consumed', bad)


# =============================================================================
# Test runner
# =============================================================================

class TestEnumLinearValid(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(test_basic_enum_linear(), 1)

    def test_merge_true(self):
        self.assertEqual(test_merge_different_variants(True), 1)

    def test_merge_false(self):
        self.assertEqual(test_merge_different_variants(False), 42)

    def test_move_payload(self):
        self.assertEqual(test_move_payload(), 1)

    def test_struct_enum_field(self):
        self.assertEqual(test_struct_enum_field(), 1)

    def test_nested_linear_struct_in_enum(self):
        self.assertEqual(test_nested_linear_struct_in_enum(), 3)

    def test_two_linear_variants(self):
        self.assertEqual(test_two_linear_variants(0), 1)

    def test_non_linear_variant_must_be_consumed(self):
        self.assertEqual(test_non_linear_variant_must_be_consumed(), 42)

    def test_return_transfers_ownership(self):
        r = test_enum_return_transfers_ownership()
        match r:
            case (Resource.Open, token):
                consume(token)

    def test_move_transfers_ownership(self):
        self.assertEqual(test_enum_move_transfers_ownership(), 1)

    def test_match_with_guard(self):
        self.assertEqual(test_match_with_guard(), 1)

    def test_filehandle_payload_consume_token(self):
        self.assertEqual(test_filehandle_payload_consume_token(), 10)

    def test_buffer_payload_consume_token(self):
        self.assertEqual(test_buffer_payload_consume_token(), 20)

    def test_connection_state_all_branches(self):
        self.assertEqual(test_connection_state_all_branches(0), 5)
        self.assertEqual(test_connection_state_all_branches(1), 99)
        self.assertEqual(test_connection_state_all_branches(2), -1)

    def test_request_get_token(self):
        self.assertEqual(test_request_get_token(), 1)

    def test_request_empty(self):
        self.assertEqual(test_request_empty(), 0)

    def test_response_ok_return_buffer(self):
        self.assertEqual(test_response_ok_return_buffer(), 33)

    def test_response_err_consume_token(self):
        self.assertEqual(test_response_err_consume_token(), -1)

    def test_response_not_found(self):
        self.assertEqual(test_response_not_found(), 0)

    def test_maybe_resource_some_open(self):
        self.assertEqual(test_maybe_resource_some_open(), 1)

    def test_maybe_resource_some_closed(self):
        self.assertEqual(test_maybe_resource_some_closed(), 77)

    def test_wrapper_struct_contains_enum(self):
        self.assertEqual(test_wrapper_struct_contains_enum(), 1)

    def test_double_wrapper_nested_struct_enum(self):
        self.assertEqual(test_double_wrapper_nested_struct_enum(), 55)

    def test_resource_pair_node_consume_both(self):
        self.assertEqual(test_resource_pair_node_consume_both(), 10)

    def test_tree_node_leaf_consume(self):
        self.assertEqual(test_tree_node_leaf_consume(), 1)

    def test_tree_node_node_consume_second(self):
        self.assertEqual(test_tree_node_node_consume_second(), 20)

    def test_return_connection_state(self):
        c = test_return_connection_state()
        self.assertEqual(consume_connection_state(c), 42)

    def test_move_connection_state(self):
        self.assertEqual(test_move_connection_state(), 7)

    def test_move_response(self):
        self.assertEqual(test_move_response(), 99)

    def test_return_tree_node(self):
        n = test_return_tree_node()
        self.assertEqual(test_consume_returned_tree_node(), 0)

    def test_consume_returned_tree_node(self):
        self.assertEqual(test_consume_returned_tree_node(), 0)

    def test_helper_consumes_filehandle_enum(self):
        self.assertEqual(test_helper_consumes_filehandle_enum(), 11)

    def test_helper_returns_filehandle_enum(self):
        self.assertEqual(test_helper_returns_filehandle_enum(), 22)

    def test_if_inside_match_case_with_complex(self):
        self.assertEqual(test_if_inside_match_case_with_complex(True), 10)
        self.assertEqual(test_if_inside_match_case_with_complex(False), -10)

    def test_nested_match_complex_payload(self):
        self.assertEqual(test_nested_match_complex_payload(), 1)

    def test_reassign_complex_enum_after_consume(self):
        self.assertEqual(test_reassign_complex_enum_after_consume(), 0)

    def test_loop_complex_enum(self):
        self.assertEqual(test_loop_complex_enum(3), 0 + 1 + 2)

    def test_complex_enum_in_if_branches(self):
        self.assertEqual(test_complex_enum_in_if_branches(0), 1)
        self.assertEqual(test_complex_enum_in_if_branches(1), -1)
        self.assertEqual(test_complex_enum_in_if_branches(2), 0)

    def test_match_result_variant(self):
        self.assertEqual(test_match_result_variant(True), 7)
        self.assertEqual(test_match_result_variant(False), -1)

    def test_optional_linear_some(self):
        self.assertEqual(test_optional_linear_some(), 1)

    def test_optional_linear_none(self):
        self.assertEqual(test_optional_linear_none(), 0)

    def test_all_linear_variants_exhaustive(self):
        self.assertEqual(test_all_linear_variants_exhaustive(), 1)

    def test_multiple_linear_enums(self):
        self.assertEqual(test_multiple_linear_enums(), 1)

    def test_enum_in_if_branches(self):
        self.assertEqual(test_enum_in_if_branches(0), 1)
        self.assertEqual(test_enum_in_if_branches(1), 10)
        self.assertEqual(test_enum_in_if_branches(2), 1)

    def test_if_inside_match_case(self):
        self.assertEqual(test_if_inside_match_case(True), 1)
        self.assertEqual(test_if_inside_match_case(False), 2)

    def test_nested_match_enum(self):
        self.assertEqual(test_nested_match_enum(), 1)

    def test_loop_create_consume_each_iteration(self):
        self.assertEqual(test_loop_create_consume_each_iteration(3), 3)

    def test_while_true_break_after_consume(self):
        self.assertEqual(test_while_true_break_after_consume(), 1)

    def test_reassign_enum_after_consume(self):
        self.assertEqual(test_reassign_enum_after_consume(), 1)

    def test_reassign_in_branches(self):
        self.assertEqual(test_reassign_in_branches(True), 5)
        self.assertEqual(test_reassign_in_branches(False), 1)

    def test_move_enum_to_helper(self):
        self.assertEqual(test_move_enum_to_helper(), 1)

    def test_helper_returns_enum(self):
        self.assertEqual(test_helper_returns_enum(), 1)

    def test_return_enum_from_branches(self):
        r = test_return_enum_from_branches(True)
        match r:
            case (Resource.Open, token):
                consume(token)
        r2 = test_return_enum_from_branches(False)
        match r2:
            case (Resource.Closed, val):
                self.assertEqual(val, 99)

    def test_early_return_in_match_case(self):
        self.assertEqual(test_early_return_in_match_case(True), 1)
        self.assertEqual(test_early_return_in_match_case(False), 2)

    def test_enum_in_struct_pair(self):
        self.assertEqual(test_enum_in_struct_pair(), 7)

    def test_match_then_ignore_nonlinear_payload(self):
        self.assertEqual(test_match_then_ignore_nonlinear_payload(), 0)

    def test_different_linear_variants_returned(self):
        t = test_different_linear_variants_returned(0)
        match t:
            case (Token.A, a):
                consume(a)
        t2 = test_different_linear_variants_returned(1)
        match t2:
            case (Token.B, b):
                consume(b)

    def test_match_enum_returned_by_helper(self):
        self.assertEqual(test_match_enum_returned_by_helper(), -1)

    def test_multiple_consumes_in_same_case(self):
        self.assertEqual(test_multiple_consumes_in_same_case(), 42)

    def test_nested_struct_enum_reassignment(self):
        self.assertEqual(test_nested_struct_enum_reassignment(), 8)

    def test_linear_enum_in_loop_with_break(self):
        self.assertEqual(test_linear_enum_in_loop_with_break(10), 2)

    def test_optional_file_some(self):
        self.assertEqual(test_optional_file_some(), 42)

    def test_optional_file_none(self):
        self.assertEqual(test_optional_file_none(), -1)

    def test_file_result_ok(self):
        self.assertEqual(test_file_result_ok(), 7)

    def test_file_result_err(self):
        self.assertEqual(test_file_result_err(), -1)

    def test_read_result_data(self):
        self.assertEqual(test_read_result_data(), 100)

    def test_read_result_eof(self):
        self.assertEqual(test_read_result_eof(), 0)

    def test_optional_file_match_only(self):
        self.assertEqual(test_optional_file_match_only(), 3)


class TestEnumLinearErrors(unittest.TestCase):
    def test_payload_not_consumed(self):
        passed, msg = test_error_payload_not_consumed()
        self.assertTrue(passed, msg)

    def test_wildcard_swallows_linear(self):
        passed, msg = test_error_wildcard_swallows_linear()
        self.assertTrue(passed, msg)

    def test_only_wildcard_on_linear_enum(self):
        passed, msg = test_error_only_wildcard_on_linear_enum()
        self.assertTrue(passed, msg)

    def test_non_linear_variant_not_consumed(self):
        passed, msg = test_error_non_linear_variant_not_consumed()
        self.assertTrue(passed, msg)

    def test_consume_payload_twice(self):
        passed, msg = test_error_consume_payload_twice()
        self.assertTrue(passed, msg)

    def test_use_payload_after_consume(self):
        passed, msg = test_error_use_payload_after_consume()
        self.assertTrue(passed, msg)

    def test_one_of_two_linear_variants_not_consumed(self):
        passed, msg = test_error_one_of_two_linear_variants_not_consumed()
        self.assertTrue(passed, msg)

    def test_nested_match_payload_not_consumed(self):
        passed, msg = test_error_nested_match_payload_not_consumed()
        self.assertTrue(passed, msg)

    def test_enum_variable_never_consumed(self):
        passed, msg = test_error_enum_variable_never_consumed()
        self.assertTrue(passed, msg)

    def test_partial_wildcard_on_linear_enum(self):
        passed, msg = test_error_partial_wildcard_on_linear_enum()
        self.assertTrue(passed, msg)

    def test_move_enum_then_use_original(self):
        passed, msg = test_error_move_enum_then_use_original()
        self.assertTrue(passed, msg)

    def test_payload_consumed_only_when_guard_true(self):
        passed, msg = test_error_payload_consumed_only_when_guard_true()
        self.assertTrue(passed, msg)

    def test_payload_not_consumed_in_one_branch(self):
        passed, msg = test_error_payload_not_consumed_in_one_branch()
        self.assertTrue(passed, msg)

    def test_enum_created_in_loop_not_consumed(self):
        passed, msg = test_error_enum_created_in_loop_not_consumed()
        self.assertTrue(passed, msg)

    def test_return_nonlinear_payload_directly(self):
        passed, msg = test_error_return_nonlinear_payload_directly()
        self.assertTrue(passed, msg)

    def test_filehandle_token_not_consumed(self):
        passed, msg = test_error_filehandle_token_not_consumed()
        self.assertTrue(passed, msg)

    def test_buffer_token_not_consumed(self):
        passed, msg = test_error_buffer_token_not_consumed()
        self.assertTrue(passed, msg)

    def test_nested_enum_payload_not_consumed(self):
        passed, msg = test_error_nested_enum_payload_not_consumed()
        self.assertTrue(passed, msg)

    def test_wrapper_payload_not_consumed(self):
        passed, msg = test_error_wrapper_payload_not_consumed()
        self.assertTrue(passed, msg)

    def test_double_wrapper_not_consumed(self):
        passed, msg = test_error_double_wrapper_not_consumed()
        self.assertTrue(passed, msg)

    def test_resource_pair_not_fully_consumed(self):
        passed, msg = test_error_resource_pair_not_fully_consumed()
        self.assertTrue(passed, msg)

    def test_tree_node_leaf_not_consumed(self):
        passed, msg = test_error_tree_node_leaf_not_consumed()
        self.assertTrue(passed, msg)

    def test_tree_node_node_partial_consumed(self):
        passed, msg = test_error_tree_node_node_partial_consumed()
        self.assertTrue(passed, msg)

    def test_outer_consumed_inner_not(self):
        passed, msg = test_error_outer_consumed_inner_not()
        self.assertTrue(passed, msg)

    def test_move_complex_enum_then_use_original(self):
        passed, msg = test_error_move_complex_enum_then_use_original()
        self.assertTrue(passed, msg)

    def test_optional_file_not_closed(self):
        passed, msg = test_error_optional_file_not_closed()
        self.assertTrue(passed, msg)

    def test_optional_file_never_consumed(self):
        passed, msg = test_error_optional_file_never_consumed()
        self.assertTrue(passed, msg)

    def test_copy_optional_file(self):
        passed, msg = test_error_copy_optional_file()
        self.assertTrue(passed, msg)

    def test_file_result_ok_not_closed(self):
        passed, msg = test_error_file_result_ok_not_closed()
        self.assertTrue(passed, msg)

    def test_read_result_data_not_consumed(self):
        passed, msg = test_error_read_result_data_not_consumed()
        self.assertTrue(passed, msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
