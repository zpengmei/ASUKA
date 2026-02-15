cdef inline u64 _splitmix64_next(u64* state) noexcept nogil:
    """SplitMix64 PRNG step (fast, deterministic, good enough for proposals)."""
    cdef u64 z
    state[0] = state[0] + <u64>0x9E3779B97F4A7C15
    z = state[0]
    z = (z ^ (z >> 30)) * <u64>0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * <u64>0x94D049BB133111EB
    return z ^ (z >> 31)


cdef inline double _rand_u01(u64* state) noexcept nogil:
    """Return a double in [0,1) from the top 53 bits of SplitMix64."""
    cdef u64 x = _splitmix64_next(state)
    return (<double>(x >> 11)) * (1.0 / 9007199254740992.0)


cdef inline int _rand_below(u64* state, int n) noexcept nogil:
    """Unbiased random int in [0,n) using rejection on a 64-bit RNG."""
    if n <= 1:
        return 0
    cdef u64 nn = <u64>n
    cdef u64 limit = (<u64>0xFFFFFFFFFFFFFFFF // nn) * nn
    cdef u64 r
    while True:
        r = _splitmix64_next(state)
        if r < limit:
            return <int>(r % nn)


cdef inline bint _contains_sorted_i32(cnp.int32_t[::1] arr, int key) noexcept nogil:
    cdef int lo = 0
    cdef int hi = <int>arr.shape[0]
    cdef int mid
    cdef int v
    while lo < hi:
        mid = (lo + hi) >> 1
        v = <int>arr[mid]
        if v < key:
            lo = mid + 1
        else:
            hi = mid
    return lo < <int>arr.shape[0] and <int>arr[lo] == key

