#pragma once

#include "strata/kernels/rope_scaling.hpp"

namespace strata::kernels {
void native_rope_set_enabled(bool enabled);
bool native_rope_enabled();

// Pinned CUDA text-only IMRoPE: F32 rows, 64 rotated channels, equal text positions in all four
// IMRoPE sections, ggml's rope_yarn applied via the process's resolved RopeScaling (none = the
// trained rotation, bit for bit). Each device position
// must be nonnegative. The position buffer remains live through graph replay.
// Supports head_dim 128/256 and exact x==out; partial overlap is rejected.
// Explicit stream required. No allocation or synchronization.
void native_rope_apply(const float* x, float* out, int rows, int head_dim,
                       int n_rot, const RopeScaling& scaling, const int* positions, void* stream);
// The same for several tokens' heads in one launch: row r is head r % heads of token r / heads, and takes
// positions[(r / heads) * pos_stride + r % heads] (a verify window's per-token position vectors).  Each row is
// bitwise what native_rope_apply gives it.
void native_rope_apply_tokens(const float* x, float* out, int rows, int head_dim, int n_rot, const RopeScaling& scaling,
                              const int* positions, int heads, int pos_stride, void* stream);
}
