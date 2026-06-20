#include <metal_stdlib>
using namespace metal;

struct Conv3DParams {
    uint b;
    uint t;
    uint h;
    uint w;
    uint cin;
    uint cout;
    uint kt;
    uint kh;
    uint kw;
    int padT;
    int padH;
    int padW;
    uint cacheT;
    uint cacheValid;
    uint hasBias;
};

struct NormParams {
    uint b;
    uint t;
    uint h;
    uint w;
    uint c;
    uint applySilu;
    uint hasBias;
};

struct Up2DConvParams {
    uint b;
    uint t;
    uint h;
    uint w;
    uint cin;
    uint cout;
    uint kh;
    uint kw;
    int padH;
    int padW;
    uint hasBias;
};

struct CausalSourceParams {
    uint b;
    uint t;
    uint h;
    uint w;
    uint c;
    uint preT;
    uint cacheT;
    uint cacheValid;
};

struct AttentionSplitParams {
    uint bt;
    uint n;
    uint c;
};

struct TensorParams {
    uint b;
    uint t;
    uint h;
    uint w;
    uint c;
};

#define MAT_TILE_M 16
#define MAT_TILE_N 16
#define MAT_TILE_K 32
#define NORM_THREADS 256
#define ATT_BQ 4
#define ATT_BK 8
#define ATT_C 384
#define ATT_THREADS 256

static inline ulong idx5(uint b, uint t, uint h, uint w, uint c,
                        uint T, uint H, uint W, uint C) {
    return (((((ulong)b * T + t) * H + h) * W + w) * C + c);
}

static inline half silu_half(float x) {
    return half(x / (1.0f + exp(-x)));
}

kernel void scale_latent(
    device const half *z [[buffer(0)]],
    device const half *mean [[buffer(1)]],
    device const half *std [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant TensorParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.b * p.t * p.h * p.w * p.c;
    if (gid >= total) return;
    uint c = gid % p.c;
    out[gid] = half(float(z[gid]) * float(std[c]) + float(mean[c]));
}

kernel void scale_latent4(
    device const half *z [[buffer(0)]],
    device const half *mean [[buffer(1)]],
    device const half *std [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant TensorParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.b * p.t * p.h * p.w * p.c;
    uint base = gid * 4;
    for (uint lane = 0; lane < 4; ++lane) {
        uint i = base + lane;
        if (i >= total) return;
        uint c = i % p.c;
        out[i] = half(float(z[i]) * float(std[c]) + float(mean[c]));
    }
}

kernel void conv1x1_tiled(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant Conv3DParams &p [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group3 [[threadgroup_position_in_grid]]
) {
    threadgroup half aTile[MAT_TILE_M * MAT_TILE_K];
    threadgroup half bTile[MAT_TILE_K * MAT_TILE_N];

    uint2 tid = tid3.xy;
    uint2 group = group3.xy;
    uint row = group.y * MAT_TILE_M + tid.y;
    uint col = group.x * MAT_TILE_N + tid.x;
    uint rows = p.b * p.t * p.h * p.w;
    float acc = 0.0f;

    for (uint k0 = 0; k0 < p.cin; k0 += MAT_TILE_K) {
        for (uint load = tid.y * MAT_TILE_N + tid.x; load < MAT_TILE_M * MAT_TILE_K; load += MAT_TILE_M * MAT_TILE_N) {
            uint lm = load / MAT_TILE_K;
            uint lk = load % MAT_TILE_K;
            uint srcRow = group.y * MAT_TILE_M + lm;
            uint ci = k0 + lk;
            aTile[load] = (srcRow < rows && ci < p.cin) ? x[(ulong)srcRow * p.cin + ci] : half(0.0);
        }
        for (uint load = tid.y * MAT_TILE_N + tid.x; load < MAT_TILE_K * MAT_TILE_N; load += MAT_TILE_M * MAT_TILE_N) {
            uint lk = load / MAT_TILE_N;
            uint ln = load % MAT_TILE_N;
            uint ci = k0 + lk;
            uint co = group.x * MAT_TILE_N + ln;
            bTile[load] = (ci < p.cin && co < p.cout) ? weight[(ulong)ci * p.cout + co] : half(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < rows && col < p.cout) {
            for (uint kk = 0; kk < MAT_TILE_K; ++kk) {
                acc += float(aTile[tid.y * MAT_TILE_K + kk]) * float(bTile[kk * MAT_TILE_N + tid.x]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && col < p.cout) {
        if (p.hasBias != 0) {
            acc += float(bias[col]);
        }
        out[(ulong)row * p.cout + col] = half(acc);
    }
}

kernel void conv3d_causal(
    device const half *x [[buffer(0)]],
    device const half *cache [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant Conv3DParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong total = (ulong)p.b * p.t * p.h * p.w * p.cout;
    if ((ulong)gid >= total) return;

    uint co = gid % p.cout;
    uint ow = (gid / p.cout) % p.w;
    uint oh = (gid / (p.cout * p.w)) % p.h;
    uint ot = (gid / (p.cout * p.w * p.h)) % p.t;
    uint ob = gid / (p.cout * p.w * p.h * p.t);

    float acc = p.hasBias != 0 ? float(bias[co]) : 0.0f;
    for (uint ktt = 0; ktt < p.kt; ++ktt) {
        int it = int(ot) + int(ktt) - p.padT;
        for (uint khh = 0; khh < p.kh; ++khh) {
            int ih = int(oh) + int(khh) - p.padH;
            if (ih < 0 || ih >= int(p.h)) continue;
            for (uint kww = 0; kww < p.kw; ++kww) {
                int iw = int(ow) + int(kww) - p.padW;
                if (iw < 0 || iw >= int(p.w)) continue;
                for (uint ci = 0; ci < p.cin; ++ci) {
                    float xv = 0.0f;
                    if (it >= 0 && it < int(p.t)) {
                        xv = float(x[idx5(ob, uint(it), uint(ih), uint(iw), ci, p.t, p.h, p.w, p.cin)]);
                    } else if (it < 0 && p.cacheValid != 0) {
                        int ct = int(p.cacheT) + it;
                        if (ct >= 0 && ct < int(p.cacheT)) {
                            xv = float(cache[idx5(ob, uint(ct), uint(ih), uint(iw), ci, p.cacheT, p.h, p.w, p.cin)]);
                        }
                    }
                    ulong wi = (((((ulong)ktt * p.kh + khh) * p.kw + kww) * p.cin + ci) * p.cout + co);
                    acc += xv * float(weight[wi]);
                }
            }
        }
    }
    out[gid] = half(acc);
}

kernel void conv3d_causal_co4(
    device const half *x [[buffer(0)]],
    device const half *cache [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant Conv3DParams &p [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint coutBlocks = (p.cout + 3) / 4;
    ulong total = (ulong)p.b * p.t * p.h * p.w * coutBlocks;
    if ((ulong)gid >= total) return;

    uint coBase = (gid % coutBlocks) * 4;
    uint ow = (gid / coutBlocks) % p.w;
    uint oh = (gid / (coutBlocks * p.w)) % p.h;
    uint ot = (gid / (coutBlocks * p.w * p.h)) % p.t;
    uint ob = gid / (coutBlocks * p.w * p.h * p.t);

    float4 acc = float4(0.0f);
    if (p.hasBias != 0) {
        acc.x = coBase + 0 < p.cout ? float(bias[coBase + 0]) : 0.0f;
        acc.y = coBase + 1 < p.cout ? float(bias[coBase + 1]) : 0.0f;
        acc.z = coBase + 2 < p.cout ? float(bias[coBase + 2]) : 0.0f;
        acc.w = coBase + 3 < p.cout ? float(bias[coBase + 3]) : 0.0f;
    }

    for (uint ktt = 0; ktt < p.kt; ++ktt) {
        int it = int(ot) + int(ktt) - p.padT;
        for (uint khh = 0; khh < p.kh; ++khh) {
            int ih = int(oh) + int(khh) - p.padH;
            if (ih < 0 || ih >= int(p.h)) continue;
            for (uint kww = 0; kww < p.kw; ++kww) {
                int iw = int(ow) + int(kww) - p.padW;
                if (iw < 0 || iw >= int(p.w)) continue;
                for (uint ci = 0; ci < p.cin; ++ci) {
                    float xv = 0.0f;
                    if (it >= 0 && it < int(p.t)) {
                        xv = float(x[idx5(ob, uint(it), uint(ih), uint(iw), ci, p.t, p.h, p.w, p.cin)]);
                    } else if (it < 0 && p.cacheValid != 0) {
                        int ct = int(p.cacheT) + it;
                        if (ct >= 0 && ct < int(p.cacheT)) {
                            xv = float(cache[idx5(ob, uint(ct), uint(ih), uint(iw), ci, p.cacheT, p.h, p.w, p.cin)]);
                        }
                    }
                    ulong wi = (((((ulong)ktt * p.kh + khh) * p.kw + kww) * p.cin + ci) * p.cout + coBase);
                    if (coBase + 0 < p.cout) acc.x += xv * float(weight[wi + 0]);
                    if (coBase + 1 < p.cout) acc.y += xv * float(weight[wi + 1]);
                    if (coBase + 2 < p.cout) acc.z += xv * float(weight[wi + 2]);
                    if (coBase + 3 < p.cout) acc.w += xv * float(weight[wi + 3]);
                }
            }
        }
    }

    ulong outBase = idx5(ob, ot, oh, ow, coBase, p.t, p.h, p.w, p.cout);
    if (coBase + 0 < p.cout) out[outBase + 0] = half(acc.x);
    if (coBase + 1 < p.cout) out[outBase + 1] = half(acc.y);
    if (coBase + 2 < p.cout) out[outBase + 2] = half(acc.z);
    if (coBase + 3 < p.cout) out[outBase + 3] = half(acc.w);
}

kernel void conv3d_causal_implicit_gemm(
    device const half *x [[buffer(0)]],
    device const half *cache [[buffer(1)]],
    device const half *weight [[buffer(2)]],
    device const half *bias [[buffer(3)]],
    device half *out [[buffer(4)]],
    constant Conv3DParams &p [[buffer(5)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group3 [[threadgroup_position_in_grid]]
) {
    threadgroup half aTile[MAT_TILE_M * MAT_TILE_K];
    threadgroup half bTile[MAT_TILE_K * MAT_TILE_N];

    uint2 tid = tid3.xy;
    uint2 group = group3.xy;
    uint rows = p.b * p.t * p.h * p.w;
    uint kTotal = p.kt * p.kh * p.kw * p.cin;
    uint row = group.y * MAT_TILE_M + tid.y;
    uint co = group.x * MAT_TILE_N + tid.x;
    float acc = 0.0f;

    uint ob = 0, ot = 0, oh = 0, ow = 0;
    if (row < rows) {
        uint r = row;
        ow = r % p.w; r /= p.w;
        oh = r % p.h; r /= p.h;
        ot = r % p.t; r /= p.t;
        ob = r;
    }

    for (uint k0 = 0; k0 < kTotal; k0 += MAT_TILE_K) {
        for (uint load = tid.y * MAT_TILE_N + tid.x; load < MAT_TILE_M * MAT_TILE_K; load += MAT_TILE_M * MAT_TILE_N) {
            uint lm = load / MAT_TILE_K;
            uint lk = load % MAT_TILE_K;
            uint srcRow = group.y * MAT_TILE_M + lm;
            uint k = k0 + lk;
            half av = half(0.0);
            if (srcRow < rows && k < kTotal) {
                uint rr = srcRow;
                uint sow = rr % p.w; rr /= p.w;
                uint soh = rr % p.h; rr /= p.h;
                uint sot = rr % p.t; rr /= p.t;
                uint sob = rr;

                uint ci = k % p.cin;
                uint tmp = k / p.cin;
                uint kww = tmp % p.kw; tmp /= p.kw;
                uint khh = tmp % p.kh; tmp /= p.kh;
                uint ktt = tmp;

                int it = int(sot) + int(ktt) - p.padT;
                int ih = int(soh) + int(khh) - p.padH;
                int iw = int(sow) + int(kww) - p.padW;
                if (ih >= 0 && ih < int(p.h) && iw >= 0 && iw < int(p.w)) {
                    if (it >= 0 && it < int(p.t)) {
                        av = x[idx5(sob, uint(it), uint(ih), uint(iw), ci, p.t, p.h, p.w, p.cin)];
                    } else if (it < 0 && p.cacheValid != 0) {
                        int ct = int(p.cacheT) + it;
                        if (ct >= 0 && ct < int(p.cacheT)) {
                            av = cache[idx5(sob, uint(ct), uint(ih), uint(iw), ci, p.cacheT, p.h, p.w, p.cin)];
                        }
                    }
                }
            }
            aTile[load] = av;
        }

        for (uint load = tid.y * MAT_TILE_N + tid.x; load < MAT_TILE_K * MAT_TILE_N; load += MAT_TILE_M * MAT_TILE_N) {
            uint lk = load / MAT_TILE_N;
            uint ln = load % MAT_TILE_N;
            uint k = k0 + lk;
            uint outC = group.x * MAT_TILE_N + ln;
            bTile[load] = (k < kTotal && outC < p.cout) ? weight[(ulong)k * p.cout + outC] : half(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < rows && co < p.cout) {
            for (uint kk = 0; kk < MAT_TILE_K; ++kk) {
                acc += float(aTile[tid.y * MAT_TILE_K + kk]) * float(bTile[kk * MAT_TILE_N + tid.x]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && co < p.cout) {
        if (p.hasBias != 0) {
            acc += float(bias[co]);
        }
        out[idx5(ob, ot, oh, ow, co, p.t, p.h, p.w, p.cout)] = half(acc);
    }
}

kernel void materialize_causal_source(
    device const half *x [[buffer(0)]],
    device const half *cache [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant CausalSourceParams &p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint sourceT = p.t + p.preT;
    ulong total = (ulong)p.b * sourceT * p.h * p.w * p.c;
    if ((ulong)gid >= total) return;

    uint c = gid % p.c;
    uint ow = (gid / p.c) % p.w;
    uint oh = (gid / (p.c * p.w)) % p.h;
    uint st = (gid / (p.c * p.w * p.h)) % sourceT;
    uint b = gid / (p.c * p.w * p.h * sourceT);

    half value = half(0.0);
    if (st < p.preT) {
        int ct = int(p.cacheT) - int(p.preT) + int(st);
        if (p.cacheValid != 0 && ct >= 0 && ct < int(p.cacheT)) {
            value = cache[idx5(b, uint(ct), oh, ow, c, p.cacheT, p.h, p.w, p.c)];
        }
    } else {
        uint it = st - p.preT;
        value = x[idx5(b, it, oh, ow, c, p.t, p.h, p.w, p.c)];
    }
    out[gid] = value;
}

kernel void update_cache2(
    device const half *x [[buffer(0)]],
    device half *oldAndNewCache [[buffer(1)]],
    constant TensorParams &p [[buffer(2)]],
    constant uint &hadPrevious [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.b * 2 * p.h * p.w * p.c;
    if (gid >= total) return;

    uint c = gid % p.c;
    uint w = (gid / p.c) % p.w;
    uint h = (gid / (p.c * p.w)) % p.h;
    uint ct = (gid / (p.c * p.w * p.h)) % 2;
    uint b = gid / (p.c * p.w * p.h * 2);

    half value = half(0.0);
    if (p.t >= 2) {
        uint srcT = p.t - 2 + ct;
        value = x[idx5(b, srcT, h, w, c, p.t, p.h, p.w, p.c)];
    } else if (p.t == 1) {
        if (ct == 0) {
            if (hadPrevious != 0) {
                value = oldAndNewCache[idx5(b, 1, h, w, c, 2, p.h, p.w, p.c)];
            }
        } else {
            value = x[idx5(b, 0, h, w, c, p.t, p.h, p.w, p.c)];
        }
    }
    oldAndNewCache[idx5(b, ct, h, w, c, 2, p.h, p.w, p.c)] = value;
}

kernel void rmsnorm_silu_tg(
    device const half *x [[buffer(0)]],
    device const half *gamma [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant NormParams &p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]
) {
    threadgroup float partial[NORM_THREADS];
    uint vector = group.x;
    uint vectors = p.b * p.t * p.h * p.w;
    if (vector >= vectors) return;

    ulong base = (ulong)vector * p.c;
    float ss = 0.0f;
    for (uint c = tid; c < p.c; c += NORM_THREADS) {
        float v = float(x[base + c]);
        ss += v * v;
    }
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = NORM_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = rsqrt(max(partial[0], 1.0e-12f));
    float scale = sqrt(float(p.c));
    for (uint c = tid; c < p.c; c += NORM_THREADS) {
        float y = float(x[base + c]) * inv * scale * float(gamma[c]);
        if (p.hasBias != 0) {
            y += float(bias[c]);
        }
        out[base + c] = p.applySilu != 0 ? silu_half(y) : half(y);
    }
}

kernel void rmsnorm_silu(
    device const half *x [[buffer(0)]],
    device const half *gamma [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant NormParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vectors = p.b * p.t * p.h * p.w;
    if (gid >= vectors) return;

    ulong base = (ulong)gid * p.c;
    float ss = 0.0f;
    for (uint c = 0; c < p.c; ++c) {
        float v = float(x[base + c]);
        ss += v * v;
    }
    float inv = rsqrt(max(ss, 1.0e-12f));
    float scale = sqrt(float(p.c));
    for (uint c = 0; c < p.c; ++c) {
        float y = float(x[base + c]) * inv * scale * float(gamma[c]);
        if (p.hasBias != 0) {
            y += float(bias[c]);
        }
        out[base + c] = p.applySilu != 0 ? silu_half(y) : half(y);
    }
}

kernel void add_tensors(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= count) return;
    out[gid] = half(float(a[gid]) + float(b[gid]));
}

kernel void add_tensors4(
    device const half *a [[buffer(0)]],
    device const half *b [[buffer(1)]],
    device half *out [[buffer(2)]],
    constant uint &count [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint base = gid * 4;
    for (uint lane = 0; lane < 4; ++lane) {
        uint i = base + lane;
        if (i >= count) return;
        out[i] = half(float(a[i]) + float(b[i]));
    }
}

kernel void upsample2d_conv2d(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant Up2DConvParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outH = p.h * 2;
    uint outW = p.w * 2;
    ulong total = (ulong)p.b * p.t * outH * outW * p.cout;
    if ((ulong)gid >= total) return;

    uint co = gid % p.cout;
    uint ow = (gid / p.cout) % outW;
    uint oh = (gid / (p.cout * outW)) % outH;
    uint ot = (gid / (p.cout * outW * outH)) % p.t;
    uint ob = gid / (p.cout * outW * outH * p.t);

    float acc = p.hasBias != 0 ? float(bias[co]) : 0.0f;
    for (uint khh = 0; khh < p.kh; ++khh) {
        int expandedH = int(oh) + int(khh) - p.padH;
        if (expandedH < 0 || expandedH >= int(outH)) continue;
        uint ih = uint(expandedH / 2);
        for (uint kww = 0; kww < p.kw; ++kww) {
            int expandedW = int(ow) + int(kww) - p.padW;
            if (expandedW < 0 || expandedW >= int(outW)) continue;
            uint iw = uint(expandedW / 2);
            for (uint ci = 0; ci < p.cin; ++ci) {
                ulong wi = ((((ulong)khh * p.kw + kww) * p.cin + ci) * p.cout + co);
                acc += float(x[idx5(ob, ot, ih, iw, ci, p.t, p.h, p.w, p.cin)]) * float(weight[wi]);
            }
        }
    }
    out[gid] = half(acc);
}

kernel void upsample2d_conv2d_co4(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant Up2DConvParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outH = p.h * 2;
    uint outW = p.w * 2;
    uint coutBlocks = (p.cout + 3) / 4;
    ulong total = (ulong)p.b * p.t * outH * outW * coutBlocks;
    if ((ulong)gid >= total) return;

    uint coBase = (gid % coutBlocks) * 4;
    uint ow = (gid / coutBlocks) % outW;
    uint oh = (gid / (coutBlocks * outW)) % outH;
    uint ot = (gid / (coutBlocks * outW * outH)) % p.t;
    uint ob = gid / (coutBlocks * outW * outH * p.t);

    float4 acc = float4(0.0f);
    if (p.hasBias != 0) {
        acc.x = coBase + 0 < p.cout ? float(bias[coBase + 0]) : 0.0f;
        acc.y = coBase + 1 < p.cout ? float(bias[coBase + 1]) : 0.0f;
        acc.z = coBase + 2 < p.cout ? float(bias[coBase + 2]) : 0.0f;
        acc.w = coBase + 3 < p.cout ? float(bias[coBase + 3]) : 0.0f;
    }

    for (uint khh = 0; khh < p.kh; ++khh) {
        int expandedH = int(oh) + int(khh) - p.padH;
        if (expandedH < 0 || expandedH >= int(outH)) continue;
        uint ih = uint(expandedH / 2);
        for (uint kww = 0; kww < p.kw; ++kww) {
            int expandedW = int(ow) + int(kww) - p.padW;
            if (expandedW < 0 || expandedW >= int(outW)) continue;
            uint iw = uint(expandedW / 2);
            for (uint ci = 0; ci < p.cin; ++ci) {
                float xv = float(x[idx5(ob, ot, ih, iw, ci, p.t, p.h, p.w, p.cin)]);
                ulong wi = ((((ulong)khh * p.kw + kww) * p.cin + ci) * p.cout + coBase);
                if (coBase + 0 < p.cout) acc.x += xv * float(weight[wi + 0]);
                if (coBase + 1 < p.cout) acc.y += xv * float(weight[wi + 1]);
                if (coBase + 2 < p.cout) acc.z += xv * float(weight[wi + 2]);
                if (coBase + 3 < p.cout) acc.w += xv * float(weight[wi + 3]);
            }
        }
    }

    ulong outBase = idx5(ob, ot, oh, ow, coBase, p.t, outH, outW, p.cout);
    if (coBase + 0 < p.cout) out[outBase + 0] = half(acc.x);
    if (coBase + 1 < p.cout) out[outBase + 1] = half(acc.y);
    if (coBase + 2 < p.cout) out[outBase + 2] = half(acc.z);
    if (coBase + 3 < p.cout) out[outBase + 3] = half(acc.w);
}

kernel void upsample2d_conv2d_implicit_gemm(
    device const half *x [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *bias [[buffer(2)]],
    device half *out [[buffer(3)]],
    constant Up2DConvParams &p [[buffer(4)]],
    uint3 tid3 [[thread_position_in_threadgroup]],
    uint3 group3 [[threadgroup_position_in_grid]]
) {
    threadgroup half aTile[MAT_TILE_M * MAT_TILE_K];
    threadgroup half bTile[MAT_TILE_K * MAT_TILE_N];

    uint2 tid = tid3.xy;
    uint2 group = group3.xy;
    uint outH = p.h * 2;
    uint outW = p.w * 2;
    uint rows = p.b * p.t * outH * outW;
    uint kTotal = p.kh * p.kw * p.cin;
    uint row = group.y * MAT_TILE_M + tid.y;
    uint co = group.x * MAT_TILE_N + tid.x;
    float acc = 0.0f;

    uint ob = 0, ot = 0, oh = 0, ow = 0;
    if (row < rows) {
        uint r = row;
        ow = r % outW; r /= outW;
        oh = r % outH; r /= outH;
        ot = r % p.t; r /= p.t;
        ob = r;
    }

    for (uint k0 = 0; k0 < kTotal; k0 += MAT_TILE_K) {
        for (uint load = tid.y * MAT_TILE_N + tid.x; load < MAT_TILE_M * MAT_TILE_K; load += MAT_TILE_M * MAT_TILE_N) {
            uint lm = load / MAT_TILE_K;
            uint lk = load % MAT_TILE_K;
            uint srcRow = group.y * MAT_TILE_M + lm;
            uint k = k0 + lk;
            half av = half(0.0);
            if (srcRow < rows && k < kTotal) {
                uint rr = srcRow;
                uint sow = rr % outW; rr /= outW;
                uint soh = rr % outH; rr /= outH;
                uint sot = rr % p.t; rr /= p.t;
                uint sob = rr;

                uint ci = k % p.cin;
                uint tmp = k / p.cin;
                uint kww = tmp % p.kw; tmp /= p.kw;
                uint khh = tmp;

                int expandedH = int(soh) + int(khh) - p.padH;
                int expandedW = int(sow) + int(kww) - p.padW;
                if (expandedH >= 0 && expandedH < int(outH) && expandedW >= 0 && expandedW < int(outW)) {
                    uint ih = uint(expandedH / 2);
                    uint iw = uint(expandedW / 2);
                    av = x[idx5(sob, sot, ih, iw, ci, p.t, p.h, p.w, p.cin)];
                }
            }
            aTile[load] = av;
        }

        for (uint load = tid.y * MAT_TILE_N + tid.x; load < MAT_TILE_K * MAT_TILE_N; load += MAT_TILE_M * MAT_TILE_N) {
            uint lk = load / MAT_TILE_N;
            uint ln = load % MAT_TILE_N;
            uint k = k0 + lk;
            uint outC = group.x * MAT_TILE_N + ln;
            bTile[load] = (k < kTotal && outC < p.cout) ? weight[(ulong)k * p.cout + outC] : half(0.0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (row < rows && co < p.cout) {
            for (uint kk = 0; kk < MAT_TILE_K; ++kk) {
                acc += float(aTile[tid.y * MAT_TILE_K + kk]) * float(bTile[kk * MAT_TILE_N + tid.x]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < rows && co < p.cout) {
        if (p.hasBias != 0) {
            acc += float(bias[co]);
        }
        out[idx5(ob, ot, oh, ow, co, p.t, outH, outW, p.cout)] = half(acc);
    }
}

kernel void split_channel_to_time2(
    device const half *x [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant TensorParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outC = p.c / 2;
    uint outT = p.t * 2;
    ulong total = (ulong)p.b * outT * p.h * p.w * outC;
    if ((ulong)gid >= total) return;

    uint c = gid % outC;
    uint w = (gid / outC) % p.w;
    uint h = (gid / (outC * p.w)) % p.h;
    uint ot = (gid / (outC * p.w * p.h)) % outT;
    uint b = gid / (outC * p.w * p.h * outT);
    uint it = ot / 2;
    uint r = ot & 1;
    uint ic = r * outC + c;
    out[gid] = x[idx5(b, it, h, w, ic, p.t, p.h, p.w, p.c)];
}

kernel void depth_to_space2_phase_major(
    device const half *x [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant TensorParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint outH = p.h * 2;
    uint outW = p.w * 2;
    ulong total = (ulong)p.b * p.t * outH * outW * p.c;
    if ((ulong)gid >= total) return;

    uint c = gid % p.c;
    uint ow = (gid / p.c) % outW;
    uint oh = (gid / (p.c * outW)) % outH;
    uint t = (gid / (p.c * outW * outH)) % p.t;
    uint b = gid / (p.c * outW * outH * p.t);
    uint phase = (oh & 1) * 2 + (ow & 1);
    uint ih = oh >> 1;
    uint iw = ow >> 1;
    uint ic = phase * p.c + c;
    out[gid] = x[idx5(b, t, ih, iw, ic, p.t, p.h, p.w, p.c * 4)];
}

kernel void split_qkv_for_sdpa(
    device const half *qkv [[buffer(0)]],
    device half *q [[buffer(1)]],
    device half *k [[buffer(2)]],
    device half *v [[buffer(3)]],
    constant AttentionSplitParams &p [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    ulong total = (ulong)p.bt * p.n * p.c;
    if ((ulong)gid >= total) return;

    uint c = gid % p.c;
    uint token = (gid / p.c) % p.n;
    uint bt = gid / (p.c * p.n);
    ulong src = ((ulong)bt * p.n + token) * p.c * 3 + c;
    q[gid] = qkv[src];
    k[gid] = qkv[src + p.c];
    v[gid] = qkv[src + p.c * 2];
}

kernel void attention_spatial_tiled_384(
    device const half *qkv [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant TensorParams &p [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group3 [[threadgroup_position_in_grid]]
) {
    threadgroup half qTile[ATT_BQ * ATT_C];
    threadgroup half kTile[ATT_BK * ATT_C];
    threadgroup half vTile[ATT_BK * ATT_C];
    threadgroup float scores[ATT_BQ * ATT_BK];
    threadgroup float weights[ATT_BQ * ATT_BK];
    threadgroup float acc[ATT_BQ * ATT_C];
    threadgroup float mRow[ATT_BQ];
    threadgroup float lRow[ATT_BQ];
    threadgroup float alphaRow[ATT_BQ];

    uint n = p.h * p.w;
    uint qStart = group3.x * ATT_BQ;
    uint bt = group3.y;
    if (bt >= p.b * p.t || p.c > ATT_C) return;
    ulong btBase = (ulong)bt * n * p.c * 3;

    for (uint i = tid; i < ATT_BQ * p.c; i += ATT_THREADS) {
        uint qi = i / p.c;
        uint d = i % p.c;
        uint qIndex = qStart + qi;
        qTile[qi * ATT_C + d] = qIndex < n ? qkv[btBase + (ulong)qIndex * p.c * 3 + d] : half(0.0);
        acc[qi * ATT_C + d] = 0.0f;
    }
    if (tid < ATT_BQ) {
        mRow[tid] = -3.402823466e38f;
        lRow[tid] = 0.0f;
        alphaRow[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float invSqrt = rsqrt(float(p.c));
    for (uint kStart = 0; kStart < n; kStart += ATT_BK) {
        for (uint i = tid; i < ATT_BK * p.c; i += ATT_THREADS) {
            uint kj = i / p.c;
            uint d = i % p.c;
            uint kIndex = kStart + kj;
            if (kIndex < n) {
                ulong base = btBase + (ulong)kIndex * p.c * 3;
                kTile[kj * ATT_C + d] = qkv[base + p.c + d];
                vTile[kj * ATT_C + d] = qkv[base + p.c * 2 + d];
            } else {
                kTile[kj * ATT_C + d] = half(0.0);
                vTile[kj * ATT_C + d] = half(0.0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < ATT_BQ * ATT_BK) {
            uint qi = tid / ATT_BK;
            uint kj = tid % ATT_BK;
            uint qIndex = qStart + qi;
            uint kIndex = kStart + kj;
            float dot = -3.402823466e38f;
            if (qIndex < n && kIndex < n) {
                dot = 0.0f;
                for (uint d = 0; d < p.c; ++d) {
                    dot += float(qTile[qi * ATT_C + d]) * float(kTile[kj * ATT_C + d]);
                }
                dot *= invSqrt;
            }
            scores[qi * ATT_BK + kj] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid < ATT_BQ) {
            float tileMax = -3.402823466e38f;
            for (uint kj = 0; kj < ATT_BK; ++kj) {
                tileMax = max(tileMax, scores[tid * ATT_BK + kj]);
            }
            float mNew = max(mRow[tid], tileMax);
            float alpha = exp(mRow[tid] - mNew);
            float tileSum = 0.0f;
            for (uint kj = 0; kj < ATT_BK; ++kj) {
                float w = exp(scores[tid * ATT_BK + kj] - mNew);
                weights[tid * ATT_BK + kj] = w;
                tileSum += w;
            }
            alphaRow[tid] = alpha;
            mRow[tid] = mNew;
            lRow[tid] = lRow[tid] * alpha + tileSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = tid; i < ATT_BQ * p.c; i += ATT_THREADS) {
            uint qi = i / p.c;
            uint d = i % p.c;
            float sum = 0.0f;
            for (uint kj = 0; kj < ATT_BK; ++kj) {
                sum += weights[qi * ATT_BK + kj] * float(vTile[kj * ATT_C + d]);
            }
            acc[qi * ATT_C + d] = acc[qi * ATT_C + d] * alphaRow[qi] + sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = tid; i < ATT_BQ * p.c; i += ATT_THREADS) {
        uint qi = i / p.c;
        uint d = i % p.c;
        uint qIndex = qStart + qi;
        if (qIndex < n) {
            out[(ulong)bt * n * p.c + (ulong)qIndex * p.c + d] = half(acc[qi * ATT_C + d] / max(lRow[qi], 1.0e-12f));
        }
    }
}

kernel void attention_spatial_naive(
    device const half *qkv [[buffer(0)]],
    device half *out [[buffer(1)]],
    constant TensorParams &p [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint n = p.h * p.w;
    ulong total = (ulong)p.b * p.t * n * p.c;
    if ((ulong)gid >= total) return;

    uint c = gid % p.c;
    uint q = (gid / p.c) % n;
    uint bt = gid / (p.c * n);
    uint b = bt / p.t;
    uint t = bt % p.t;
    uint qh = q / p.w;
    uint qw = q % p.w;
    ulong qbase = idx5(b, t, qh, qw, 0, p.t, p.h, p.w, p.c * 3);

    float maxLogit = -3.402823466e38f;
    float invSqrt = rsqrt(float(p.c));
    for (uint k = 0; k < n; ++k) {
        uint kh = k / p.w;
        uint kw = k % p.w;
        ulong kbase = idx5(b, t, kh, kw, p.c, p.t, p.h, p.w, p.c * 3);
        float dot = 0.0f;
        for (uint d = 0; d < p.c; ++d) {
            dot += float(qkv[qbase + d]) * float(qkv[kbase + d]);
        }
        maxLogit = max(maxLogit, dot * invSqrt);
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (uint k = 0; k < n; ++k) {
        uint kh = k / p.w;
        uint kw = k % p.w;
        ulong kbase = idx5(b, t, kh, kw, p.c, p.t, p.h, p.w, p.c * 3);
        float dot = 0.0f;
        for (uint d = 0; d < p.c; ++d) {
            dot += float(qkv[qbase + d]) * float(qkv[kbase + d]);
        }
        float a = exp(dot * invSqrt - maxLogit);
        denom += a;
        ulong vbase = idx5(b, t, kh, kw, p.c * 2, p.t, p.h, p.w, p.c * 3);
        acc += a * float(qkv[vbase + c]);
    }
    out[gid] = half(acc / max(denom, 1.0e-12f));
}

kernel void nhwtc3_to_bgra8(
    device const half *x [[buffer(0)]],
    texture2d_array<half, access::write> outTexture [[texture(0)]],
    constant TensorParams &p [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    if (gid.x >= p.w || gid.y >= p.h || gid.z >= p.t) return;
    uint b = 0;
    ulong base = idx5(b, gid.z, gid.y, gid.x, 0, p.t, p.h, p.w, p.c);
    float r = clamp((float(x[base + 0]) + 1.0f) * 0.5f, 0.0f, 1.0f);
    float g = clamp((float(x[base + 1]) + 1.0f) * 0.5f, 0.0f, 1.0f);
    float bl = clamp((float(x[base + 2]) + 1.0f) * 0.5f, 0.0f, 1.0f);
    outTexture.write(half4(half(r), half(g), half(bl), half(1.0)), gid.xy, gid.z);
}
