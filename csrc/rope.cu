#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cassert>
#include <cmath>
#include <cfloat>
#include <type_traits>

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#define CUDA_CHECK(cmd) do { \
    cudaError_t result = cmd; \
    if (result != cudaSuccess) { \
        printf("[ERROR] CUDA error %s:%d '%s': (%d) %s\n", __FILE__, __LINE__, #cmd, (int)result, cudaGetErrorString(result)); \
    } \
} while(0)

// =============================================================================
// Helper: convert vector of T to float4 (from xcompute m_rope)
// =============================================================================
template<typename T>
__device__ __forceinline__ float4 to_float4(const T* ptr) {
    if constexpr (std::is_same_v<T, half>) {
        half2 h2_0 = reinterpret_cast<const half2*>(ptr)[0];
        half2 h2_1 = reinterpret_cast<const half2*>(ptr)[1];
        float2 f2_0 = __half22float2(h2_0);
        float2 f2_1 = __half22float2(h2_1);
        return make_float4(f2_0.x, f2_0.y, f2_1.x, f2_1.y);
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        __nv_bfloat162 b2_0 = reinterpret_cast<const __nv_bfloat162*>(ptr)[0];
        __nv_bfloat162 b2_1 = reinterpret_cast<const __nv_bfloat162*>(ptr)[1];
#if __CUDA_ARCH__ >= 800
        float2 f2_0 = __bfloat1622float2(b2_0);
        float2 f2_1 = __bfloat1622float2(b2_1);
        return make_float4(f2_0.x, f2_0.y, f2_1.x, f2_1.y);
#else
        printf("Unsupported on arch < 800");
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        return reinterpret_cast<const float4*>(ptr)[0];
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

template<typename T>
__device__ __forceinline__ void from_float4(T* ptr, const float4& f4) {
    if constexpr (std::is_same_v<T, half>) {
        half2 h2_0 = __float22half2_rn(make_float2(f4.x, f4.y));
        half2 h2_1 = __float22half2_rn(make_float2(f4.z, f4.w));
        reinterpret_cast<half2*>(ptr)[0] = h2_0;
        reinterpret_cast<half2*>(ptr)[1] = h2_1;
    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
        __nv_bfloat162 b2_0 = __float22bfloat162_rn(make_float2(f4.x, f4.y));
        __nv_bfloat162 b2_1 = __float22bfloat162_rn(make_float2(f4.z, f4.w));
        reinterpret_cast<__nv_bfloat162*>(ptr)[0] = b2_0;
        reinterpret_cast<__nv_bfloat162*>(ptr)[1] = b2_1;
#else
        printf("Unsupported on arch < 800");
#endif
    } else if constexpr (std::is_same_v<T, float>) {
        reinterpret_cast<float4*>(ptr)[0] = f4;
    } else {
        static_assert(sizeof(T) == 0, "Unsupported type");
    }
}

// =============================================================================
// MRopeKernel - Forward (from xcompute, adapted for PyTorch)
// q, k layout: [batch, seq_len, heads, head_dim]
// cos, sin layout: [3, batch, seq_len, head_dim/2]
// =============================================================================
template<class T>
__global__ void MRopeKernel(const T* cos, const T* sin,
                            const T* q, const int q_h, const T* k, const int k_h,
                            T* q_embed, T* k_embed, const int first, const int second,
                            const int d) {
    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    T* cos_smem = reinterpret_cast<T*>(cos_sin);
    T* sin_smem = cos_smem + half_dim;
    int b = blockIdx.x;
    int s = blockIdx.y;

    int64_t offset = gridDim.x * gridDim.y * half_dim;
    int64_t cos_sin_b_stride = gridDim.y * half_dim;
    int64_t cos_sin_s_stride = half_dim;
#define SIN_GMEM(a, b, c, d) sin[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM(a, b, c, d) cos[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        if (i < first) {
            cos_smem[i] = COS_GMEM(0, b, s, i);
            sin_smem[i] = SIN_GMEM(0, b, s, i);
        } else if (i < (second + first)) {
            cos_smem[i] = COS_GMEM(1, b, s, i);
            sin_smem[i] = SIN_GMEM(1, b, s, i);
        } else {
            cos_smem[i] = COS_GMEM(2, b, s, i);
            sin_smem[i] = SIN_GMEM(2, b, s, i);
        }
    }
    __syncthreads();

    int64_t q_b_offset = gridDim.y * q_h * d;
    int64_t q_s_offset = q_h * d;
    int64_t q_h_offset = d;
    int64_t k_b_offset = gridDim.y * k_h * d;
    int64_t k_s_offset = k_h * d;
    int64_t k_h_offset = d;
#define Q_GMEM(a, b, c, d) q[(a) * q_b_offset + (b) * q_s_offset + (c) * q_h_offset + d]
#define K_GMEM(a, b, c, d) k[(a) * k_b_offset + (b) * k_s_offset + (c) * k_h_offset + d]
#define Q_EMBED_GMEM(a, b, c, d) q_embed[(a) * q_b_offset + (b) * q_s_offset + (c) * q_h_offset + d]
#define K_EMBED_GMEM(a, b, c, d) k_embed[(a) * k_b_offset + (b) * k_s_offset + (c) * k_h_offset + d]

    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            __nv_bfloat162 q_x0 = reinterpret_cast<const __nv_bfloat162*>(&Q_GMEM(b, s, h_idx, base_d))[0];
            __nv_bfloat162 q_x1 = reinterpret_cast<const __nv_bfloat162*>(&Q_GMEM(b, s, h_idx, base_d))[1];
            __nv_bfloat162 q_x2 = reinterpret_cast<const __nv_bfloat162*>(&Q_GMEM(b, s, h_idx, base_d + half_dim))[0];
            __nv_bfloat162 q_x3 = reinterpret_cast<const __nv_bfloat162*>(&Q_GMEM(b, s, h_idx, base_d + half_dim))[1];

            __nv_bfloat162 cos_vec_0 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[0];
            __nv_bfloat162 cos_vec_1 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[1];
            __nv_bfloat162 sin_vec_0 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[0];
            __nv_bfloat162 sin_vec_1 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[1];
#if __CUDA_ARCH__ >= 800
            __nv_bfloat162 q0_rot = __nv_bfloat162(
                                    q_x0.x * cos_vec_0.x - q_x2.x * sin_vec_0.x,
                                    q_x0.y * cos_vec_0.y - q_x2.y * sin_vec_0.y
                                );
            __nv_bfloat162 q1_rot = __nv_bfloat162(
                                    q_x1.x * cos_vec_1.x - q_x3.x * sin_vec_1.x,
                                    q_x1.y * cos_vec_1.y - q_x3.y * sin_vec_1.y
                                );
            __nv_bfloat162 q2_rot = __nv_bfloat162(
                                    q_x2.x * cos_vec_0.x + q_x0.x * sin_vec_0.x,
                                    q_x2.y * cos_vec_0.y + q_x0.y * sin_vec_0.y
                                );
            __nv_bfloat162 q3_rot = __nv_bfloat162(
                                    q_x3.x * cos_vec_1.x + q_x1.x * sin_vec_1.x,
                                    q_x3.y * cos_vec_1.y + q_x1.y * sin_vec_1.y
                                );
            reinterpret_cast<__nv_bfloat162*>(&Q_EMBED_GMEM(b, s, h_idx, base_d))[0] = q0_rot;
            reinterpret_cast<__nv_bfloat162*>(&Q_EMBED_GMEM(b, s, h_idx, base_d))[1] = q1_rot;
            reinterpret_cast<__nv_bfloat162*>(&Q_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[0] = q2_rot;
            reinterpret_cast<__nv_bfloat162*>(&Q_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[1] = q3_rot;
#endif
        } else if constexpr (std::is_same_v<T, __half>) {
            __half2 q_x0 = reinterpret_cast<const __half2*>(&Q_GMEM(b, s, h_idx, base_d))[0];
            __half2 q_x1 = reinterpret_cast<const __half2*>(&Q_GMEM(b, s, h_idx, base_d))[1];
            __half2 q_x2 = reinterpret_cast<const __half2*>(&Q_GMEM(b, s, h_idx, base_d + half_dim))[0];
            __half2 q_x3 = reinterpret_cast<const __half2*>(&Q_GMEM(b, s, h_idx, base_d + half_dim))[1];

            __half2 cos_vec_0 = reinterpret_cast<__half2*>(&cos_smem[base_d])[0];
            __half2 cos_vec_1 = reinterpret_cast<__half2*>(&cos_smem[base_d])[1];
            __half2 sin_vec_0 = reinterpret_cast<__half2*>(&sin_smem[base_d])[0];
            __half2 sin_vec_1 = reinterpret_cast<__half2*>(&sin_smem[base_d])[1];

            __half2 q0_rot = __halves2half2(
                q_x0.x * cos_vec_0.x - q_x2.x * sin_vec_0.x,
                q_x0.y * cos_vec_0.y - q_x2.y * sin_vec_0.y
            );
            __half2 q1_rot = __halves2half2(
                q_x1.x * cos_vec_1.x - q_x3.x * sin_vec_1.x,
                q_x1.y * cos_vec_1.y - q_x3.y * sin_vec_1.y
            );
            __half2 q2_rot = __halves2half2(
                q_x2.x * cos_vec_0.x + q_x0.x * sin_vec_0.x,
                q_x2.y * cos_vec_0.y + q_x0.y * sin_vec_0.y
            );
            __half2 q3_rot = __halves2half2(
                q_x3.x * cos_vec_1.x + q_x1.x * sin_vec_1.x,
                q_x3.y * cos_vec_1.y + q_x1.y * sin_vec_1.y
            );

            reinterpret_cast<__half2*>(&Q_EMBED_GMEM(b, s, h_idx, base_d))[0] = q0_rot;
            reinterpret_cast<__half2*>(&Q_EMBED_GMEM(b, s, h_idx, base_d))[1] = q1_rot;
            reinterpret_cast<__half2*>(&Q_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[0] = q2_rot;
            reinterpret_cast<__half2*>(&Q_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[1] = q3_rot;
        } else {
            float4 q_x0 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d));
            float4 q_x1 = to_float4<T>(&Q_GMEM(b, s, h_idx, base_d + half_dim));

            float4 cos_vec = to_float4<T>(&cos_smem[base_d]);
            float4 sin_vec = to_float4<T>(&sin_smem[base_d]);

            float4 q0_rot = make_float4(
                q_x0.x * cos_vec.x - q_x1.x * sin_vec.x,
                q_x0.y * cos_vec.y - q_x1.y * sin_vec.y,
                q_x0.z * cos_vec.z - q_x1.z * sin_vec.z,
                q_x0.w * cos_vec.w - q_x1.w * sin_vec.w
            );

            float4 q1_rot = make_float4(
                q_x1.x * cos_vec.x + q_x0.x * sin_vec.x,
                q_x1.y * cos_vec.y + q_x0.y * sin_vec.y,
                q_x1.z * cos_vec.z + q_x0.z * sin_vec.z,
                q_x1.w * cos_vec.w + q_x0.w * sin_vec.w
            );

            from_float4<T>(&Q_EMBED_GMEM(b, s, h_idx, base_d), q0_rot);
            from_float4<T>(&Q_EMBED_GMEM(b, s, h_idx, base_d + half_dim), q1_rot);
        }
    }

    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            __nv_bfloat162 k_x0 = reinterpret_cast<const __nv_bfloat162*>(&K_GMEM(b, s, h_idx, base_d))[0];
            __nv_bfloat162 k_x1 = reinterpret_cast<const __nv_bfloat162*>(&K_GMEM(b, s, h_idx, base_d))[1];
            __nv_bfloat162 k_x2 = reinterpret_cast<const __nv_bfloat162*>(&K_GMEM(b, s, h_idx, base_d + half_dim))[0];
            __nv_bfloat162 k_x3 = reinterpret_cast<const __nv_bfloat162*>(&K_GMEM(b, s, h_idx, base_d + half_dim))[1];

            __nv_bfloat162 cos_vec_0 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[0];
            __nv_bfloat162 cos_vec_1 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[1];
            __nv_bfloat162 sin_vec_0 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[0];
            __nv_bfloat162 sin_vec_1 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[1];
#if __CUDA_ARCH__ >= 800
            __nv_bfloat162 k0_rot = __nv_bfloat162(
                                    k_x0.x * cos_vec_0.x - k_x2.x * sin_vec_0.x,
                                    k_x0.y * cos_vec_0.y - k_x2.y * sin_vec_0.y
                                );
            __nv_bfloat162 k1_rot = __nv_bfloat162(
                                    k_x1.x * cos_vec_1.x - k_x3.x * sin_vec_1.x,
                                    k_x1.y * cos_vec_1.y - k_x3.y * sin_vec_1.y
                                );
            __nv_bfloat162 k2_rot = __nv_bfloat162(
                                    k_x2.x * cos_vec_0.x + k_x0.x * sin_vec_0.x,
                                    k_x2.y * cos_vec_0.y + k_x0.y * sin_vec_0.y
                                );
            __nv_bfloat162 k3_rot = __nv_bfloat162(
                                    k_x3.x * cos_vec_1.x + k_x1.x * sin_vec_1.x,
                                    k_x3.y * cos_vec_1.y + k_x1.y * sin_vec_1.y
                                );
            reinterpret_cast<__nv_bfloat162*>(&K_EMBED_GMEM(b, s, h_idx, base_d))[0] = k0_rot;
            reinterpret_cast<__nv_bfloat162*>(&K_EMBED_GMEM(b, s, h_idx, base_d))[1] = k1_rot;
            reinterpret_cast<__nv_bfloat162*>(&K_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[0] = k2_rot;
            reinterpret_cast<__nv_bfloat162*>(&K_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[1] = k3_rot;
#endif
        } else if constexpr (std::is_same_v<T, __half>) {
            __half2 k_x0 = reinterpret_cast<const __half2*>(&K_GMEM(b, s, h_idx, base_d))[0];
            __half2 k_x1 = reinterpret_cast<const __half2*>(&K_GMEM(b, s, h_idx, base_d))[1];
            __half2 k_x2 = reinterpret_cast<const __half2*>(&K_GMEM(b, s, h_idx, base_d + half_dim))[0];
            __half2 k_x3 = reinterpret_cast<const __half2*>(&K_GMEM(b, s, h_idx, base_d + half_dim))[1];

            __half2 cos_vec_0 = reinterpret_cast<__half2*>(&cos_smem[base_d])[0];
            __half2 cos_vec_1 = reinterpret_cast<__half2*>(&cos_smem[base_d])[1];
            __half2 sin_vec_0 = reinterpret_cast<__half2*>(&sin_smem[base_d])[0];
            __half2 sin_vec_1 = reinterpret_cast<__half2*>(&sin_smem[base_d])[1];

            __half2 k0_rot = __halves2half2(
                k_x0.x * cos_vec_0.x - k_x2.x * sin_vec_0.x,
                k_x0.y * cos_vec_0.y - k_x2.y * sin_vec_0.y
            );
            __half2 k1_rot = __halves2half2(
                k_x1.x * cos_vec_1.x - k_x3.x * sin_vec_1.x,
                k_x1.y * cos_vec_1.y - k_x3.y * sin_vec_1.y
            );
            __half2 k2_rot = __halves2half2(
                k_x2.x * cos_vec_0.x + k_x0.x * sin_vec_0.x,
                k_x2.y * cos_vec_0.y + k_x0.y * sin_vec_0.y
            );
            __half2 k3_rot = __halves2half2(
                k_x3.x * cos_vec_1.x + k_x1.x * sin_vec_1.x,
                k_x3.y * cos_vec_1.y + k_x1.y * sin_vec_1.y
            );

            reinterpret_cast<__half2*>(&K_EMBED_GMEM(b, s, h_idx, base_d))[0] = k0_rot;
            reinterpret_cast<__half2*>(&K_EMBED_GMEM(b, s, h_idx, base_d))[1] = k1_rot;
            reinterpret_cast<__half2*>(&K_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[0] = k2_rot;
            reinterpret_cast<__half2*>(&K_EMBED_GMEM(b, s, h_idx, base_d + half_dim))[1] = k3_rot;
        } else {
            float4 k_x0 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d));
            float4 k_x1 = to_float4<T>(&K_GMEM(b, s, h_idx, base_d + half_dim));

            float4 cos_vec = to_float4<T>(&cos_smem[base_d]);
            float4 sin_vec = to_float4<T>(&sin_smem[base_d]);

            float4 k0_rot = make_float4(
                k_x0.x * cos_vec.x - k_x1.x * sin_vec.x,
                k_x0.y * cos_vec.y - k_x1.y * sin_vec.y,
                k_x0.z * cos_vec.z - k_x1.z * sin_vec.z,
                k_x0.w * cos_vec.w - k_x1.w * sin_vec.w
            );

            float4 k1_rot = make_float4(
                k_x1.x * cos_vec.x + k_x0.x * sin_vec.x,
                k_x1.y * cos_vec.y + k_x0.y * sin_vec.y,
                k_x1.z * cos_vec.z + k_x0.z * sin_vec.z,
                k_x1.w * cos_vec.w + k_x0.w * sin_vec.w
            );

            from_float4<T>(&K_EMBED_GMEM(b, s, h_idx, base_d), k0_rot);
            from_float4<T>(&K_EMBED_GMEM(b, s, h_idx, base_d + half_dim), k1_rot);
        }
    }
}

// =============================================================================
// MRopeKernelBackward (from xcompute)
// =============================================================================
template<class T>
__global__ void MRopeKernelBackward(
    const T* cos,
    const T* sin,
    const T* grad_q_embed,
    const T* grad_k_embed,
    T* grad_q,
    T* grad_k,
    const int first, const int second,
    const int q_h,
    const int k_h,
    const int d) {

    extern __shared__ char cos_sin[];
    const int half_dim = d / 2;
    T* cos_smem = reinterpret_cast<T*>(cos_sin);
    T* sin_smem = cos_smem + half_dim;

    int b = blockIdx.x;
    int s = blockIdx.y;

    int64_t offset = gridDim.x * gridDim.y * half_dim;
    int64_t cos_sin_b_stride = gridDim.y * half_dim;
    int64_t cos_sin_s_stride = half_dim;

#define SIN_GMEM_BWD(a, b, c, d) sin[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]
#define COS_GMEM_BWD(a, b, c, d) cos[(a) * offset + (b) * cos_sin_b_stride + (c) * cos_sin_s_stride + (d)]

    for (int i = threadIdx.x; i < half_dim; i += blockDim.x) {
        if (i < first) {
            cos_smem[i] = COS_GMEM_BWD(0, b, s, i);
            sin_smem[i] = SIN_GMEM_BWD(0, b, s, i);
        } else if (i < (second + first)) {
            cos_smem[i] = COS_GMEM_BWD(1, b, s, i);
            sin_smem[i] = SIN_GMEM_BWD(1, b, s, i);
        } else {
            cos_smem[i] = COS_GMEM_BWD(2, b, s, i);
            sin_smem[i] = SIN_GMEM_BWD(2, b, s, i);
        }
    }
    __syncthreads();

    int64_t grad_q_b_offset = gridDim.y * q_h * d;
    int64_t grad_q_s_offset = q_h * d;
    int64_t grad_q_h_offset = d;
    int64_t grad_k_b_offset = gridDim.y * k_h * d;
    int64_t grad_k_s_offset = k_h * d;
    int64_t grad_k_h_offset = d;

#define GQ_EMBED(a, b, c, d) grad_q_embed[(a) * grad_q_b_offset + (b) * grad_q_s_offset + (c) * grad_q_h_offset + (d)]
#define GK_EMBED(a, b, c, d) grad_k_embed[(a) * grad_k_b_offset + (b) * grad_k_s_offset + (c) * grad_k_h_offset + (d)]
#define GQ(a, b, c, d) grad_q[(a) * grad_q_b_offset + (b) * grad_q_s_offset + (c) * grad_q_h_offset + (d)]
#define GK(a, b, c, d) grad_k[(a) * grad_k_b_offset + (b) * grad_k_s_offset + (c) * grad_k_h_offset + (d)]

    for (int j = threadIdx.x * 4; j < q_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
            __nv_bfloat162 gq0_rot_0 = reinterpret_cast<const __nv_bfloat162*>(&GQ_EMBED(b, s, h_idx, base_d))[0];
            __nv_bfloat162 gq0_rot_1 = reinterpret_cast<const __nv_bfloat162*>(&GQ_EMBED(b, s, h_idx, base_d))[1];
            __nv_bfloat162 gq1_rot_0 = reinterpret_cast<const __nv_bfloat162*>(&GQ_EMBED(b, s, h_idx, base_d + half_dim))[0];
            __nv_bfloat162 gq1_rot_1 = reinterpret_cast<const __nv_bfloat162*>(&GQ_EMBED(b, s, h_idx, base_d + half_dim))[1];

            __nv_bfloat162 cos_0 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[0];
            __nv_bfloat162 cos_1 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[1];
            __nv_bfloat162 sin_0 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[0];
            __nv_bfloat162 sin_1 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[1];

            __nv_bfloat162 gq0_0 = __nv_bfloat162(
                gq0_rot_0.x * cos_0.x + gq1_rot_0.x * sin_0.x,
                gq0_rot_0.y * cos_0.y + gq1_rot_0.y * sin_0.y
            );
            __nv_bfloat162 gq0_1 = __nv_bfloat162(
                gq0_rot_1.x * cos_1.x + gq1_rot_1.x * sin_1.x,
                gq0_rot_1.y * cos_1.y + gq1_rot_1.y * sin_1.y
            );

            __nv_bfloat162 gq1_0 = __nv_bfloat162(
                gq1_rot_0.x * cos_0.x - gq0_rot_0.x * sin_0.x,
                gq1_rot_0.y * cos_0.y - gq0_rot_0.y * sin_0.y
            );
            __nv_bfloat162 gq1_1 = __nv_bfloat162(
                gq1_rot_1.x * cos_1.x - gq0_rot_1.x * sin_1.x,
                gq1_rot_1.y * cos_1.y - gq0_rot_1.y * sin_1.y
            );

            reinterpret_cast<__nv_bfloat162*>(&GQ(b, s, h_idx, base_d))[0] = gq0_0;
            reinterpret_cast<__nv_bfloat162*>(&GQ(b, s, h_idx, base_d))[1] = gq0_1;
            reinterpret_cast<__nv_bfloat162*>(&GQ(b, s, h_idx, base_d + half_dim))[0] = gq1_0;
            reinterpret_cast<__nv_bfloat162*>(&GQ(b, s, h_idx, base_d + half_dim))[1] = gq1_1;
#endif
        } else if constexpr (std::is_same_v<T, __half>) {
            __half2 gq0_rot_0 = reinterpret_cast<const __half2*>(&GQ_EMBED(b, s, h_idx, base_d))[0];
            __half2 gq0_rot_1 = reinterpret_cast<const __half2*>(&GQ_EMBED(b, s, h_idx, base_d))[1];
            __half2 gq1_rot_0 = reinterpret_cast<const __half2*>(&GQ_EMBED(b, s, h_idx, base_d + half_dim))[0];
            __half2 gq1_rot_1 = reinterpret_cast<const __half2*>(&GQ_EMBED(b, s, h_idx, base_d + half_dim))[1];

            __half2 cos_0 = reinterpret_cast<__half2*>(&cos_smem[base_d])[0];
            __half2 cos_1 = reinterpret_cast<__half2*>(&cos_smem[base_d])[1];
            __half2 sin_0 = reinterpret_cast<__half2*>(&sin_smem[base_d])[0];
            __half2 sin_1 = reinterpret_cast<__half2*>(&sin_smem[base_d])[1];

            __half2 gq0_0 = __halves2half2(
                gq0_rot_0.x * cos_0.x + gq1_rot_0.x * sin_0.x,
                gq0_rot_0.y * cos_0.y + gq1_rot_0.y * sin_0.y
            );
            __half2 gq0_1 = __halves2half2(
                gq0_rot_1.x * cos_1.x + gq1_rot_1.x * sin_1.x,
                gq0_rot_1.y * cos_1.y + gq1_rot_1.y * sin_1.y
            );

            __half2 gq1_0 = __halves2half2(
                gq1_rot_0.x * cos_0.x - gq0_rot_0.x * sin_0.x,
                gq1_rot_0.y * cos_0.y - gq0_rot_0.y * sin_0.y
            );
            __half2 gq1_1 = __halves2half2(
                gq1_rot_1.x * cos_1.x - gq0_rot_1.x * sin_1.x,
                gq1_rot_1.y * cos_1.y - gq0_rot_1.y * sin_1.y
            );

            reinterpret_cast<__half2*>(&GQ(b, s, h_idx, base_d))[0] = gq0_0;
            reinterpret_cast<__half2*>(&GQ(b, s, h_idx, base_d))[1] = gq0_1;
            reinterpret_cast<__half2*>(&GQ(b, s, h_idx, base_d + half_dim))[0] = gq1_0;
            reinterpret_cast<__half2*>(&GQ(b, s, h_idx, base_d + half_dim))[1] = gq1_1;
        } else {
            float4 gq0_rot = to_float4<T>(&GQ_EMBED(b, s, h_idx, base_d));
            float4 gq1_rot = to_float4<T>(&GQ_EMBED(b, s, h_idx, base_d + half_dim));

            float4 cos_vec = to_float4<T>(&cos_smem[base_d]);
            float4 sin_vec = to_float4<T>(&sin_smem[base_d]);

            float4 gq0 = make_float4(
                gq0_rot.x * cos_vec.x + gq1_rot.x * sin_vec.x,
                gq0_rot.y * cos_vec.y + gq1_rot.y * sin_vec.y,
                gq0_rot.z * cos_vec.z + gq1_rot.z * sin_vec.z,
                gq0_rot.w * cos_vec.w + gq1_rot.w * sin_vec.w
            );
            float4 gq1 = make_float4(
                gq1_rot.x * cos_vec.x - gq0_rot.x * sin_vec.x,
                gq1_rot.y * cos_vec.y - gq0_rot.y * sin_vec.y,
                gq1_rot.z * cos_vec.z - gq0_rot.z * sin_vec.z,
                gq1_rot.w * cos_vec.w - gq0_rot.w * sin_vec.w
            );

            from_float4<T>(&GQ(b, s, h_idx, base_d), gq0);
            from_float4<T>(&GQ(b, s, h_idx, base_d + half_dim), gq1);
        }
    }

    for (int j = threadIdx.x * 4; j < k_h * half_dim; j += blockDim.x * 4) {
        int h_idx = j / half_dim;
        int base_d = j % half_dim;
        if (base_d + 3 >= half_dim) continue;

        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
#if __CUDA_ARCH__ >= 800
            __nv_bfloat162 gk0_rot_0 = reinterpret_cast<const __nv_bfloat162*>(&GK_EMBED(b, s, h_idx, base_d))[0];
            __nv_bfloat162 gk0_rot_1 = reinterpret_cast<const __nv_bfloat162*>(&GK_EMBED(b, s, h_idx, base_d))[1];
            __nv_bfloat162 gk1_rot_0 = reinterpret_cast<const __nv_bfloat162*>(&GK_EMBED(b, s, h_idx, base_d + half_dim))[0];
            __nv_bfloat162 gk1_rot_1 = reinterpret_cast<const __nv_bfloat162*>(&GK_EMBED(b, s, h_idx, base_d + half_dim))[1];

            __nv_bfloat162 cos_0 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[0];
            __nv_bfloat162 cos_1 = reinterpret_cast<__nv_bfloat162*>(&cos_smem[base_d])[1];
            __nv_bfloat162 sin_0 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[0];
            __nv_bfloat162 sin_1 = reinterpret_cast<__nv_bfloat162*>(&sin_smem[base_d])[1];

            __nv_bfloat162 gk0_0 = __nv_bfloat162(
                gk0_rot_0.x * cos_0.x + gk1_rot_0.x * sin_0.x,
                gk0_rot_0.y * cos_0.y + gk1_rot_0.y * sin_0.y
            );
            __nv_bfloat162 gk0_1 = __nv_bfloat162(
                gk0_rot_1.x * cos_1.x + gk1_rot_1.x * sin_1.x,
                gk0_rot_1.y * cos_1.y + gk1_rot_1.y * sin_1.y
            );

            __nv_bfloat162 gk1_0 = __nv_bfloat162(
                gk1_rot_0.x * cos_0.x - gk0_rot_0.x * sin_0.x,
                gk1_rot_0.y * cos_0.y - gk0_rot_0.y * sin_0.y
            );
            __nv_bfloat162 gk1_1 = __nv_bfloat162(
                gk1_rot_1.x * cos_1.x - gk0_rot_1.x * sin_1.x,
                gk1_rot_1.y * cos_1.y - gk0_rot_1.y * sin_1.y
            );

            reinterpret_cast<__nv_bfloat162*>(&GK(b, s, h_idx, base_d))[0] = gk0_0;
            reinterpret_cast<__nv_bfloat162*>(&GK(b, s, h_idx, base_d))[1] = gk0_1;
            reinterpret_cast<__nv_bfloat162*>(&GK(b, s, h_idx, base_d + half_dim))[0] = gk1_0;
            reinterpret_cast<__nv_bfloat162*>(&GK(b, s, h_idx, base_d + half_dim))[1] = gk1_1;
#endif
        } else if constexpr (std::is_same_v<T, __half>) {
            __half2 gk0_rot_0 = reinterpret_cast<const __half2*>(&GK_EMBED(b, s, h_idx, base_d))[0];
            __half2 gk0_rot_1 = reinterpret_cast<const __half2*>(&GK_EMBED(b, s, h_idx, base_d))[1];
            __half2 gk1_rot_0 = reinterpret_cast<const __half2*>(&GK_EMBED(b, s, h_idx, base_d + half_dim))[0];
            __half2 gk1_rot_1 = reinterpret_cast<const __half2*>(&GK_EMBED(b, s, h_idx, base_d + half_dim))[1];

            __half2 cos_0 = reinterpret_cast<__half2*>(&cos_smem[base_d])[0];
            __half2 cos_1 = reinterpret_cast<__half2*>(&cos_smem[base_d])[1];
            __half2 sin_0 = reinterpret_cast<__half2*>(&sin_smem[base_d])[0];
            __half2 sin_1 = reinterpret_cast<__half2*>(&sin_smem[base_d])[1];

            __half2 gk0_0 = __halves2half2(
                __hadd(__hmul(gk0_rot_0.x, cos_0.x), __hmul(gk1_rot_0.x, sin_0.x)),
                __hadd(__hmul(gk0_rot_0.y, cos_0.y), __hmul(gk1_rot_0.y, sin_0.y))
            );
            __half2 gk0_1 = __halves2half2(
                __hadd(__hmul(gk0_rot_1.x, cos_1.x), __hmul(gk1_rot_1.x, sin_1.x)),
                __hadd(__hmul(gk0_rot_1.y, cos_1.y), __hmul(gk1_rot_1.y, sin_1.y))
            );

            __half2 gk1_0 = __halves2half2(
                __hsub(__hmul(gk1_rot_0.x, cos_0.x), __hmul(gk0_rot_0.x, sin_0.x)),
                __hsub(__hmul(gk1_rot_0.y, cos_0.y), __hmul(gk0_rot_0.y, sin_0.y))
            );
            __half2 gk1_1 = __halves2half2(
                __hsub(__hmul(gk1_rot_1.x, cos_1.x), __hmul(gk0_rot_1.x, sin_1.x)),
                __hsub(__hmul(gk1_rot_1.y, cos_1.y), __hmul(gk0_rot_1.y, sin_1.y))
            );

            reinterpret_cast<__half2*>(&GK(b, s, h_idx, base_d))[0] = gk0_0;
            reinterpret_cast<__half2*>(&GK(b, s, h_idx, base_d))[1] = gk0_1;
            reinterpret_cast<__half2*>(&GK(b, s, h_idx, base_d + half_dim))[0] = gk1_0;
            reinterpret_cast<__half2*>(&GK(b, s, h_idx, base_d + half_dim))[1] = gk1_1;
        } else {
            float4 gk0_rot = to_float4<T>(&GK_EMBED(b, s, h_idx, base_d));
            float4 gk1_rot = to_float4<T>(&GK_EMBED(b, s, h_idx, base_d + half_dim));

            float4 cos_vec = to_float4<T>(&cos_smem[base_d]);
            float4 sin_vec = to_float4<T>(&sin_smem[base_d]);

            float4 gk0 = make_float4(
                gk0_rot.x * cos_vec.x + gk1_rot.x * sin_vec.x,
                gk0_rot.y * cos_vec.y + gk1_rot.y * sin_vec.y,
                gk0_rot.z * cos_vec.z + gk1_rot.z * sin_vec.z,
                gk0_rot.w * cos_vec.w + gk1_rot.w * sin_vec.w
            );
            float4 gk1 = make_float4(
                gk1_rot.x * cos_vec.x - gk0_rot.x * sin_vec.x,
                gk1_rot.y * cos_vec.y - gk0_rot.y * sin_vec.y,
                gk1_rot.z * cos_vec.z - gk0_rot.z * sin_vec.z,
                gk1_rot.w * cos_vec.w - gk0_rot.w * sin_vec.w
            );

            from_float4<T>(&GK(b, s, h_idx, base_d), gk0);
            from_float4<T>(&GK(b, s, h_idx, base_d + half_dim), gk1);
        }
    }
}

// =============================================================================
// Converts mrope_section_doubled -> first, second
// Supports [batch, seq_len, heads, dim] layout (from model)
// =============================================================================
template <typename T>
void launch_mrope_forward_impl(
    const T* q, const T* k, const T* cos, const T* sin,
    T* q_out, T* k_out,
    int first, int second,
    int batch_size, int seq_len, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream)
{
    constexpr int Nthreads = 256;
    dim3 grid(batch_size, seq_len);
    size_t smem_size = head_dim * sizeof(T);

    MRopeKernel<T><<<grid, Nthreads, smem_size, stream>>>(
        const_cast<T*>(cos), const_cast<T*>(sin),
        q, q_heads, k, kv_heads,
        q_out, k_out,
        first, second, head_dim);
}

template <typename T>
void launch_mrope_backward_impl(
    const T* grad_q_out, const T* grad_k_out,
    const T* cos, const T* sin,
    T* grad_q, T* grad_k,
    int first, int second,
    int batch_size, int seq_len, int q_heads, int kv_heads, int head_dim,
    cudaStream_t stream)
{
    constexpr int Nthreads = 256;
    dim3 grid(batch_size, seq_len);
    size_t smem_size = head_dim * sizeof(T);

    MRopeKernelBackward<T><<<grid, Nthreads, smem_size, stream>>>(
        cos, sin,
        grad_q_out, grad_k_out,
        grad_q, grad_k,
        first, second,
        q_heads, kv_heads, head_dim);
}

void launch_multimodal_rope_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor q_out, torch::Tensor k_out,
    std::vector<int> mrope_section_doubled)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Scheme A: convert mrope_section_doubled to first, second (half_dim values)
    int first = mrope_section_doubled[0] / 2;
    int second = mrope_section_doubled[1] / 2;

    // Support [batch, seq_len, heads, head_dim] (from model) - xcompute layout
    int batch_size = q.size(0);
    int seq_len = q.size(1);
    int q_heads = q.size(2);
    int head_dim = q.size(3);
    int kv_heads = k.size(2);

    if (q.scalar_type() == torch::kFloat32) {
        launch_mrope_forward_impl<float>(
            static_cast<const float*>(q.data_ptr()),
            static_cast<const float*>(k.data_ptr()),
            static_cast<const float*>(cos.data_ptr()),
            static_cast<const float*>(sin.data_ptr()),
            static_cast<float*>(q_out.data_ptr()),
            static_cast<float*>(k_out.data_ptr()),
            first, second,
            batch_size, seq_len, q_heads, kv_heads, head_dim,
            stream);
    } else if (q.scalar_type() == torch::kFloat16) {
        launch_mrope_forward_impl<half>(
            static_cast<const half*>(q.data_ptr()),
            static_cast<const half*>(k.data_ptr()),
            static_cast<const half*>(cos.data_ptr()),
            static_cast<const half*>(sin.data_ptr()),
            static_cast<half*>(q_out.data_ptr()),
            static_cast<half*>(k_out.data_ptr()),
            first, second,
            batch_size, seq_len, q_heads, kv_heads, head_dim,
            stream);
    } else if (q.scalar_type() == torch::kBFloat16) {
        launch_mrope_forward_impl<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(q.data_ptr()),
            static_cast<const __nv_bfloat16*>(k.data_ptr()),
            static_cast<const __nv_bfloat16*>(cos.data_ptr()),
            static_cast<const __nv_bfloat16*>(sin.data_ptr()),
            static_cast<__nv_bfloat16*>(q_out.data_ptr()),
            static_cast<__nv_bfloat16*>(k_out.data_ptr()),
            first, second,
            batch_size, seq_len, q_heads, kv_heads, head_dim,
            stream);
    } else {
        throw std::runtime_error("multimodal_rope: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}

void launch_multimodal_rope_backward(
    torch::Tensor grad_q_out, torch::Tensor grad_k_out,
    torch::Tensor q, torch::Tensor k, torch::Tensor cos, torch::Tensor sin,
    torch::Tensor grad_q, torch::Tensor grad_k,
    std::vector<int> mrope_section_doubled)
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int first = mrope_section_doubled[0] / 2;
    int second = mrope_section_doubled[1] / 2;

    int batch_size = grad_q_out.size(0);
    int seq_len = grad_q_out.size(1);
    int q_heads = grad_q.size(2);
    int head_dim = grad_q_out.size(3);
    int kv_heads = grad_k.size(2);

    if (grad_q_out.scalar_type() == torch::kFloat32) {
        launch_mrope_backward_impl<float>(
            static_cast<const float*>(grad_q_out.data_ptr()),
            static_cast<const float*>(grad_k_out.data_ptr()),
            static_cast<const float*>(cos.data_ptr()),
            static_cast<const float*>(sin.data_ptr()),
            static_cast<float*>(grad_q.data_ptr()),
            static_cast<float*>(grad_k.data_ptr()),
            first, second,
            batch_size, seq_len, q_heads, kv_heads, head_dim,
            stream);
    } else if (grad_q_out.scalar_type() == torch::kFloat16) {
        launch_mrope_backward_impl<half>(
            static_cast<const half*>(grad_q_out.data_ptr()),
            static_cast<const half*>(grad_k_out.data_ptr()),
            static_cast<const half*>(cos.data_ptr()),
            static_cast<const half*>(sin.data_ptr()),
            static_cast<half*>(grad_q.data_ptr()),
            static_cast<half*>(grad_k.data_ptr()),
            first, second,
            batch_size, seq_len, q_heads, kv_heads, head_dim,
            stream);
    } else if (grad_q_out.scalar_type() == torch::kBFloat16) {
        launch_mrope_backward_impl<__nv_bfloat16>(
            static_cast<const __nv_bfloat16*>(grad_q_out.data_ptr()),
            static_cast<const __nv_bfloat16*>(grad_k_out.data_ptr()),
            static_cast<const __nv_bfloat16*>(cos.data_ptr()),
            static_cast<const __nv_bfloat16*>(sin.data_ptr()),
            static_cast<__nv_bfloat16*>(grad_q.data_ptr()),
            static_cast<__nv_bfloat16*>(grad_k.data_ptr()),
            first, second,
            batch_size, seq_len, q_heads, kv_heads, head_dim,
            stream);
    } else {
        throw std::runtime_error("multimodal_rope_bwd: unsupported dtype");
    }

    CUDA_CHECK(cudaGetLastError());
}
