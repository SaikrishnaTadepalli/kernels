#include <cuda_runtime.h>
#include <cfloat>


namespace map_reduce_lib {

    constexpr int BLOCK_SIZE = 1024;
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS_PER_BLOCK = ( BLOCK_SIZE / WARP_SIZE );

    // ==================================================
    // Map Library
    // ==================================================
    namespace map_lib {
        // Kernel for performing the map operation on the GPU.
        template <typename T1, typename T2, typename Op>
        __global__ void map_kernel(
            const T1* input,
            T2* output,
            Op op,
            int N
        ) {
            int global_index = blockIdx.x * blockDim.x + threadIdx.x;
            if (global_index < N) {
                T1 input_val = input[global_index];
                T2 output_val = op(input_val);
                output[global_index] = output_val;
            }
        }

        // Map Class encapsulating common map operations and the main runner for map execution.
        class Map {
        public:
            // --------------------------------------------------
            // Common Map Operators
            // --------------------------------------------------
            
            /*
                Scales the input by a given factor.
            */
            struct Scale {
                float factor;

                __host__ __device__ explicit 
                Scale(float f) : factor(f) {}

                __device__
                float operator()(float x) const {
                    return x * factor;
                }
            };

            /*
                Negates the input.
            */
            struct Negate {
                __device__
                float operator()(float x) const {
                    return -x;
                }
            };
            
            /*
                Computes the square of the input.
            */
            struct Square {
                __device__
                float operator()(float x) const {
                    return x * x;
                }
            };

            /*
                Clamps the input to a specified range.
            */
            struct Clamp {
                float min_val;
                float max_val;

                __host__ __device__ explicit
                Clamp(float min_v, float max_v) : min_val(min_v), max_val(max_v) {}

                __device__
                float operator()(float x) const {
                    return fminf(fmaxf(x, min_val), max_val);
                }
            };

            /*
                Computes the reciprocal of the input.
            */
            struct Reciprocal {
                __device__
                float operator()(float x) const {
                    return 1.0f / x;
                }
            };

            /*
                GEQ (greater-than-or-equal-to) comparison of 
                the input with a specified threshold.

                Returns 1.0f if the input is >= threshold, otherwise 0.0f.
            */
            struct GEQ {
                float threshold;

                __host__ __device__ explicit
                GEQ(float t) : threshold(t) {}

                __device__
                float operator()(float x) const {
                    return x >= threshold ? 1.0f : 0.0f;
                }
            };

            /*
                LEQ (less-than-or-equal-to) comparison of 
                the input with a specified threshold.

                Returns 1.0f if the input is <= threshold, otherwise 0.0f.
            */
            struct LEQ {
                float threshold;

                __host__ __device__ explicit
                LEQ(float t) : threshold(t) {}

                __device__
                float operator()(float x) const {
                    return x <= threshold ? 1.0f : 0.0f;
                }
            };

            // --------------------------------------------------
            // Main Runner for Map Operation
            // --------------------------------------------------

            /*
                Maps items in 'input' to 'output' using the specified 
                operation 'op'.

                Requires:
                - 'input' and 'output' must be device pointers
                - 'input' and 'output' must point to arrays of size 'N'
                - 'op' must be a callable object that can be invoked on 
                    elements of 'input' and returns a value assignable 
                    to elements of 'output'.
            */
            template<typename T1, typename T2, typename Op>
            static void map_handler(
                const T1* input,
                T2* output,
                int N,
                Op op
            ) {
                // --------------------------------------------------
                // Input Validation
                // --------------------------------------------------
                if (input == nullptr || output == nullptr || N <= 0) {
                    return; // Invalid input, return early
                }

                // --------------------------------------------------
                // Setup Kernel Dimensions
                // --------------------------------------------------
                dim3 threads_per_block(BLOCK_SIZE);
                dim3 blocks_per_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

                // --------------------------------------------------
                // Launch Kernel
                // --------------------------------------------------
                map_kernel<T1, T2, Op>
                    <<<blocks_per_grid, threads_per_block>>>
                    (input, output, op, N);
                
                cudaDeviceSynchronize();
            }
        }; // Map
    } // namespace map_lib




    // ==================================================
    // Reduce Library
    // ==================================================
    namespace reduce_lib {
        // Kernel for performing the reduce operation on the GPU.
        /*
            - This works best if BLOCK_SIZE is 1024. But, it can 
                be adjusted based on the specific GPU 
                architecture and problem size. 
                
            For safety, ensure that BLOCK_SIZE > 32. I am currently
            not sure what would happend with the shfl_down_sync's
            if we don't have all the threads in a warp participating.
        */
        template <typename T, typename Op>
        __global__ void reduce_kernel(
            const T* input,
            T* output,
            Op op,
            int N
        ) {
            // --------------------------------------------------
            // Get Indices
            // --------------------------------------------------
            int local_index = threadIdx.x;
            int global_index = blockIdx.x * blockDim.x + threadIdx.x;
            int warp_index = local_index / WARP_SIZE;
            int lane_index = local_index % WARP_SIZE;

            // --------------------------------------------------
            // Setup Shared Memory Region
            // --------------------------------------------------
            __shared__ T shared_memory[NUM_WARPS_PER_BLOCK];
            if (local_index < NUM_WARPS_PER_BLOCK) {
                shared_memory[local_index] = op.identity();
            }
            __syncthreads();

            // --------------------------------------------------
            // Perform Warp-Level Reduction
            // --------------------------------------------------
            T warp_result = (
                (global_index < N ? input[global_index] : op.identity())
            );

            for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)  {
                T neighbor_warp_result = __shfl_down_sync(
                    0xFFFFFFFF, warp_result, stride
                );
                warp_result = op(warp_result, neighbor_warp_result);
            }

            if (lane_index == 0) {
                shared_memory[warp_index] = warp_result;
            }

            __syncthreads();

            // --------------------------------------------------
            // Perform Block-Level Reduction using Warp-0 (first warp in the block)
            // --------------------------------------------------
            if (warp_index == 0) {
                warp_result = (
                    (lane_index < NUM_WARPS_PER_BLOCK) ? shared_memory[lane_index] : op.identity()
                );
                for (int stride = NUM_WARPS_PER_BLOCK / 2; stride > 0; stride /= 2)  {
                    T neighbor_warp_result = __shfl_down_sync(
                        0xFFFFFFFF, warp_result, stride
                    );
                    warp_result = op(warp_result, neighbor_warp_result);
                }
            }

            // --------------------------------------------------
            // Set Final Output for the Block
            // --------------------------------------------------
            if ((warp_index == 0) && (lane_index == 0)) {
                output[blockIdx.x] = warp_result;
            }
        }

        // Reduce Class encapsulating common reduce operations and the main runner for reduce execution.
        class Reduce {
        public:
            // --------------------------------------------------
            // Common Reduce Operators
            // --------------------------------------------------
            struct SumOp {
                __device__ float operator()(float a, float b) const { return a + b; }
                __device__ float identity() const { return 0.0f; }
            };

            struct ProdOp {
                __device__ float operator()(float a, float b) const { return a * b; }
                __device__ float identity() const { return 1.0f; }
            };

            struct MaxOp {
                __device__ float operator()(float a, float b) const { return fmaxf(a, b); }
                __device__ float identity() const { return -INFINITY; }
            };

            struct MinOp {
                __device__ float operator()(float a, float b) const { return fminf(a, b); }
                __device__ float identity() const { return INFINITY; }
            };

            // Note: bool reduces fine here too -- __shfl_down_sync is a
            // generic templated intrinsic on CUDA 9.0+, not limited to
            // the built-in numeric types.
            struct AndOp {
                __device__ bool operator()(bool a, bool b) const { return a && b; }
                __device__ bool identity() const { return true; }
            };

            struct OrOp {
                __device__ bool operator()(bool a, bool b) const { return a || b; }
                __device__ bool identity() const { return false; }
            };

            struct XorOp {
                __device__ bool operator()(bool a, bool b) const { return a != b; }
                __device__ bool identity() const { return false; }
            };
            

            // --------------------------------------------------
            // Main Runner for Reduce Operation
            // --------------------------------------------------

            /*
                Reduces items in 'input' using the specified 
                operation 'op'.

                Requires:
                - 'input' and 'output' must be device pointers
                - 'input' must be of size 'N'
                - 'output' must be of size '1'
                - 'op' must be a callable object that can be invoked on 
                    elements of 'input'.
            */
            template <typename T, typename Op>
            static void reduce_handler(
                const T* input,
                T* output,
                Op op,
                int N
            ) {
                 // --------------------------------------------------
                // Input Validation
                // --------------------------------------------------
                if (input == nullptr || output == nullptr || N <= 0) {
                    return; // Invalid input, return early
                }

                // --------------------------------------------------
                // Initialize Buffers for Reduction
                // --------------------------------------------------
                
                T* buffer1;
                size_t buffer1_size = N * sizeof(T);
                cudaMalloc(&buffer1, buffer1_size);
                cudaMemcpy(buffer1, input, buffer1_size, cudaMemcpyDeviceToDevice);

                T* buffer2;
                int buffer2_num_items = (N + BLOCK_SIZE - 1) / BLOCK_SIZE; // After first reduction step
                size_t buffer2_size = buffer2_num_items * sizeof(T);
                cudaMalloc(&buffer2, buffer2_size);

                T* buffers[2] = { buffer1, buffer2 };
                int cur_input_buffer = 0;
                int cur_output_buffer = 1;

                // --------------------------------------------------
                // Reduction Loop
                // --------------------------------------------------
                while (N > 1) {
                    // --------------------------------------------------
                    // Setup Kernel Dimensions for Current Reduction Step
                    // --------------------------------------------------
                    dim3 threads_per_block( BLOCK_SIZE );
                    dim3 blocks_per_grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

                    // --------------------------------------------------
                    // Launch Reduction Kernel for Current Step
                    // --------------------------------------------------
                    reduce_kernel<T, Op>
                        <<<blocks_per_grid, threads_per_block>>>
                        (buffers[cur_input_buffer], buffers[cur_output_buffer], op, N);
                    
                    cudaDeviceSynchronize();

                    // --------------------------------------------------
                    // Prepare for next iteration
                    // --------------------------------------------------
                    N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    cur_input_buffer = 1 - cur_input_buffer;
                    cur_output_buffer = 1 - cur_output_buffer;
                }

                // --------------------------------------------------
                // Copy Result Back to Output
                // --------------------------------------------------
                cudaMemcpy(output, buffers[cur_input_buffer], sizeof(T), cudaMemcpyDeviceToDevice);
                
                // --------------------------------------------------
                // Free Buffers
                // --------------------------------------------------
                cudaFree(buffer1);
                cudaFree(buffer2);

            }
        }; // Reduce

    } // namespace reduce_lib
    
} // namespace map_reduce_lib


/*
    ==================================================
    Usage Examples
    ==================================================

    Suppose you already have an extern "C" entry point that
    receives device pointers (e.g. handed to you by a Python
    binding, another translation unit, etc.). Here's how to
    call into map_reduce_lib from it.

    --------------------------------------------------
    Example 1: Map only -- scale every element by 2.0
    --------------------------------------------------

    extern "C" void scale_by_two(
        const float* d_input,
        float* d_output,
        int N
    ) {
        map_reduce_lib::map_lib::Map::map_handler(
            d_input,
            d_output,
            N,
            map_reduce_lib::map_lib::Map::Scale(2.0f)
        );
    }

    --------------------------------------------------
    Example 2: Reduce only -- sum all elements down to one value
    --------------------------------------------------

    extern "C" void sum_all(
        const float* d_input,
        float* d_output, // must point to a single float on the device
        int N
    ) {
        map_reduce_lib::reduce_lib::Reduce::reduce_handler(
            d_input,
            d_output,
            map_reduce_lib::reduce_lib::Reduce::SumOp(),
            N
        );
    }

    --------------------------------------------------
    Example 3: Map + Reduce pipeline -- count how many
    elements are >= some threshold K
    --------------------------------------------------

    extern "C" void count_geq(
        const float* d_input,
        float* d_output, // must point to a single float on the device
        int N,
        float K
    ) {
        // Need a scratch buffer of size N for the mapped 0/1 values.
        float* d_mapped;
        cudaMalloc(&d_mapped, N * sizeof(float));

        map_reduce_lib::map_lib::Map::map_handler(
            d_input,
            d_mapped,
            N,
            map_reduce_lib::map_lib::Map::GEQ(K)
        );

        map_reduce_lib::reduce_lib::Reduce::reduce_handler(
            d_mapped,
            d_output,
            map_reduce_lib::reduce_lib::Reduce::SumOp(),
            N
        );

        cudaFree(d_mapped);
    }

    --------------------------------------------------
    Notes
    --------------------------------------------------
    - Both map_handler and reduce_handler expect DEVICE
        pointers already; they do not do any host<->device
        copying for 'input'/'output' themselves.
    - reduce_handler's 'output' only needs room for a single
        T on the device -- cudaMalloc(&d_output, sizeof(T))
        is enough, not N * sizeof(T).
    - To get the final reduce result back on the host:
        T h_result;
        cudaMemcpy(&h_result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
*/
