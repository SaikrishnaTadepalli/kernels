#include <cuda_runtime.h>


#define BLOCK_COL_SIZE 32
#define BLOCK_ROW_SIZE 32

// Guarentee for max Kernel dimensions
#define KERNEL_MAX_SIZE (21 * 21)

  
template <typename T>
struct Matrix {
    T* ptr; int rows; int cols;
    __device__ __forceinline__ int num_rows() { return rows; }
    __device__ __forceinline__ int num_cols() { return cols; }
    __device__ __forceinline__ int num_items() { return rows * cols; }
    __device__ __forceinline__ T& operator()(int r, int c) { return ptr[(r * cols) + c]; }
    __device__ __forceinline__ T& operator()(int flat_index) { return ptr[flat_index]; }
};


struct Index {
    int _row_index; 
    int _col_index; 
    int _flat_index;
    __device__ __forceinline__ int row_index(){ return _row_index; }
    __device__ __forceinline__ int col_index(){ return _col_index; }
    __device__ __forceinline__ int flat_index(){ return _flat_index; }
};

  
__constant__ float const_kernel[KERNEL_MAX_SIZE];


__global__ void convolutional_kernel(
    Matrix<const float> input,
    Matrix<float> output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
) {
    // --------------------------------------------------
    // Get Kernel (from Constant Memory)
    // --------------------------------------------------
    
    // NOTE: This MUST be built on the device. 
    // Taking the address of a '__constant__' symbol on the 
    // host gives the address of the host-side shadow 
    // variable, which is not a valid device pointer.
    Matrix<const float> kernel { const_kernel, kernel_rows, kernel_cols };

    // --------------------------------------------------
    // Get Dimensions
    // --------------------------------------------------
    Index thread_lcl { // Thread Index within the Block
        (int)threadIdx.y, 
        (int)threadIdx.x,
        (int)(threadIdx.y * blockDim.x + threadIdx.x)
    };
    Index block { 
        (int)blockIdx.y, 
        (int)blockIdx.x,
        (int)(blockIdx.y * gridDim.x + blockIdx.x)
    };
    Index thread_gbl { // Thread Index within the Grid
        (int)( (blockIdx.y * blockDim.y) + threadIdx.y ),
        (int)( (blockIdx.x * blockDim.x) + threadIdx.x ),
        (int)( ((blockIdx.y * blockDim.y) + threadIdx.y) * input_cols + ((blockIdx.x * blockDim.x) + threadIdx.x) )
    };
    
    int num_threads_in_block = blockDim.x * blockDim.y;

    // --------------------------------------------------
    // Setup Shared Memory Region
    // --------------------------------------------------
    
    // Declare Shared Memory Region for Input Tile
    int tile_rows = blockDim.y + kernel_rows - 1;
    int tile_cols = blockDim.x + kernel_cols - 1;
    
    extern __shared__ float shared_memory[];
    Matrix<float> input_tile { shared_memory, tile_rows, tile_cols };
    
    int halo_row_radius = kernel.num_rows() / 2; 
    int halo_col_radius = kernel.num_cols() / 2;
    
    // Co-operatively load the Input Tile
    for (
        int i = thread_lcl.flat_index();
        i < input_tile.num_items();
        i += num_threads_in_block
    ) {
        // Index in the Tile we want to load
        int tile_row_index = i / input_tile.num_cols();
        int tile_col_index = i % input_tile.num_cols();
        
        // Desired Index In the Input
        int input_row_index = ( ( block.row_index() * blockDim.y ) + tile_row_index - halo_row_radius );
        int input_col_index = ( ( block.col_index() * blockDim.x ) + tile_col_index - halo_col_radius );
        
        // Guard against out-of-bounds global indices (with padding value of '0' for halo) 
        float input_val = 0.0f; 
        if (
            ( (0 <= input_row_index) && (input_row_index < input.num_rows()) ) &&
            ( (0 <= input_col_index) && (input_col_index < input.num_cols()) )
        ) { input_val = input(input_row_index, input_col_index); }
        
        // Set the value in the Input Tile
        input_tile(tile_row_index, tile_col_index) = input_val;
    }
    
    /*
        tile: (-2, -2) -> (6, 8) | -> (48 total)
        threads: (0, 0) -> (1, 3) | -> (8 total)
        
        Input Tile + Halo:
        H H H H H H H H
        H H H H H H H H
        H H I I I I H H
        H H I I I I H H
        H H H H H H H H
        H H H H H H H H
        
        Tile Relative Index -> Tile Absolute Index -> Tile Flat Index | Thread Index -> Thread Flat Index
        (-2, -2) -> (0, 0) -> 0 | (0, 0) -> 0
        (-2, -1) -> (0, 1) -> 1 | (0, 1) -> 1
        (-2, 0) -> (0, 2) -> 2 | (0, 2) -> 2
        (-2, 1) -> (0, 3) -> 3 | (0, 3) -> 3
        (-2, 2) -> (0, 4) -> 4 | (1, 0) -> 4
        (-2, 3) -> (0, 5) -> 5 | (1, 1) -> 5
        (-2, 4) -> (0, 6) -> 6 | (1, 2) -> 6
        (-2, 5) -> (0, 7) -> 7 | (1, 3) -> 7
        (-1, -2) -> (1, 0) -> 8 | (0, 0) -> 0
        (-1, -1) -> (1, 1) -> 9 | (0, 1) -> 1
        (-1, 0) -> (1, 2) -> 10 | (0, 2) -> 2
        (-1, 1) -> (1, 3) -> 11 | (0, 3) -> 3
        (-1, 2) -> (1, 4) -> 12 | (1, 0) -> 4
        (-1, 3) -> (1, 5) -> 13 | (1, 1) -> 5
        (-1, 4) -> (1, 6) -> 14 | (1, 2) -> 6
        (-1, 5) -> (1, 7) -> 15 | (1, 3) -> 7
        ( 0, -2) -> (2, 0) -> 16 | (0, 0) -> 0
        ( 0, -1) -> (2, 1) -> 17 | (0, 1) -> 1
        ( 0, 0) -> (2, 2) -> 18 | (0, 2) -> 2
        ( 0, 1) -> (2, 3) -> 19 | (0, 3) -> 3
        ( 0, 2) -> (2, 4) -> 20 | (1, 0) -> 4
        ( 0, 3) -> (2, 5) -> 21 | (1, 1) -> 5
        ( 0, 4) -> (2, 6) -> 22 | (1, 2) -> 6
        ( 0, 5) -> (2, 7) -> 23 | (1, 3) -> 7
        ...
    */
    
    // Sync so that we don't use the input_tile until it's all loaded
    __syncthreads();

    // --------------------------------------------------
    // Calculate Convolution
    // --------------------------------------------------
    
    // Only calculate convolution if we are a valid thread
    bool valid_thread = (
        ( thread_gbl.row_index() < input_rows ) && ( thread_gbl.col_index() < input_cols)
    );
    
    if (valid_thread) {
        float convolution_res = 0.0f;
        for (int r = 0; r < kernel_rows; r++) {
            for (int c = 0; c < kernel_cols; c++) {
                // Get Kernel Item
                float kernel_val = kernel(r, c);
                
                // Get Input Tile Item
                int tile_r = thread_lcl.row_index() + r; 
                int tile_c = thread_lcl.col_index() + c;
                float input_tile_val = input_tile(tile_r, tile_c);
                
                // Update Convolution Value
                convolution_res += (input_tile_val * kernel_val);
            }
        }
        
        // Write Convolution Result
        output(thread_gbl.row_index(), thread_gbl.col_index()) = convolution_res;
    }
    
}

  
// input, kernel, output are device pointers
extern "C" void solve(
    const float* input,
    const float* kernel,
    float* output,
    int input_rows,
    int input_cols,
    int kernel_rows,
    int kernel_cols
) {
    // --------------------------------------------------
    // Move Kernel to Constant Memory
    // --------------------------------------------------
    
    size_t kernel_mem_size = (
        kernel_rows * kernel_cols * sizeof(float)
    );
    
    cudaMemcpyToSymbol(
        const_kernel,
        kernel,
        kernel_mem_size,
        0,
        cudaMemcpyDeviceToDevice
    );
    
    // Kernel is small, readonly, and is needed by every
    // thread. Moving it to constant memory should help
    // keep caches clear, which may be helpful for the
    // input processing. This however is not absolutely
    // necessary.

    // --------------------------------------------------
    // Setup Dimensions
    // --------------------------------------------------
    
    // 'output' is the same size as the 'input'
    dim3 threads_per_block ( BLOCK_COL_SIZE, BLOCK_ROW_SIZE );
    dim3 blocks_per_grid (
        ( ( input_cols + BLOCK_COL_SIZE - 1 ) / BLOCK_COL_SIZE ),
        ( ( input_rows + BLOCK_ROW_SIZE - 1 ) / BLOCK_ROW_SIZE )
    );
    
    int input_tile_size = (
        ( BLOCK_ROW_SIZE + kernel_rows - 1) *
        ( BLOCK_COL_SIZE + kernel_cols - 1 )
    );
    
    size_t input_tile_mem_size = input_tile_size * sizeof(float);

    // --------------------------------------------------
    // Launch Kernel
    // --------------------------------------------------
    
    Matrix<const float> input_matrix { input, input_rows, input_cols };
    Matrix<float> output_matrix { output, input_rows, input_cols };
    
    // The kernel Matrix is built on the device from 'const_kernel'
    // (see convolutional_kernel), NOT here on the host.
    convolutional_kernel<<<
        blocks_per_grid,
        threads_per_block,
        input_tile_mem_size
    >>> (
        input_matrix,
        output_matrix,
        input_rows, input_cols,
        kernel_rows, kernel_cols
    );

}
