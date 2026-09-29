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
