#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle
{
  namespace cuda
  {

#define BASE_THREAD_NUM 256

#define TILE 4
    typedef float scalar_t;
    const size_t ELEM_SIZE = sizeof(scalar_t);

    struct CudaArray
    {
      CudaArray(const size_t size)
      {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess)
          throw std::runtime_error(cudaGetErrorString(err));
        this->size = size;
      }
      ~CudaArray() { cudaFree(ptr); }
      size_t ptr_as_int() { return (size_t)ptr; }

      scalar_t *ptr;
      size_t size;
    };

    struct CudaDims
    {
      dim3 block, grid;
    };

    CudaDims CudaOneDim(size_t size)
    {
      /**
       * Utility function to get cuda dimensions for 1D call
       */
      CudaDims dim;
      size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
      dim.block = dim3(BASE_THREAD_NUM, 1, 1);
      dim.grid = dim3(num_blocks, 1, 1);
      return dim;
    }

#define MAX_VEC_SIZE 8
    struct CudaVec
    {
      uint32_t size;
      int32_t data[MAX_VEC_SIZE];
    };

    CudaVec VecToCuda(const std::vector<int32_t> &x)
    {
      CudaVec shape;
      if (x.size() > MAX_VEC_SIZE)
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
      shape.size = x.size();
      for (size_t i = 0; i < x.size(); i++)
      {
        shape.data[i] = x[i];
      }
      return shape;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Fill call
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void FillKernel(scalar_t *out, scalar_t val, size_t size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = val;
    }

    void Fill(CudaArray *out, scalar_t val)
    {
      CudaDims dim = CudaOneDim(out->size);
      FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Compact and setitem cals
    ////////////////////////////////////////////////////////////////////////////////

    // Untility function to convert contiguous index i to memory location from strides

    __global__ void CompactKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                  CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   a: CUDA pointer to a array
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN SOLUTION

      // Ensure gid is within the bounds of the array
      if (gid >= size)
        return;

      // Initialize a_index to the provided offset
      size_t a_index = offset;

      // Compute multi-dimensional indices from gid (for the compact 'out' array)
      size_t temp = gid;
      for (int j = shape.size - 1; j >= 0; --j)
      {
        // Calculate the index in dimension j
        size_t idx = temp % shape.data[j];
        temp /= shape.data[j];
        // Update a_index using the strides
        a_index += idx * strides.data[j];
      }

      // Write the corresponding element from 'a' into 'out'
      out[gid] = a[a_index];
      /// END SOLUTION
    }

    void Compact(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                 std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the
       * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give
       * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
       * the functions after this, however, you'll need to define these kernels as you see fit to
       * execute the underlying function.
       *
       * Args:
       *   a: non-compact represntation of the array, given as input
       *   out: compact version of the array to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *a* array (not out, which has compact strides)
       *   offset: offset of the *a* array (not out, which has zero offset, being compact)
       */

      // Nothing needs to be added here
      CudaDims dim = CudaOneDim(out->size);
      CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                             VecToCuda(strides), offset);
    }

    __global__ void EwiseSetitemKernel(const scalar_t *a, scalar_t *out, size_t size, CudaVec shape,
                                       CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   a: CUDA pointer to a array
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN SOLUTION

      // Ensure gid is within the bounds of the array
      if (gid >= size)
        return;

      // Initialize a_index to the provided offset
      size_t out_index = offset;

      // Compute multi-dimensional indices from gid (for the compact 'out' array)
      size_t temp = gid;
      for (int j = shape.size - 1; j >= 0; --j)
      {
        // Calculate the index in dimension j
        size_t idx = temp % shape.data[j];
        temp /= shape.data[j];
        // Update a_index using the strides
        out_index += idx * strides.data[j];
      }

      // size_t temp = gid;
      // for (int j = 0; j < shape.size; ++j) {  // Iterate from 0 to shape.size - 1
      //     size_t idx = temp % shape.data[j];
      //     temp /= shape.data[j];
      //     out_index += idx * strides.data[j];  // Use strides for correct positioning
      // }

      // Write the corresponding element from 'a' into 'out'
      out[out_index] = a[gid];
      /// END SOLUTION
    }

    void EwiseSetitem(const CudaArray &a, CudaArray *out, std::vector<int32_t> shape,
                      std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
       * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
       *
       * Args:
       *   a: _compact_ array whose items will be written to out
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension for a and out
       *   strides: strides of the *out* array (not a, which has compact strides)
       *   offset: offset of the *out* array (not a, which has zero offset, being compact)
       */
      /// BEGIN SOLUTION
      // for(auto i: shape){
      //   std::cout<<i<<" ";
      // }
      // std::cout<<"\n";

      // std::cout<<"out size - "<<out->size<<"\n";
      // std::cout<<"a size - "<<a.size<<"\n";
      // for(auto i: strides){
      //   std::cout<<i<<" ";
      // }
      // std::cout<<"\n";
      CudaDims dim = CudaOneDim(a.size);
      EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                                  VecToCuda(strides), offset);
      /// END SOLUTION
    }

    __global__ void ScalarSetitemKernel(scalar_t val, scalar_t *out, size_t size, CudaVec shape,
                                        CudaVec strides, size_t offset)
    {
      /**
       * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the
       * non-compact input a, to the corresponding item (at location gid) in the compact array out.
       *
       * Args:
       *   val: scalar value to write to out
       *   out: CUDA point to out array
       *   size: size of out array
       *   shape: vector of shapes of out array (of type CudaVec, for past passing to CUDA kernel)
       *   strides: vector of strides of out array
       *   offset: offset of out array
       */
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      /// BEGIN SOLUTION

      // Ensure gid is within the bounds of the array
      if (gid >= size)
        return;

      // Initialize a_index to the provided offset
      size_t out_index = offset;

      // Compute multi-dimensional indices from gid (for the compact 'out' array)
      size_t temp = gid;
      for (int j = shape.size - 1; j >= 0; --j)
      {
        // Calculate the index in dimension j
        size_t idx = temp % shape.data[j];
        temp /= shape.data[j];
        // Update a_index using the strides
        out_index += idx * strides.data[j];
      }

      // Write the corresponding element into 'out'
      out[out_index] = val;
      /// END SOLUTION
    }

    void ScalarSetitem(size_t size, scalar_t val, CudaArray *out, std::vector<int32_t> shape,
                       std::vector<int32_t> strides, size_t offset)
    {
      /**
       * Set items is a (non-compact) array
       *
       * Args:
       *   size: number of elements to write in out array (note that this will note be the same as
       *         out.size, because out is a non-compact subset array);  it _will_ be the same as the
       *         product of items in shape, but covenient to just pass it here.
       *   val: scalar value to write to
       *   out: non-compact array whose items are to be written
       *   shape: shapes of each dimension of out
       *   strides: strides of the out array
       *   offset: offset of the out array
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(size);
      ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                                   VecToCuda(strides), offset);
      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Elementwise and scalar operations
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void EwiseAddKernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + b[gid];
    }

    void EwiseAdd(const CudaArray &a, const CudaArray &b, CudaArray *out)
    {
      /**
       * Add together two CUDA arrays.
       * Args:
       *   a: Input array 'a' to be added
       *   b: Input array 'b' to be added
       *   out: Output array to store the result of 'a + b'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Kernel will execute on 'dim.grid' blocks, each containing 'dim.block' threads.
      EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
    }

    __global__ void ScalarAddKernel(const scalar_t *a, scalar_t val, scalar_t *out, size_t size)
    {
      // Calculate the global index of the thread.
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
      if (gid < size)
        out[gid] = a[gid] + val;
    }

    void ScalarAdd(const CudaArray &a, scalar_t val, CudaArray *out)
    {
      /**
       * Add a scalar value to every element of a CUDA array.
       * Args:
       *   a: Input array 'a'
       *   val: Scalar value to be added
       *   out: Output array to store the result of 'a + val'
       */
      CudaDims dim = CudaOneDim(out->size);

      // Launch the ScalarAddKernel that will add the scalar 'val' to each element of array 'a',
      // and store the result in array 'out'.
      ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
    }

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

// Macro for element-wise operations
#define DEFINE_EWISE_KERNEL(func_name, device_op)                                                     \
  __global__ void func_name##Kernel(const scalar_t *a, const scalar_t *b, scalar_t *out, size_t size) \
  {                                                                                                   \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                               \
    if (gid < size)                                                                                   \
    {                                                                                                 \
      out[gid] = device_op(a[gid], b[gid]);                                                           \
    }                                                                                                 \
  }                                                                                                   \
  void func_name(const CudaArray &a, const CudaArray &b, CudaArray *out)                              \
  {                                                                                                   \
    CudaDims dim = CudaOneDim(out->size);                                                             \
    func_name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);                    \
  }

// Macro for scalar operations
#define DEFINE_SCALAR_KERNEL(func_name, device_op)                                             \
  __global__ void func_name##Kernel(const scalar_t *a, scalar_t b, scalar_t *out, size_t size) \
  {                                                                                            \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                                        \
    if (gid < size)                                                                            \
    {                                                                                          \
      out[gid] = device_op(a[gid], b);                                                         \
    }                                                                                          \
  }                                                                                            \
  void func_name(const CudaArray &a, scalar_t b, CudaArray *out)                               \
  {                                                                                            \
    CudaDims dim = CudaOneDim(out->size);                                                      \
    func_name##Kernel<<<dim.grid, dim.block>>>(a.ptr, b, out->ptr, out->size);                 \
  }

// Macro for unary operations (like log, exp, tanh)
#define DEFINE_UNARY_KERNEL(func_name, device_op)                                  \
  __global__ void func_name##Kernel(const scalar_t *a, scalar_t *out, size_t size) \
  {                                                                                \
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;                            \
    if (gid < size)                                                                \
    {                                                                              \
      out[gid] = device_op(a[gid]);                                                \
    }                                                                              \
  }                                                                                \
  void func_name(const CudaArray &a, CudaArray *out)                               \
  {                                                                                \
    CudaDims dim = CudaOneDim(out->size);                                          \
    func_name##Kernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size);        \
  }

    // Element-wise and scalar operations
    __device__ scalar_t mul_op(scalar_t a, scalar_t b) { return a * b; }
    __device__ scalar_t div_op(scalar_t a, scalar_t b) { return a / b; }
    __device__ scalar_t pow_op(scalar_t a, scalar_t b) { return powf(a, b); }
    __device__ scalar_t max_op(scalar_t a, scalar_t b) { return fmaxf(a, b); }
    __device__ scalar_t eq_op(scalar_t a, scalar_t b) { return a == b; }
    __device__ scalar_t ge_op(scalar_t a, scalar_t b) { return a >= b; }
    __device__ scalar_t log_op(scalar_t a) { return logf(a); }
    __device__ scalar_t exp_op(scalar_t a) { return expf(a); }
    __device__ scalar_t tanh_op(scalar_t a) { return tanhf(a); }

    // Element-wise functions
    DEFINE_EWISE_KERNEL(EwiseMul, mul_op)
    DEFINE_EWISE_KERNEL(EwiseDiv, div_op)
    DEFINE_EWISE_KERNEL(EwiseMaximum, max_op)
    DEFINE_EWISE_KERNEL(EwiseEq, eq_op)
    DEFINE_EWISE_KERNEL(EwiseGe, ge_op)

    // Scalar functions
    DEFINE_SCALAR_KERNEL(ScalarMul, mul_op)
    DEFINE_SCALAR_KERNEL(ScalarDiv, div_op)
    DEFINE_SCALAR_KERNEL(ScalarPower, pow_op)
    DEFINE_SCALAR_KERNEL(ScalarMaximum, max_op)
    DEFINE_SCALAR_KERNEL(ScalarEq, eq_op)
    DEFINE_SCALAR_KERNEL(ScalarGe, ge_op)

    // Unary functions
    DEFINE_UNARY_KERNEL(EwiseLog, log_op)
    DEFINE_UNARY_KERNEL(EwiseExp, exp_op)
    DEFINE_UNARY_KERNEL(EwiseTanh, tanh_op)

    __global__ void MatmulKernel(const scalar_t *A, const scalar_t *B, scalar_t *C, uint32_t M, uint32_t N, uint32_t P)
    {
      // Shared memory for tiles of A and B
      __shared__ scalar_t sA[TILE][TILE];
      __shared__ scalar_t sB[TILE][TILE];

      // Compute the row and column indices for the output matrix C
      int row = blockIdx.y * TILE + threadIdx.y; // Row index for C
      int col = blockIdx.x * TILE + threadIdx.x; // Column index for C

      // Initialize the partial output value for the C matrix
      scalar_t Cvalue = 0.0f;

      // Loop over the tiles of the matrices
      for (int tileIdx = 0; tileIdx < (N + TILE - 1) / TILE; ++tileIdx)
      {
        // Load data into shared memory
        if (row < M && (tileIdx * TILE + threadIdx.x) < N)
        {
          sA[threadIdx.y][threadIdx.x] = A[row * N + (tileIdx * TILE + threadIdx.x)];
        }
        else
        {
          sA[threadIdx.y][threadIdx.x] = 0.0f; // Out of bounds
        }

        if (col < P && (tileIdx * TILE + threadIdx.y) < N)
        {
          sB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE + threadIdx.y) * P + col];
        }
        else
        {
          sB[threadIdx.y][threadIdx.x] = 0.0f; // Out of bounds
        }

        // Synchronize to ensure all data is loaded
        __syncthreads();

        // Compute the partial sum for the block
        for (int k = 0; k < TILE; ++k)
        {
          Cvalue += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        // Synchronize before loading the next tile
        __syncthreads();
      }

      // Write the result back to global memory
      if (row < M && col < P)
      {
        C[row * P + col] = Cvalue; // Using P for column index in C
      }
    }

    void Matmul(const CudaArray &a, const CudaArray &b, CudaArray *out, uint32_t M, uint32_t N,
                uint32_t P)
    {
      /**
       * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
       * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
       * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
       * over (i,j) entries in the output array.  However, to really get the full benefit of this
       * problem, we would encourage you to use cooperative fetching, shared memory register tiling,
       * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
       * the CPU backend, here you should implement a single function that works across all size
       * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
       * implementations, this function here will largely just set up the kernel call, and you should
       * implement the logic in a separate MatmulKernel() call.
       *
       *
       * Args:
       *   a: compact 2D array of size m x n
       *   b: comapct 2D array of size n x p
       *   out: compact 2D array of size m x p to write the output to
       *   M: rows of a / out
       *   N: columns of a / rows of b
       *   P: columns of b / out
       */

      /// BEGIN SOLUTION
      // assert(false && "Not Implemented");
      // Define grid and block sizes
      dim3 blockDim(TILE, TILE);                                  // TILE_SIZE x TILE_SIZE threads per block
      dim3 gridDim((P + TILE - 1) / TILE, (M + TILE - 1) / TILE); // Ensure full coverage

      // Launch the kernel
      MatmulKernel<<<gridDim, blockDim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
      /// END SOLUTION
    }

    ////////////////////////////////////////////////////////////////////////////////
    // Max and sum reductions
    ////////////////////////////////////////////////////////////////////////////////

    __global__ void ReduceMaxKernel(const scalar_t *a, scalar_t *out, size_t out_size, size_t reduce_size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      // Each thread reduces one element of `out`
      if (gid < out_size)
      {
        scalar_t max_val = a[gid * reduce_size]; // Initialize with the first value in the block
        for (size_t i = 1; i < reduce_size; ++i)
        {
          scalar_t val = a[gid * reduce_size + i];
          if (val > max_val)
          {
            max_val = val; // Update maximum value
          }
        }
        out[gid] = max_val;
      }
    }

    void ReduceMax(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
       * for simplicity you can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      // Determine the grid and block dimensions
      CudaDims dim = CudaOneDim(out->size);

      // Launch the kernel
      ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
      /// END SOLUTION
    }

    __global__ void ReduceSumKernel(const scalar_t *a, scalar_t *out, size_t out_size, size_t reduce_size)
    {
      size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

      // Each thread reduces one element of `out`
      if (gid < out_size)
      {
        scalar_t sum = 0.0;
        for (size_t i = 0; i < reduce_size; ++i)
        {
          sum += a[gid * reduce_size + i];
        }
        out[gid] = sum;
      }
    }

    void ReduceSum(const CudaArray &a, CudaArray *out, size_t reduce_size)
    {
      /**
       * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you
       * can perform each reduction in a single CUDA thread.
       *
       * Args:
       *   a: compact array of size a.size = out.size * reduce_size to reduce over
       *   out: compact array to write into
       *   redice_size: size of the dimension to reduce over
       */
      /// BEGIN SOLUTION
      CudaDims dim = CudaOneDim(out->size);

      // Launch the kernel
      ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, reduce_size);
      /// END SOLUTION
    }

    __global__ void PscanKernel(const scalar_t *a, scalar_t *x, scalar_t *out, size_t batch_size, size_t dim, size_t seqlen, size_t dstate)
    {
      const size_t n = seqlen;

      __shared__ scalar_t temp_a[dstate][n];
      __shared__ scalar_t temp_x[dstate][n];

      const int batch_id = blockIdx.x;
      const int dim_id = blockIdx.y;

      const int thid = threadIdx.x;
      const int off = batch_id * dim * seqlen * dstate + dim_id * seqlen * dstate;

      // load data to shared memory
      temp_a[threadIdx.y][2 * thid] = a[off + (2 * thid) * dstate + threadIdx.y];
      temp_a[threadIdx.y][2 * thid + 1] = a[off + (2 * thid + 1) * dstate + threadIdx.y];

      temp_x[threadIdx.y][2 * thid] = x[off + (2 * thid) * dstate + threadIdx.y];
      temp_x[threadIdx.y][2 * thid + 1] = x[off + (2 * thid + 1) * dstate + threadIdx.y];

      // up sweep
      int offset = 1;
      for (int d = n >> 1; d > 0; d >>= 1)
      {
        __syncthreads();
        if (thid < d)
        {
          const int ai = offset * (2 * thid + 1) - 1;
          const int bi = offset * (2 * thid + 2) - 1;

          // make_float2(ab1.x * ab0.x, ab1.x * ab0.y + ab1.y); from ssm cuda code
          temp_x[threadIdx.y][bi] += temp_a[threadIdx.y][bi] * temp_x[threadIdx.y][ai];
          temp_a[threadIdx.y][bi] *= temp_a[threadIdx.y][ai];
        }
        offset *= 2;
      }

      // clear last element
      if (thid == 0)
        temp_x[threadIdx.y][n - 1] = 0;

      // down sweep
      for (int d = 1; d < n; d *= 2)
      {
        offset >>= 1;
        __syncthreads();

        if (thid < d)
        {
          const int ai = offset * (2 * thid + 1) - 1;
          const int bi = offset * (2 * thid + 2) - 1;

          const scalar_t t_a = temp_a[threadIdx.y][ai];
          const scalar_t t_x = temp_x[threadIdx.y][ai];

          temp_a[threadIdx.y][ai] = temp_a[threadIdx.y][bi];
          temp_x[threadIdx.y][ai] = temp_x[threadIdx.y][bi];

          temp_a[threadIdx.y][bi] *= t_a;
          temp_x[threadIdx.y][bi] += temp_a[threadIdx.y][bi] * t_x;
        }
      }

      __syncthreads();

      // write back to global memory with inclusive prefix scan
      out[off + (2 * thid) * dstate + threadIdx.y] = temp_x[threadIdx.y][2 * thid + 1];
      if (2 * thid + 2 < n)
      {
        out[off + (2 * thid + 1) * dstate + threadIdx.y] = temp_x[threadIdx.y][2 * thid + 2];
      }
      else
      {
        out[off + (2 * thid + 1) * dstate + threadIdx.y] = x[off + (2 * thid + 1) * threadIdx.y] + a[off + (2 * thid + 1) * threadIdx.y] * temp_x[threadIdx.y][2 * thid + 1];
      }
    }

    void Pscan(const CudaArray &a, CudaArray &x, CudaArray *out, std::vector<int32_t> shape)
    {
      const int32_t batch_size = shape[0];
      const int32_t dim = shape[1];
      const int32_t seqlen = shape[2];
      const int32_t dstate = shape[3];

      dim3 grid = dim3(batch_size, dim, 1);
      dim3 block = dim3(seqlen / 2, dstate, 1);
      PscanKernel<<<grid, block>>>(a.ptr, x.ptr, out->ptr, out->size, batch_size, dim, seqlen, dstate);
    }

  } // namespace cuda
} // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray &a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset)
        {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer); });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray *out)
        {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err)); });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);

  m.def("pscan", Pscan);
}
