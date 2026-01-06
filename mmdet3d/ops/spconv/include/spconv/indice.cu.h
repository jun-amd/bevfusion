// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef INDICE_CU_H_
#define INDICE_CU_H_
#include <spconv/geometry.h>
#include <tensorview/helper_kernel.cu.h>
#include <tensorview/tensorview.h>

namespace spconv {
template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void prepareIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPos<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 256>
__global__ void prepareDeConvIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<Index> indicesOut,
    tv::TensorView<IndexGrid> gridsOut, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indiceNum, tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index kernelVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    kernelVolume *= kernelSize[i];
  }
  Index numValidPoints = 0;
  Index validPoints[KernelMaxVolume * (NDim + 1)];
  Index *pointPtr = nullptr;
  auto indicePairsDim2 = indicePairs.dim(2);
  Index index;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    numValidPoints = getValidOutPosTranspose<Index, NDim>(
        indicesIn.data() + ix * (NDim + 1) + 1, kernelSize.data(),
        stride.data(), padding.data(), dilation.data(), outSpatialShape.data(),
        validPoints);
    for (Index i = 0; i < numValidPoints; ++i) {
      pointPtr = validPoints + i * (NDim + 1);
      auto offset = pointPtr[NDim];
      auto oldNum = atomicAdd(indiceNum.data() + offset, Index(1));
      indicePairs(offset, 0, oldNum) = ix;
      index = tv::rowArrayIdx<Index, NDim>(pointPtr, outSpatialShape.data()) +
              spatialVolume * indicesIn(ix, 0);
      indicePairs(offset, 1, oldNum) = index;
      indicePairUnique[offset * indicePairsDim2 + oldNum] = index;
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void assignGridAndIndiceOutKernel(
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numAct, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int batchSize) {
  Index index;
  auto indicesOutPtr = indicesOut.data();
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    index = indicePairUnique[ix];
    gridsOut[index] = ix;
    index = tv::rowArrayIdxInv<Index, NDim>(
        index, indicesOutPtr + ix * (NDim + 1) + 1, outSpatialShape.data());
    indicesOut[ix * (NDim + 1)] = index % batchSize;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void assignIndicePairsKernel(
    tv::TensorView<Index> indicesOut, tv::TensorView<IndexGrid> gridsOut,
    int numActIn, tv::TensorView<Index> indicePairs,
    tv::TensorView<Index> indicePairUnique,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  Index index;
  int kernelVolume = indicePairs.dim(0);
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    for (int i = 0; i < kernelVolume; ++i) {
      index = indicePairs(i, 1, ix);
      if (index > -1) {
        indicePairs(i, 1, ix) = gridsOut[index];
      }
    }
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void prepareSubMGridKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {
  auto numActIn = indicesIn.dim(0);
  Index spatialVolume = 1;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index = 0;
  for (int ix : tv::KernelLoopX<int>(numActIn)) {
    index = tv::rowArrayIdx<Index, NDim>(indicesIn.data() + ix * (NDim + 1) + 1,
                                         outSpatialShape.data()) +
            spatialVolume * indicesIn(ix, 0);
    gridsOut[index] = ix;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim,
          int KernelMaxVolume = 4096, int MAX_PAIRS_PER_BLOCK = 1024>
__global__ void __launch_bounds__(256) getSubMIndicePairsKernel(
    tv::TensorView<const Index> indicesIn, tv::TensorView<IndexGrid> gridsOut,
    tv::TensorView<Index> indicePairs, tv::TensorView<Index> indiceNum,
    const tv::SimpleVector<Index, NDim> kernelSize,
    const tv::SimpleVector<Index, NDim> stride,
    const tv::SimpleVector<Index, NDim> padding,
    const tv::SimpleVector<Index, NDim> dilation,
    const tv::SimpleVector<Index, NDim> outSpatialShape) {

  // Shared memory for caching
  extern __shared__ char sharedMem[];
  Index *sharedIndiceNum = reinterpret_cast<Index *>(sharedMem);
  Index *sharedLocalCounter = sharedIndiceNum + KernelMaxVolume;
  Index *sharedPairs = sharedLocalCounter + KernelMaxVolume;

  const Index *__restrict__ indicesInPtr = indicesIn.data();
  IndexGrid *__restrict__ gridsOutPtr = gridsOut.data();
  Index *__restrict__ indicePairsPtr = indicePairs.data();
  Index *__restrict__ indiceNumPtr = indiceNum.data();

  auto numActIn = indicesIn.dim(0);

  // Load parameters into registers
  Index outShapeReg[NDim];
  Index kSizeReg[NDim];
  Index strReg[NDim];
  Index padReg[NDim];
  Index dilReg[NDim];
  Index spatialVolume = 1;
  Index kernelVolume = 1;

#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    outShapeReg[i] = outSpatialShape[i];
    kSizeReg[i] = kernelSize[i];
    strReg[i] = stride[i];
    padReg[i] = padding[i];
    dilReg[i] = dilation[i];
    spatialVolume *= outShapeReg[i];
    kernelVolume *= kSizeReg[i];
  }

  const auto indicePairsDim1 = indicePairs.dim(1);
  const auto indicePairsDim2 = indicePairs.dim(2);
  const Index pairStride = indicePairsDim1 * indicePairsDim2;

  // Initialize shared memory
  for (int i = threadIdx.x; i < KernelMaxVolume; i += blockDim.x) {
    sharedIndiceNum[i] = 0;
    sharedLocalCounter[i] = 0;
  }
  __syncthreads();

  __shared__ Index sharedPairCount;
  if (threadIdx.x == 0) {
    sharedPairCount = 0;
  }
  __syncthreads();

  // Calculate work range
  int elementsPerBlock = (numActIn + gridDim.x - 1) / gridDim.x;
  int blockStart = blockIdx.x * elementsPerBlock;
  int blockEnd = min(blockStart + elementsPerBlock, (int)numActIn);

  // Process elements - use inline getValidOutPos logic
  for (int ix = blockStart + threadIdx.x; ix < blockEnd; ix += blockDim.x) {
    // Load input data
    const Index batchIdx = indicesInPtr[ix * (NDim + 1)];
    Index inPos[NDim];
    Index upper[NDim];
    Index lower[NDim];
    Index counterSize[NDim];
    Index numPoints = 1;

#pragma unroll
    for (int d = 0; d < NDim; ++d) {
      inPos[d] = indicesInPtr[ix * (NDim + 1) + 1 + d];
      upper[d] = (inPos[d] + padReg[d]) / strReg[d];
      lower[d] = (inPos[d] - (kSizeReg[d] - 1) * dilReg[d] - 1 + strReg[d] +
                  padReg[d]) /
                 strReg[d];
      counterSize[d] = (upper[d] - lower[d]) / dilReg[d] + 1;
      numPoints *= counterSize[d];
    }

    const Index batchOffset = spatialVolume * batchIdx;

    // Iterate through all kernel positions using counter-based approach
    Index counter[NDim];
#pragma unroll
    for (int d = 0; d < NDim; ++d) {
      counter[d] = 0;
    }

    for (Index i = 0; i < numPoints; ++i) {
      // Compute val, index, and offset for current counter state
      bool valid = true;
      Index m = 1;
      Index offset = 0;
      Index index = 0;
      Index indexMult = 1;

// Compute offset from NDim-1 to 0 (matching original getValidOutPos)
#pragma unroll
      for (int j = NDim - 1; j >= 0; --j) {
        Index val = upper[j] - counter[j] * dilReg[j];
        if (val < 0 || val >= outShapeReg[j]) {
          valid = false;
        }
        offset += m * (inPos[j] - val * strReg[j] + padReg[j]) / dilReg[j];
        m *= kSizeReg[j];
      }

      // Compute index in forward order (0 to NDim-1) for row-major layout
      index = 0;
#pragma unroll
      for (int j = 0; j < NDim; ++j) {
        Index val = upper[j] - counter[j] * dilReg[j];
        index = index * outShapeReg[j] + val;
      }
      index += batchOffset;

      if (valid) {
        const IndexGrid gridVal = gridsOutPtr[index];

        if (gridVal > -1) {
          // Atomically get slot in shared memory
          Index pairIdx = atomicAdd(&sharedPairCount, Index(1));
          if (pairIdx < MAX_PAIRS_PER_BLOCK) {
            sharedPairs[pairIdx * 3] = offset;
            sharedPairs[pairIdx * 3 + 1] = ix;
            sharedPairs[pairIdx * 3 + 2] = gridVal;
            atomicAdd(&sharedIndiceNum[offset], Index(1));
          } else {
            // Fallback to global memory
            const Index oldNum = atomicAdd(indiceNumPtr + offset, Index(1));
            const Index baseIdx = offset * pairStride + oldNum;
            indicePairsPtr[baseIdx + indicePairsDim2] = gridVal;
            indicePairsPtr[baseIdx] = ix;
          }
        }
      }

      // Increment counter (like odometer)
      counter[NDim - 1] += 1;
#pragma unroll
      for (int c = NDim - 1; c >= 0; --c) {
        if (counter[c] == counterSize[c] && c > 0) {
          counter[c - 1] += 1;
          counter[c] = 0;
        }
      }
    }
  }

  __syncthreads();

  // Flush shared memory to global
  Index totalPairs = min(sharedPairCount, (Index)MAX_PAIRS_PER_BLOCK);

  __shared__ Index globalBaseNum[KernelMaxVolume];
  for (int i = threadIdx.x; i < kernelVolume; i += blockDim.x) {
    if (sharedIndiceNum[i] > 0) {
      globalBaseNum[i] = atomicAdd(indiceNumPtr + i, sharedIndiceNum[i]);
    } else {
      globalBaseNum[i] = 0;
    }
    sharedLocalCounter[i] = 0;
  }
  __syncthreads();

  for (int pairIdx = threadIdx.x; pairIdx < totalPairs; pairIdx += blockDim.x) {
    Index offset = sharedPairs[pairIdx * 3];
    Index ix_val = sharedPairs[pairIdx * 3 + 1];
    Index gridVal = sharedPairs[pairIdx * 3 + 2];

    Index localIdx = atomicAdd(&sharedLocalCounter[offset], Index(1));
    Index globalIdx = globalBaseNum[offset] + localIdx;
    const Index baseIdx = offset * pairStride + globalIdx;
    indicePairsPtr[baseIdx + indicePairsDim2] = gridVal;
    indicePairsPtr[baseIdx] = ix_val;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void resetGridKernel(const Index *indicePairUnique,
                                tv::TensorView<IndexGrid> gridsOut,
                                int numAct) {
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    gridsOut[indicePairUnique[ix]] = -1;
  }
}

template <typename Index, typename IndexGrid, unsigned NDim>
__global__ void resetGridSubMKernel(
    const Index *indices, tv::TensorView<IndexGrid> gridsOut,
    const tv::SimpleVector<Index, NDim> outSpatialShape, int numAct) {
  int outSpatialShapeReg[NDim];
  for (int i = 0; i < NDim; ++i) {
    outSpatialShapeReg[i] = outSpatialShape[i];
  }
  Index spatialVolume = 1;
  auto indsPtr = indices;
#pragma unroll
  for (int i = 0; i < NDim; ++i) {
    spatialVolume *= outSpatialShape[i];
  }
  Index index;
  for (int ix : tv::KernelLoopX<int>(numAct)) {
    indsPtr = indices + ix * (NDim + 1);
    index = tv::rowArrayIdx<Index, NDim>(indsPtr + 1, outSpatialShapeReg);
    gridsOut[index + spatialVolume * indsPtr[0]] = -1;
  }
}

}  // namespace spconv

#endif
