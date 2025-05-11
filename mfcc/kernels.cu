                                         #include <cuda_runtime.h>
                                           #include <cufft.h>

                                           // Power spectrumini hisoblash uchun CUDA kernel
                                           __global__ void powerSpectrumKernel(cufftComplex* fftOut, float* powerSpec, int n) {
                                               int idx = blockIdx.x * blockDim.x + threadIdx.x;
                                               if (idx < n/2 + 1) {
                                                   float re = fftOut[idx].x;
                                                   float im = fftOut[idx].y;
                                                   powerSpec[idx] = re * re + im * im;
                                               }
                                           }

                                           // Mel filtrlarini qo‘llash uchun CUDA kernel
                                           __global__ void applyMelFiltersKernel(float* powerSpec, float* filterBanks, float* melEnergies, int numFilters, int frameSize) {
                                               int idx = blockIdx.x * blockDim.x + threadIdx.x;
                                               if (idx < numFilters) {
                                                   float energy = 0.0f;
                                                   for (int j = 0; j < frameSize/2 + 1; j++) {
                                                       energy += powerSpec[j] * filterBanks[idx * (frameSize/2 + 1) + j];
                                                   }
                                                   melEnergies[idx] = energy;
                                               }
                                           }

                                           // Log operatsiyasi uchun CUDA kernel
                                           __global__ void logKernel(float* input, float* output, int n) {
                                               int idx = blockIdx.x * blockDim.x + threadIdx.x;
                                               if (idx < n) {
                                                   output[idx] = logf(input[idx] + 1e-6f);
                                               }
                                           }

                                           // DCT uchun CUDA kernel
                                           __global__ void dctKernel(float* input, float* output, int n, int numCoeffs, float sqrt2OverN) {
                                               int idx = blockIdx.x * blockDim.x + threadIdx.x;
                                               if (idx < numCoeffs) {
                                                   float sum = 0.0f;
                                                   for (int m = 0; m < n; m++) {
                                                       float angle = 3.14159265359 * idx * (m + 0.5) / n;
                                                       sum += input[m] * cosf(angle);
                                                   }
                                                   output[idx] = sum * sqrt2OverN;
                                               }
                                           }

                                           // CUDA kernelni Go’dan chaqirish uchun yordamchi funksiyalar
                                           extern "C" void launchPowerSpectrumKernel(cufftComplex* fftOut, float* powerSpec, int n, int gridSize, int blockSize, cudaStream_t stream) {
                                               powerSpectrumKernel<<<gridSize, blockSize, 0, stream>>>(fftOut, powerSpec, n);
                                           }

                                           extern "C" void launchApplyMelFiltersKernel(float* powerSpec, float* filterBanks, float* melEnergies, int numFilters, int frameSize, int gridSize, int blockSize, cudaStream_t stream) {
                                               applyMelFiltersKernel<<<gridSize, blockSize, 0, stream>>>(powerSpec, filterBanks, melEnergies, numFilters, frameSize);
                                           }

                                           extern "C" void launchLogKernel(float* input, float* output, int n, int gridSize, int blockSize, cudaStream_t stream) {
                                               logKernel<<<gridSize, blockSize, 0, stream>>>(input, output, n);
                                           }

                                           extern "C" void launchDctKernel(float* input, float* output, int n, int numCoeffs, float sqrt2OverN, int gridSize, int blockSize, cudaStream_t stream) {
                                               dctKernel<<<gridSize, blockSize, 0, stream>>>(input, output, n, numCoeffs, sqrt2OverN);
                                           }