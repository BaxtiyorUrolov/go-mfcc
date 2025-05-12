package mfcc

/*
#cgo CFLAGS: -I/usr/local/cuda-12.8/targets/x86_64-linux/include -Wall
#cgo LDFLAGS: -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -lcudart -lcufft /home/baxtiyor/go-mfcc/kernels.o
#include <cuda_runtime.h>
#include <cufft.h>

// Yordamchi funksiyalar deklaratsiyasi
void launchPowerSpectrumKernel(cufftComplex* fftOut, float* powerSpec, int n, int gridSize, int blockSize, cudaStream_t stream);
void launchApplyMelFiltersKernel(float* powerSpec, float* filterBanks, float* melEnergies, int numFilters, int frameSize, int gridSize, int blockSize, cudaStream_t stream);
void launchLogKernel(float* input, float* output, int n, int gridSize, int blockSize, cudaStream_t stream);
void launchDctKernel(float* input, float* output, int n, int numCoeffs, float sqrt2OverN, int gridSize, int blockSize, cudaStream_t stream);
*/
import "C"
import (
	"fmt"
	"math"
	"unsafe"
)

// GPUContext - GPU’da hisoblash uchun kontekst
type GPUContext struct {
	plan          C.cufftHandle
	stream        C.cudaStream_t
	deviceFrame   unsafe.Pointer
	deviceFFT     unsafe.Pointer
	devicePower   unsafe.Pointer
	deviceMel     unsafe.Pointer
	deviceLog     unsafe.Pointer
	deviceDCT     unsafe.Pointer
	deviceFilters unsafe.Pointer
	hostBuffer    []float32
	frameLength   int
	numFilters    int
	numCoeffs     int
}

// NewGPUContext - Yangi GPU kontekstini yaratish
func NewGPUContext(frameLength, numFilters, numCoefficients int) (*GPUContext, error) {
	var ctx GPUContext
	ctx.frameLength = frameLength
	ctx.numFilters = numFilters
	ctx.numCoeffs = numCoefficients

	if res := C.cudaStreamCreate(&ctx.stream); res != C.cudaSuccess {
		return nil, fmt.Errorf("CUDA stream yaratishda xatolik: %v", res)
	}

	var plan C.cufftHandle
	if res := C.cufftPlan1d(&plan, C.int(frameLength), C.CUFFT_R2C, 1); res != C.CUFFT_SUCCESS {
		return nil, fmt.Errorf("cuFFT reja yaratishda xatolik: %v", res)
	}
	ctx.plan = plan

	if res := C.cudaMalloc(&ctx.deviceFrame, C.size_t(frameLength*4)); res != C.cudaSuccess {
		return nil, fmt.Errorf("deviceFrame uchun xotira ajratishda xatolik: %v", res)
	}
	if res := C.cudaMalloc(&ctx.deviceFFT, C.size_t((frameLength/2+1)*8)); res != C.cudaSuccess {
		return nil, fmt.Errorf("deviceFFT uchun xotira ajratishda xatolik: %v", res)
	}
	if res := C.cudaMalloc(&ctx.devicePower, C.size_t((frameLength/2+1)*4)); res != C.cudaSuccess {
		return nil, fmt.Errorf("devicePower uchun xotira ajratishda xatolik: %v", res)
	}
	if res := C.cudaMalloc(&ctx.deviceMel, C.size_t(numFilters*4)); res != C.cudaSuccess {
		return nil, fmt.Errorf("deviceMel uchun xotira ajratishda xatolik: %v", res)
	}
	if res := C.cudaMalloc(&ctx.deviceLog, C.size_t(numFilters*4)); res != C.cudaSuccess {
		return nil, fmt.Errorf("deviceLog uchun xotira ajratishda xatolik: %v", res)
	}
	if res := C.cudaMalloc(&ctx.deviceDCT, C.size_t(numCoefficients*4)); res != C.cudaSuccess {
		return nil, fmt.Errorf("deviceDCT uchun xotira ajratishda xatolik: %v", res)
	}
	if res := C.cudaMalloc(&ctx.deviceFilters, C.size_t(numFilters*(frameLength/2+1)*4)); res != C.cudaSuccess {
		return nil, fmt.Errorf("deviceFilters uchun xotira ajratishda xatolik: %v", res)
	}

	ctx.hostBuffer = make([]float32, frameLength)
	return &ctx, nil
}

// ComputeMFCC - GPU’da MFCC ni hisoblash
func (ctx *GPUContext) ComputeMFCC(frames [][]float32, filterBanks [][]float32, window []float32, cfg Config) ([][]float32, error) {
	numFrames := len(frames)
	mfccs := make([][]float32, numFrames)

	flatFilters := make([]float32, ctx.numFilters*(ctx.frameLength/2+1))
	for i, filter := range filterBanks {
		copy(flatFilters[i*(ctx.frameLength/2+1):], filter)
	}
	if res := C.cudaMemcpy(ctx.deviceFilters, unsafe.Pointer(&flatFilters[0]), C.size_t(len(flatFilters)*4), C.cudaMemcpyHostToDevice); res != C.cudaSuccess {
		return nil, fmt.Errorf("filtrlar bankini GPU’ga ko‘chirishda xatolik: %v", res)
	}

	blockSize := C.int(256)
	powerGridSize := C.int((ctx.frameLength/2 + 1 + int(blockSize) - 1) / int(blockSize))
	melGridSize := C.int((ctx.numFilters + int(blockSize) - 1) / int(blockSize))
	dctGridSize := C.int((ctx.numCoeffs + int(blockSize) - 1) / int(blockSize))

	for i, frame := range frames {
		if len(frame) != ctx.frameLength {
			return nil, fmt.Errorf("ramka uzunligi mos kelmadi")
		}

		for j := range frame {
			ctx.hostBuffer[j] = frame[j] * window[j]
		}

		if res := C.cudaMemcpy(ctx.deviceFrame, unsafe.Pointer(&ctx.hostBuffer[0]), C.size_t(ctx.frameLength*4), C.cudaMemcpyHostToDevice); res != C.cudaSuccess {
			return nil, fmt.Errorf("ma’lumotni GPU’ga ko‘chirishda xatolik: %v", res)
		}

		if res := C.cufftExecR2C(ctx.plan, (*C.float)(ctx.deviceFrame), (*C.cufftComplex)(ctx.deviceFFT)); res != C.CUFFT_SUCCESS {
			return nil, fmt.Errorf("FFT hisoblashda xatolik: %v", res)
		}

		C.launchPowerSpectrumKernel(
			(*C.cufftComplex)(ctx.deviceFFT),
			(*C.float)(ctx.devicePower),
			C.int(ctx.frameLength),
			powerGridSize,
			blockSize,
			ctx.stream,
		)
		C.cudaStreamSynchronize(ctx.stream)

		C.launchApplyMelFiltersKernel(
			(*C.float)(ctx.devicePower),
			(*C.float)(ctx.deviceFilters),
			(*C.float)(ctx.deviceMel),
			C.int(ctx.numFilters),
			C.int(ctx.frameLength),
			melGridSize,
			blockSize,
			ctx.stream,
		)
		C.cudaStreamSynchronize(ctx.stream)

		C.launchLogKernel(
			(*C.float)(ctx.deviceMel),
			(*C.float)(ctx.deviceLog),
			C.int(ctx.numFilters),
			melGridSize,
			blockSize,
			ctx.stream,
		)
		C.cudaStreamSynchronize(ctx.stream)

		sqrt2OverN := float32(math.Sqrt(2.0 / float64(ctx.numFilters)))
		C.launchDctKernel(
			(*C.float)(ctx.deviceLog),
			(*C.float)(ctx.deviceDCT),
			C.int(ctx.numFilters),
			C.int(ctx.numCoeffs),
			C.float(sqrt2OverN),
			dctGridSize,
			blockSize,
			ctx.stream,
		)
		C.cudaStreamSynchronize(ctx.stream)

		dctResult := make([]float32, ctx.numCoeffs)
		if res := C.cudaMemcpy(unsafe.Pointer(&dctResult[0]), ctx.deviceDCT, C.size_t(ctx.numCoeffs*4), C.cudaMemcpyDeviceToHost); res != C.cudaSuccess {
			return nil, fmt.Errorf("DCT natijasini GPU’dan olishda xatolik: %v", res)
		}
		mfccs[i] = dctResult
	}

	return mfccs, nil
}

// Cleanup - GPU resurslarini ozod qilish
func (ctx *GPUContext) Cleanup() error {
	if ctx.plan != 0 {
		C.cufftDestroy(ctx.plan)
	}
	if unsafe.Pointer(ctx.stream) != nil {
		C.cudaStreamDestroy(ctx.stream)
	}
	if ctx.deviceFrame != nil {
		C.cudaFree(ctx.deviceFrame)
	}
	if ctx.deviceFFT != nil {
		C.cudaFree(ctx.deviceFFT)
	}
	if ctx.devicePower != nil {
		C.cudaFree(ctx.devicePower)
	}
	if ctx.deviceMel != nil {
		C.cudaFree(ctx.deviceMel)
	}
	if ctx.deviceLog != nil {
		C.cudaFree(ctx.deviceLog)
	}
	if ctx.deviceDCT != nil {
		C.cudaFree(ctx.deviceDCT)
	}
	if ctx.deviceFilters != nil {
		C.cudaFree(ctx.deviceFilters)
	}
	return nil
}
