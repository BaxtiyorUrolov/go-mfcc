package mfcc

// MFCCProcessor - Asosiy tuzilma
type MFCCProcessor struct {
	config      Config
	filterBanks [][]float32
	windowFunc  []float32
	memPool     *MemoryPool
	gpuCtx      GPUContext
}
