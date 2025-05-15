package internal

import (
	"fmt"
)

// MFCCProcessor - Asosiy tuzilma, protsessor va xotira boshqaruvi
type MFCCProcessor struct {
	config      Config
	filterBanks [][]float32 // Mel filtrlar banki
	windowFunc  []float32   // Oyna funksiyasi
	memPool     *MemoryPool // Xotira havzasi
	gpuCtx      *GPUContext // GPU konteksti (agar ishlatilsa)
}

// NewMFCCProcessor - Yangi protsessor yaratish
func NewMFCCProcessor(cfg Config) (*MFCCProcessor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %v", err)
	}

	// Filtrlar bankini oldindan yaratish
	filterBanks := createMelFilterBanks(cfg.SampleRate, cfg.FrameLength, cfg.NumFilters, cfg.LowFreq, cfg.HighFreq)
	windowFunc := createWindow(cfg.FrameLength, cfg.WindowType)

	var gpuCtx *GPUContext
	var err error
	if cfg.UseGPU {
		gpuCtx, err = NewGPUContext(cfg.FrameLength, cfg.NumFilters, cfg.NumCoefficients)
		if err != nil {
			return nil, fmt.Errorf("GPU kontekstini yaratishda xatolik: %v", err)
		}
	}

	return &MFCCProcessor{
		config:      cfg,
		filterBanks: filterBanks,
		windowFunc:  windowFunc,
		memPool:     NewMemoryPool(cfg.MaxConcurrency, cfg.FrameLength, cfg.NumFilters, cfg.NumCoefficients),
		gpuCtx:      gpuCtx,
	}, nil
}

// Close - Resurslarni ozod qilish
func (p *MFCCProcessor) Close() error {
	if p.gpuCtx != nil {
		return p.gpuCtx.Cleanup()
	}
	return nil
}
