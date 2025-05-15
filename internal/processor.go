package internal

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
)

// FrameFeatures - Har bir ramka uchun chiqariladigan xususiyatlarni saqlash uchun tuzilma
// Bu tuzilma model o‘qitish uchun barcha xususiyatlarni jamlaydi
type FrameFeatures struct {
	MFCC             []float32 // MFCC koeffitsientlari
	ZCR              float32   // Zero-Crossing Rate
	Pitch            float32   // Fundamental chastota
	SpectralCentroid float32   // Spectral Centroid
	SpectralRollOff  float32   // Spectral Roll-off
	Energy           float32   // Ramka energiyasi
}

// Processor - Audio xususiyatlarini hisoblash uchun asosiy tuzilma
type Processor struct {
	config      Config
	filterBanks [][]float32
	window      []float32
	memPool     *MemoryPool
	gpuCtx      *GPUContext
	mu          sync.Mutex
}

// NewProcessor - Yangi protsessor yaratish
func NewProcessor(cfg Config) (*Processor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, fmt.Errorf("konfiguratsiyada xatolik: %w", err)
	}

	// Mel filtrlarini yaratish
	filterBanks := createMelFilterBanks(cfg.SampleRate, cfg.FrameLength, cfg.NumFilters, cfg.LowFreq, cfg.HighFreq)
	// Oyna funksiyasini yaratish
	window := createWindow(cfg.FrameLength, cfg.WindowType)

	var gpuCtx *GPUContext
	var err error
	if cfg.UseGPU {
		gpuCtx, err = NewGPUContext(cfg.FrameLength, cfg.NumFilters, cfg.NumCoefficients)
		if err != nil {
			return nil, fmt.Errorf("GPU kontekstini yaratishda xatolik: %w", err)
		}
	}

	return &Processor{
		config:      cfg,
		filterBanks: filterBanks,
		window:      window,
		memPool:     NewMemoryPool(cfg.MaxConcurrency, cfg.FrameLength, cfg.NumFilters, cfg.NumCoefficients),
		gpuCtx:      gpuCtx,
	}, nil
}

// Config - Protsessor konfiguratsiyasini qaytaradi
func (p *Processor) Config() Config {
	return p.config
}

// Process - Audio signalni qayta ishlaydi va barcha xususiyatlarni hisoblaydi
func (p *Processor) Process(audio []float32) ([]FrameFeatures, error) {
	if len(audio) == 0 {
		return nil, errors.New("audio kirishi bo‘sh")
	}

	// Pre-emphasis qo‘llash
	emphasized := p.applyPreEmphasis(audio)
	// Signalni ramkalarga bo‘lish
	frames := p.frameSignal(emphasized)

	var features []FrameFeatures
	var err error

	switch {
	case p.config.UseGPU && p.gpuCtx != nil:
		// GPU’da faqat MFCC hisoblanadi, qolgan xususiyatlar CPU’da
		mfccs, err := p.gpuCtx.ComputeMFCC(frames, p.filterBanks, p.window, p.config)
		if err != nil {
			return nil, fmt.Errorf("GPU’da MFCC hisoblashda xatolik: %w", err)
		}
		features = make([]FrameFeatures, len(frames))
		for i, frame := range frames {
			powerSpectrum := computePowerSpectrum(frame)
			features[i] = FrameFeatures{
				MFCC:             mfccs[i],
				ZCR:              computeZCR(frame),
				Pitch:            computePitch(frame, float32(p.config.SampleRate)),
				SpectralCentroid: computeSpectralCentroid(powerSpectrum, float32(p.config.SampleRate)),
				SpectralRollOff:  computeSpectralRollOff(powerSpectrum, float32(p.config.SampleRate), 0.85),
				Energy:           computeEnergy(frame),
			}
		}
	case p.config.Parallel:
		features = p.processParallel(frames)
	default:
		features = p.processSequential(frames)
	}

	if err != nil {
		return nil, fmt.Errorf("xususiyatlarni hisoblashda xatolik: %w", err)
	}

	return features, nil
}

// processSequential - Sequential tarzda xususiyatlarni hisoblash
func (p *Processor) processSequential(frames [][]float32) []FrameFeatures {
	features := make([]FrameFeatures, len(frames))
	for i, frame := range frames {
		features[i] = p.computeFrameFeatures(frame)
	}
	return features
}

// processParallel - Parallel tarzda xususiyatlarni hisoblash
func (p *Processor) processParallel(frames [][]float32) []FrameFeatures {
	numFrames := len(frames)
	features := make([]FrameFeatures, numFrames)

	numWorkers := runtime.NumCPU()
	if numWorkers > p.config.MaxConcurrency {
		numWorkers = p.config.MaxConcurrency
	}
	chunkSize := (numFrames + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end && j < numFrames; j++ {
				features[j] = p.computeFrameFeatures(frames[j])
			}
		}(start, end)
	}

	wg.Wait()
	return features
}

// computeFrameFeatures - Bitta ramka uchun barcha xususiyatlarni hisoblash
func (p *Processor) computeFrameFeatures(frame []float32) FrameFeatures {
	// Ramka uzunligini tekshirish va to‘ldirish
	if len(frame) != p.config.FrameLength {
		frame = padFrame(frame, p.config.FrameLength)
	}

	// Xotira havzasidan buferlarni olish
	frameBuf := p.memPool.GetFrameBuffer()
	melBuf := p.memPool.GetMelBuffer()
	logBuf := p.memPool.GetLogBuffer()
	dctBuf := p.memPool.GetDCTBuffer()
	defer p.memPool.PutFrameBuffer(frameBuf)
	defer p.memPool.PutMelBuffer(melBuf)
	defer p.memPool.PutLogBuffer(logBuf)
	defer p.memPool.PutDCTBuffer(dctBuf)

	// Oyna funksiyasini qo‘llash
	applyWindow(frame, p.window, frameBuf)
	// Power spectrumini hisoblash
	powerSpectrum := computePowerSpectrum(frameBuf)
	// Mel energiyalarini hisoblash
	melEnergies := applyMelFilters(powerSpectrum, p.filterBanks, melBuf)
	// Logarifmik shkalaga o‘tkazish
	logMelEnergies := applyLog(melEnergies, logBuf)
	// DCT ni qo‘llash va MFCC chiqarish
	mfcc := applyDCT(logMelEnergies, p.config.NumCoefficients, dctBuf)

	// Barcha qo‘shimcha xususiyatlarni hisoblash
	return FrameFeatures{
		MFCC:             mfcc,
		ZCR:              computeZCR(frame),
		Pitch:            computePitch(frame, float32(p.config.SampleRate)),
		SpectralCentroid: computeSpectralCentroid(powerSpectrum, float32(p.config.SampleRate)),
		SpectralRollOff:  computeSpectralRollOff(powerSpectrum, float32(p.config.SampleRate), 0.85),
		Energy:           computeEnergy(frame),
	}
}

// Close - Resurslarni ozod qilish
func (p *Processor) Close() error {
	if p.gpuCtx != nil {
		return p.gpuCtx.Cleanup()
	}
	return nil
}

// applyPreEmphasis - Pre-emphasis ni qo‘llash
func (p *Processor) applyPreEmphasis(signal []float32) []float32 {
	if p.config.PreEmphasis == 0 {
		return signal
	}

	result := make([]float32, len(signal))
	result[0] = signal[0]
	for i := 1; i < len(signal); i++ {
		result[i] = signal[i] - p.config.PreEmphasis*signal[i-1]
	}
	return result
}

// frameSignal - Signalni ramkalarga bo‘lish
func (p *Processor) frameSignal(signal []float32) [][]float32 {
	numFrames := 1 + (len(signal)-p.config.FrameLength)/p.config.HopLength
	frames := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * p.config.HopLength
		end := start + p.config.FrameLength
		if end > len(signal) {
			end = len(signal)
		}
		frames[i] = signal[start:end]
	}
	return frames
}

// padFrame - Ramkani kerakli uzunlikka to‘ldirish
func padFrame(frame []float32, length int) []float32 {
	if len(frame) >= length {
		return frame[:length]
	}

	padded := make([]float32, length)
	copy(padded, frame)
	for i := len(frame); i < length; i++ {
		padded[i] = 0 // Nollar bilan to‘ldirish
	}
	return padded
}
