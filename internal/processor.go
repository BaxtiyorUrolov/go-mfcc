package internal

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
)

// Processor - MFCC hisoblash uchun asosiy tuzilma
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
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	filterBanks := createMelFilterBanks(cfg.SampleRate, cfg.FrameLength, cfg.NumFilters, cfg.LowFreq, cfg.HighFreq)
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

// Process - Audio signalni qayta ishlaydi va delta/delta-delta ni hisoblaydi
func (p *Processor) Process(audio []float32) ([][]float32, error) {
	if len(audio) == 0 {
		return nil, errors.New("audio kirishi bo‘sh")
	}

	emphasized := p.applyPreEmphasis(audio)
	frames := p.frameSignal(emphasized)

	var mfccs [][]float32
	var err error

	switch {
	case p.config.UseGPU && p.gpuCtx != nil:
		mfccs, err = p.gpuCtx.ComputeMFCC(frames, p.filterBanks, p.window, p.config)
	case p.config.Parallel:
		mfccs = p.processParallel(frames)
	default:
		mfccs = p.processSequential(frames)
	}

	if err != nil {
		return nil, fmt.Errorf("MFCC hisoblashda xatolik: %w", err)
	}

	// Delta va delta-delta ni butun ramka ketma-ketligi uchun hisoblash
	width := 2 // Delta uchun oynaning kengligi
	deltas := computeDelta(mfccs, width)
	deltaDeltas := computeDeltaDelta(deltas, width)

	// Delta-delta natijalarini qaytarish (masalan, faqat delta-delta)
	return deltaDeltas, nil
}

// processSequential - Sequential tarzda hisoblash
func (p *Processor) processSequential(frames [][]float32) [][]float32 {
	mfccs := make([][]float32, len(frames))
	for i, frame := range frames {
		mfccs[i] = p.computeFrameMFCC(frame)
	}
	return mfccs
}

// processParallel - Parallel tarzda hisoblash
func (p *Processor) processParallel(frames [][]float32) [][]float32 {
	numFrames := len(frames)
	mfccs := make([][]float32, numFrames)

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
				mfccs[j] = p.computeFrameMFCC(frames[j])
			}
		}(start, end)
	}

	wg.Wait()
	return mfccs
}

// computeFrameMFCC - Bitta ramka uchun MFCC ni hisoblash
func (p *Processor) computeFrameMFCC(frame []float32) []float32 {
	// Agar ramka uzunligi mos kelmasa, to‘ldirish
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
	// DCT ni qo‘llash
	mfcc := applyDCT(logMelEnergies, p.config.NumCoefficients, dctBuf)

	return mfcc
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

// padFrame - Ramkani to‘ldirish
func padFrame(frame []float32, length int) []float32 {
	if len(frame) >= length {
		return frame[:length]
	}

	padded := make([]float32, length)
	copy(padded, frame)
	return padded
}

// applyMelFilters - Mel filtrlarini qo‘llash
func applyMelFilters(powerSpectrum []float32, filterBanks [][]float32, melBuf []float32) []float32 {
	for i := range melBuf {
		melBuf[i] = 0
	}

	for i, filter := range filterBanks {
		var energy float32
		for j, val := range filter {
			if j < len(powerSpectrum) {
				energy += val * powerSpectrum[j]
			}
		}
		melBuf[i] = energy
	}
	return melBuf
}

// computeDelta - Delta koeffitsientlarini hisoblash
func computeDelta(mfccs [][]float32, width int) [][]float32 {
	// Agar ramka soni yetarli bo‘lmasa, asl MFCC’larni qaytarish
	if len(mfccs) < 2*width+1 {
		return mfccs
	}

	deltas := make([][]float32, len(mfccs))
	for i := range mfccs {
		deltas[i] = make([]float32, len(mfccs[i]))
	}

	for t := 0; t < len(mfccs); t++ {
		for j := 0; j < len(mfccs[t]); j++ {
			var sum float32
			var count float32
			for w := -width; w <= width; w++ {
				tau := t + w
				if tau >= 0 && tau < len(mfccs) {
					sum += float32(w) * mfccs[tau][j]
					count += float32(math.Abs(float64(w)))
				}
			}
			if count > 0 {
				deltas[t][j] = 2 * sum / (count * float32(width*width))
			}
		}
	}
	return deltas
}

// computeDeltaDelta - Delta-delta koeffitsientlarini hisoblash
func computeDeltaDelta(mfccs [][]float32, width int) [][]float32 {
	// Avval delta’larni hisoblaymiz, keyin ular uchun delta-delta ni hisoblaymiz
	deltas := computeDelta(mfccs, width)
	return computeDelta(deltas, width)
}
