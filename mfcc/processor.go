package mfcc

import (
	"errors"
	"fmt"
	"runtime" // CPU yadrolari sonini aniqlash uchun
	"sync"
)

// Processor - MFCC hisoblash uchun asosiy tuzilma
type Processor struct {
	config      Config
	filterBanks [][]float32 // Mel filtrlar banki
	window      []float32   // Oyna funksiyasi
	memPool     *MemoryPool // Xotira havzasi
	gpuCtx      *GPUContext // GPU konteksti
	mu          sync.Mutex  // Sinxronlash uchun qulf
}

// NewProcessor - Yangi protsessor yaratish
func NewProcessor(cfg Config) (*Processor, error) {
	if err := cfg.Validate(); err != nil { // Konfiguratsiyani tekshirish
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	filterBanks := createMelFilterBanks(cfg.SampleRate, cfg.FrameLength, cfg.NumFilters, cfg.LowFreq, cfg.HighFreq) // Filtrlar bankini yaratish
	window := createWindow(cfg.FrameLength, cfg.WindowType)                                                         // Oyna funksiyasini yaratish

	var gpuCtx *GPUContext
	var err error
	if cfg.UseGPU {
		if gpuCtx, err = NewGPUContext(cfg.FrameLength, cfg.NumFilters, cfg.NumCoefficients); err != nil { // GPU kontekstini yaratish
			return nil, fmt.Errorf("failed to initialize GPU: %w", err)
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

// Process - Audio signalni qayta ishlaydi va MFCC ni hisoblaydi
func (p *Processor) Process(audio []float32) ([][]float32, error) {
	if len(audio) == 0 { // Agar audio bo‘sh bo‘lsa, xatolik qaytarish
		return nil, errors.New("empty audio input")
	}

	emphasized := p.applyPreEmphasis(audio) // Pre-emphasis qo‘llash
	frames := p.frameSignal(emphasized)     // Signalni ramkalarga bo‘lish

	var mfccs [][]float32
	var err error

	switch {
	case p.config.UseGPU && p.gpuCtx != nil: // GPU ishlatilsa
		mfccs, err = p.gpuCtx.ComputeMFCC(frames, p.filterBanks, p.window, p.config)
	case p.config.Parallel: // Parallel hisoblash
		mfccs = p.processParallel(frames)
	default: // Sequential hisoblash
		mfccs = p.processSequential(frames)
	}

	if err != nil {
		return nil, fmt.Errorf("MFCC computation failed: %w", err)
	}

	return mfccs, nil
}

// processSequential - Sequential tarzda ramkalarni qayta ishlaydi
func (p *Processor) processSequential(frames [][]float32) [][]float32 {
	mfccs := make([][]float32, len(frames))
	for i, frame := range frames {
		mfccs[i] = p.computeFrameMFCC(frame) // Har bir ramka uchun MFCC hisoblash
	}
	return mfccs
}

// processParallel - Parallel tarzda ramkalarni qayta ishlaydi
func (p *Processor) processParallel(frames [][]float32) [][]float32 {
	numFrames := len(frames)
	mfccs := make([][]float32, numFrames)

	// CPU yadrolari soniga qarab goroutinlar sonini aniqlaymiz
	numWorkers := runtime.NumCPU()
	if numWorkers > p.config.MaxConcurrency {
		numWorkers = p.config.MaxConcurrency
	}

	// Har bir worker uchun ramkalar sonini hisoblaymiz
	chunkSize := (numFrames + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	worker := func(start, end int) {
		defer wg.Done()
		for i := start; i < end && i < numFrames; i++ {
			mfccs[i] = p.computeFrameMFCC(frames[i]) // Ramkani qayta ishlaydi
		}
	}

	// Ramkalarni partiyalarga bo‘lib ishlaymiz
	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		go worker(start, end) // Har bir partiyani alohida goroutin bilan ishlaydi
	}

	wg.Wait() // Barcha goroutinlar tugashini kutyapmiz
	return mfccs
}

// computeFrameMFCC - Bitta ramka uchun MFCC ni hisoblash
func (p *Processor) computeFrameMFCC(frame []float32) []float32 {
	if len(frame) != p.config.FrameLength { // Agar ramka uzunligi noto‘g‘ri bo‘lsa, to‘ldirish
		frame = padFrame(frame, p.config.FrameLength)
	}

	// Vaqtinchalik buferlarni xotira havzasidan olamiz
	frameBuf := p.memPool.GetFrameBuffer()
	melBuf := p.memPool.GetMelBuffer()
	logBuf := p.memPool.GetLogBuffer()
	dctBuf := p.memPool.GetDCTBuffer()
	defer p.memPool.PutFrameBuffer(frameBuf) // Ish tugagach qaytarish
	defer p.memPool.PutMelBuffer(melBuf)
	defer p.memPool.PutLogBuffer(logBuf)
	defer p.memPool.PutDCTBuffer(dctBuf)

	applyWindow(frame, p.window, frameBuf)                               // Oyna funksiyasini qo‘llash
	powerSpectrum := computePowerSpectrum(frameBuf)                      // Power spectrumini hisoblash
	melEnergies := applyMelFilters(powerSpectrum, p.filterBanks, melBuf) // Mel filtrlarini qo‘llash
	logMelEnergies := applyLog(melEnergies, logBuf)                      // Logarifm qo‘llash
	mfcc := applyDCT(logMelEnergies, p.config.NumCoefficients, dctBuf)   // DCT qo‘llash

	return mfcc
}

// Close - GPU kontekstini to‘xtatish
func (p *Processor) Close() error {
	if p.gpuCtx != nil {
		return p.gpuCtx.Cleanup() // GPU resurslarini ozod qilish
	}
	return nil
}

// applyPreEmphasis - Pre-emphasis ni signalga qo‘llash
func (p *Processor) applyPreEmphasis(signal []float32) []float32 {
	if p.config.PreEmphasis == 0 { // Agar koeffitsient 0 bo‘lsa, o‘zgartirmaymiz
		return signal
	}

	result := make([]float32, len(signal))
	result[0] = signal[0]

	for i := 1; i < len(signal); i++ { // Pre-emphasis formulasini qo‘llash
		result[i] = signal[i] - p.config.PreEmphasis*signal[i-1]
	}

	return result
}

// frameSignal - Signalni ramkalarga bo‘lish
func (p *Processor) frameSignal(signal []float32) [][]float32 {
	numFrames := 1 + (len(signal)-p.config.FrameLength)/p.config.HopLength // Ramkalar soni
	frames := make([][]float32, numFrames)

	for i := 0; i < numFrames; i++ { // Har bir ramkani ajratish
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
	if len(frame) >= length { // Agar uzunlik yetarli bo‘lsa, kesib olish
		return frame[:length]
	}

	padded := make([]float32, length) // Yangi massiv yaratib, to‘ldiramiz
	copy(padded, frame)
	return padded
}

// applyMelFilters - Mel filtrlarini power spectrumga qo‘llash
func applyMelFilters(powerSpectrum []float32, filterBanks [][]float32, melBuf []float32) []float32 {
	for i := range melBuf { // Buferni nol bilan to‘ldirish
		melBuf[i] = 0
	}

	for i, filter := range filterBanks { // Har bir filtrni qo‘llash
		var energy float32
		for j, val := range filter {
			energy += val * powerSpectrum[j] // Filtr va spectrumni ko‘paytirish
		}
		melBuf[i] = energy
	}

	return melBuf
}
