package mfcc

import "sync"

// MemoryPool - Xotira havzasini boshqaruvchi tuzilma
type MemoryPool struct {
	frameBuffers [][]float32 // Ramka buferlari uchun xotira
	melBuffers   [][]float32 // Mel energiya buferlari uchun xotira
	logBuffers   [][]float32 // Logarifmik buferlar uchun xotira
	dctBuffers   [][]float32 // DCT buferlari uchun xotira
	mu           sync.Mutex  // Bir vaqtning o‘zida kirishni sinxronlash uchun
}

// NewMemoryPool - Yangi xotira havzasini yaratish
func NewMemoryPool(maxConcurrency, frameLength, numFilters, numCoefficients int) *MemoryPool {
	pool := &MemoryPool{
		frameBuffers: make([][]float32, maxConcurrency), // MaxConcurrency soniga mos buferlar
		melBuffers:   make([][]float32, maxConcurrency),
		logBuffers:   make([][]float32, maxConcurrency),
		dctBuffers:   make([][]float32, maxConcurrency),
	}

	for i := 0; i < maxConcurrency; i++ { // Har bir buferni oldindan ajratamiz
		pool.frameBuffers[i] = make([]float32, frameLength)
		pool.melBuffers[i] = make([]float32, numFilters)
		pool.logBuffers[i] = make([]float32, numFilters)
		pool.dctBuffers[i] = make([]float32, numCoefficients)
	}

	return pool
}

// GetFrameBuffer - Ramka buferini olish
func (p *MemoryPool) GetFrameBuffer() []float32 {
	p.mu.Lock() // Sinxronlash uchun qulf
	defer p.mu.Unlock()
	if len(p.frameBuffers) == 0 { // Agar buferlar tugab qolsa, yangi yaratamiz
		return make([]float32, cap(p.frameBuffers[0]))
	}
	buf := p.frameBuffers[0] // Eng birinchi buferni olamiz
	p.frameBuffers = p.frameBuffers[1:]
	return buf
}

// PutFrameBuffer - Ramka buferini qaytarish
func (p *MemoryPool) PutFrameBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.frameBuffers = append(p.frameBuffers, buf) // Buferni qayta havzaga qo‘yamiz
}

// GetMelBuffer - Mel buferini olish
func (p *MemoryPool) GetMelBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.melBuffers) == 0 {
		return make([]float32, cap(p.melBuffers[0]))
	}
	buf := p.melBuffers[0]
	p.melBuffers = p.melBuffers[1:]
	return buf
}

// PutMelBuffer - Mel buferini qaytarish
func (p *MemoryPool) PutMelBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.melBuffers = append(p.melBuffers, buf)
}

// GetLogBuffer - Log buferini olish
func (p *MemoryPool) GetLogBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.logBuffers) == 0 {
		return make([]float32, cap(p.logBuffers[0]))
	}
	buf := p.logBuffers[0]
	p.logBuffers = p.logBuffers[1:]
	return buf
}

// PutLogBuffer - Log buferini qaytarish
func (p *MemoryPool) PutLogBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.logBuffers = append(p.logBuffers, buf)
}

// GetDCTBuffer - DCT buferini olish
func (p *MemoryPool) GetDCTBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.dctBuffers) == 0 {
		return make([]float32, cap(p.dctBuffers[0]))
	}
	buf := p.dctBuffers[0]
	p.dctBuffers = p.dctBuffers[1:]
	return buf
}

// PutDCTBuffer - DCT buferini qaytarish
func (p *MemoryPool) PutDCTBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.dctBuffers = append(p.dctBuffers, buf)
}
