package internal

import "sync"

// MemoryPool - Xotira havzasi
type MemoryPool struct {
	frameBuffers [][]float32
	melBuffers   [][]float32
	logBuffers   [][]float32
	dctBuffers   [][]float32
	mu           sync.Mutex
}

// NewMemoryPool - Yangi xotira havzasini yaratish
func NewMemoryPool(maxConcurrency, frameLength, numFilters, numCoefficients int) *MemoryPool {
	pool := &MemoryPool{
		frameBuffers: make([][]float32, 0, maxConcurrency),
		melBuffers:   make([][]float32, 0, maxConcurrency),
		logBuffers:   make([][]float32, 0, maxConcurrency),
		dctBuffers:   make([][]float32, 0, maxConcurrency),
	}

	for i := 0; i < maxConcurrency; i++ {
		pool.frameBuffers = append(pool.frameBuffers, make([]float32, frameLength))
		pool.melBuffers = append(pool.melBuffers, make([]float32, numFilters))
		pool.logBuffers = append(pool.logBuffers, make([]float32, numFilters))
		pool.dctBuffers = append(pool.dctBuffers, make([]float32, numCoefficients))
	}

	return pool
}

// GetFrameBuffer - Ramka buferini olish
func (p *MemoryPool) GetFrameBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.frameBuffers) == 0 {
		return make([]float32, cap(p.frameBuffers[0]))
	}
	buf := p.frameBuffers[len(p.frameBuffers)-1]
	p.frameBuffers = p.frameBuffers[:len(p.frameBuffers)-1]
	return buf
}

// PutFrameBuffer - Ramka buferini qaytarish
func (p *MemoryPool) PutFrameBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.frameBuffers = append(p.frameBuffers, buf)
}

// GetMelBuffer - Mel buferini olish
func (p *MemoryPool) GetMelBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if len(p.melBuffers) == 0 {
		return make([]float32, cap(p.melBuffers[0]))
	}
	buf := p.melBuffers[len(p.melBuffers)-1]
	p.melBuffers = p.melBuffers[:len(p.melBuffers)-1]
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
	buf := p.logBuffers[len(p.logBuffers)-1]
	p.logBuffers = p.logBuffers[:len(p.logBuffers)-1]
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
	buf := p.dctBuffers[len(p.dctBuffers)-1]
	p.dctBuffers = p.dctBuffers[:len(p.dctBuffers)-1]
	return buf
}

// PutDCTBuffer - DCT buferini qaytarish
func (p *MemoryPool) PutDCTBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.dctBuffers = append(p.dctBuffers, buf)
}
