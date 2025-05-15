package internal

import "sync"

// MemoryPool - Xotira havzasi, buferlarni qayta ishlatish uchun ishlatiladi
// Bu xotira optimallashtirish uchun yordam beradi
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
		frameBuffers: make([][]float32, maxConcurrency),
		melBuffers:   make([][]float32, maxConcurrency),
		logBuffers:   make([][]float32, maxConcurrency),
		dctBuffers:   make([][]float32, maxConcurrency),
	}

	for i := 0; i < maxConcurrency; i++ {
		pool.frameBuffers[i] = make([]float32, frameLength)
		pool.melBuffers[i] = make([]float32, numFilters)
		pool.logBuffers[i] = make([]float32, numFilters)
		pool.dctBuffers[i] = make([]float32, numCoefficients)
	}

	return pool
}

// GetFrameBuffer - Frame buferini olish
func (p *MemoryPool) GetFrameBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, buf := range p.frameBuffers {
		if len(buf) > 0 {
			p.frameBuffers[i] = nil
			return buf
		}
	}
	// Agar barcha buferlar ishlatilgan bo‘lsa, yangi yaratish
	return make([]float32, 0)
}

// PutFrameBuffer - Frame buferini qaytarish
func (p *MemoryPool) PutFrameBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range p.frameBuffers {
		if p.frameBuffers[i] == nil {
			p.frameBuffers[i] = buf
			return
		}
	}
}

// GetMelBuffer - Mel buferini olish
func (p *MemoryPool) GetMelBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, buf := range p.melBuffers {
		if len(buf) > 0 {
			p.melBuffers[i] = nil
			return buf
		}
	}
	return make([]float32, 0)
}

// PutMelBuffer - Mel buferini qaytarish
func (p *MemoryPool) PutMelBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range p.melBuffers {
		if p.melBuffers[i] == nil {
			p.melBuffers[i] = buf
			return
		}
	}
}

// GetLogBuffer - Log buferini olish
func (p *MemoryPool) GetLogBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, buf := range p.logBuffers {
		if len(buf) > 0 {
			p.logBuffers[i] = nil
			return buf
		}
	}
	return make([]float32, 0)
}

// PutLogBuffer - Log buferini qaytarish
func (p *MemoryPool) PutLogBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range p.logBuffers {
		if p.logBuffers[i] == nil {
			p.logBuffers[i] = buf
			return
		}
	}
}

// GetDCTBuffer - DCT buferini olish
func (p *MemoryPool) GetDCTBuffer() []float32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i, buf := range p.dctBuffers {
		if len(buf) > 0 {
			p.dctBuffers[i] = nil
			return buf
		}
	}
	return make([]float32, 0)
}

// PutDCTBuffer - DCT buferini qaytarish
func (p *MemoryPool) PutDCTBuffer(buf []float32) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for i := range p.dctBuffers {
		if p.dctBuffers[i] == nil {
			p.dctBuffers[i] = buf
			return
		}
	}
}
