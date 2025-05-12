package mfcc

import (
	"sync"
)

// Streamer - Streaming uchun ma’lumotlarni qayta ishlaydi
type Streamer struct {
	proc        *Processor
	buffer      []float32
	bufferMutex sync.Mutex
	resultChan  chan []float32
	closeChan   chan struct{}
	wg          sync.WaitGroup
}

// NewStreamer - Yangi streamer yaratish
func (p *Processor) NewStreamer() *Streamer {
	s := &Streamer{
		proc:       p,
		buffer:     make([]float32, 0, p.config.FrameLength*4),    // Boshlang‘ich hajmni optimallashtirish
		resultChan: make(chan []float32, p.config.MaxConcurrency), // Buffer hajmini maxConcurrency ga moslashtirish
		closeChan:  make(chan struct{}),
	}
	s.wg.Add(1)
	go s.processLoop()
	return s
}

// Write - Ma’lumotlarni buferga yozish
func (s *Streamer) Write(data []float32) {
	s.bufferMutex.Lock()
	defer s.bufferMutex.Unlock()
	s.buffer = append(s.buffer, data...)
}

// Read - Natijani olish
func (s *Streamer) Read() []float32 {
	select {
	case mfcc := <-s.resultChan:
		return mfcc
	case <-s.closeChan:
		return nil
	}
}

// Close - Streamer’ni to‘xtatish
func (s *Streamer) Close() {
	close(s.closeChan)
	s.wg.Wait()
	close(s.resultChan)
}

// processLoop - Doimiy ravishda mavjud ramkalarni qayta ishlaydi
func (s *Streamer) processLoop() {
	defer s.wg.Done()

	for {
		select {
		case <-s.closeChan:
			return
		default:
			s.processAvailableFrames()
		}
	}
}

// processAvailableFrames - Mavjud ramkalarni qayta ishlaydi
func (s *Streamer) processAvailableFrames() {
	s.bufferMutex.Lock()
	defer s.bufferMutex.Unlock()

	cfg := s.proc.config
	for len(s.buffer) >= cfg.FrameLength {
		frame := s.buffer[:cfg.FrameLength]
		s.buffer = s.buffer[cfg.HopLength:]

		mfcc := s.proc.computeFrameMFCC(frame)
		select {
		case s.resultChan <- mfcc: // Non-blocking yozish
		default:
			// Agar kanal to‘la bo‘lsa, o‘tkazib yuboramiz
		}
	}
}
