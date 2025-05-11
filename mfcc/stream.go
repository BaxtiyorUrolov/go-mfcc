package mfcc

import (
	"sync"
)

type Streamer struct {
	proc        *Processor
	buffer      []float32
	bufferMutex sync.Mutex
	resultChan  chan []float32
	closeChan   chan struct{}
	wg          sync.WaitGroup
}

func (p *Processor) NewStreamer() *Streamer {
	s := &Streamer{
		proc:       p,
		buffer:     make([]float32, 0, p.config.FrameLength*4),
		resultChan: make(chan []float32, 100),
		closeChan:  make(chan struct{}),
	}

	s.wg.Add(1)
	go s.processLoop()

	return s
}

func (s *Streamer) Write(data []float32) {
	s.bufferMutex.Lock()
	defer s.bufferMutex.Unlock()
	s.buffer = append(s.buffer, data...)
}

func (s *Streamer) Read() []float32 {
	select {
	case mfcc := <-s.resultChan:
		return mfcc
	case <-s.closeChan:
		return nil
	}
}

func (s *Streamer) Close() {
	close(s.closeChan)
	s.wg.Wait()
	close(s.resultChan)
}

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

func (s *Streamer) processAvailableFrames() {
	s.bufferMutex.Lock()
	defer s.bufferMutex.Unlock()

	cfg := s.proc.config
	for len(s.buffer) >= cfg.FrameLength {
		frame := s.buffer[:cfg.FrameLength]
		s.buffer = s.buffer[cfg.HopLength:]

		mfcc := s.proc.computeFrameMFCC(frame)
		s.resultChan <- mfcc
	}
}
