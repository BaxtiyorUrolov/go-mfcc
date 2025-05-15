package mfcc

import (
	"testing"
)

func TestNewProcessor(t *testing.T) {
	cfg := DefaultConfig()
	processor, err := NewProcessor(cfg)
	if err != nil {
		t.Fatalf("NewProcessor xatolik: %v", err)
	}
	if processor == nil {
		t.Fatal("NewProcessor null qaytardi")
	}
	if err := processor.Close(); err != nil {
		t.Fatalf("Close xatolik: %v", err)
	}
}

func TestProcess(t *testing.T) {
	cfg := DefaultConfig()
	processor, err := NewProcessor(cfg)
	if err != nil {
		t.Fatalf("NewProcessor xatolik: %v", err)
	}
	defer processor.Close()

	audio := make([]float32, cfg.FrameLength)
	for i := range audio {
		audio[i] = float32(i) / float32(cfg.FrameLength)
	}

	mfccs, err := processor.Process(audio)
	if err != nil {
		t.Fatalf("Process xatolik: %v", err)
	}
	if len(mfccs) == 0 {
		t.Fatal("Process bo‘sh MFCC qaytardi")
	}
}

func BenchmarkProcess(b *testing.B) {
	cfg := DefaultConfig()
	processor, err := NewProcessor(cfg)
	if err != nil {
		b.Fatalf("NewProcessor xatolik: %v", err)
	}
	defer processor.Close()

	audio := make([]float32, cfg.FrameLength)
	for i := range audio {
		audio[i] = float32(i) / float32(cfg.FrameLength)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := processor.Process(audio)
		if err != nil {
			b.Fatalf("Process xatolik: %v", err)
		}
	}
}

func BenchmarkProcessBatch(b *testing.B) {
	cfg := DefaultConfig()
	cfg.MaxConcurrency = 4
	cfg.Parallel = true
	processor, err := NewProcessor(cfg)
	if err != nil {
		b.Fatalf("NewProcessor xatolik: %v", err)
	}
	defer processor.Close()

	audios := make([][]float32, 100)
	for i := range audios {
		audios[i] = make([]float32, cfg.FrameLength)
		for j := range audios[i] {
			audios[i][j] = float32(j) / float32(cfg.FrameLength)
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := processor.ProcessBatch(audios)
		if err != nil {
			b.Fatalf("ProcessBatch xatolik: %v", err)
		}
	}
}
