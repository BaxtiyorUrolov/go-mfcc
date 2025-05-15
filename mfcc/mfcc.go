package mfcc

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/BaxtiyorUrolov/go-mfcc/internal"
)

// WindowType - Oyna funksiyalari uchun tiplarni aniqlash
type WindowType string

const (
	Hamming  WindowType = "hamming"     // Hamming oynasi turi
	Hanning  WindowType = "hanning"     // Hanning oynasi turi
	Blackman WindowType = "blackman"    // Blackman oynasi turi
	Rect     WindowType = "rectangular" // To‘rtburchak oynasi turi
)

// Config - MFCC hisoblash konfiguratsiyasi uchun tuzilma
// JSON teglari orqali konfiguratsiyani tashqi fayllardan yuklab olish mumkin
type Config struct {
	SampleRate      int        `json:"sample_rate"`      // Audio namunalar tezligi (Hz)
	FrameLength     int        `json:"frame_length"`     // Har bir ramkaning uzunligi (namunalar soni)
	HopLength       int        `json:"hop_length"`       // Ramkalar orasidagi qadam uzunligi
	NumCoefficients int        `json:"num_coefficients"` // MFCC koeffitsientlari soni
	NumFilters      int        `json:"num_filters"`      // Mel filtrlar banki soni
	WindowType      WindowType `json:"window_type"`      // Ishlatiladigan oyna turi
	PreEmphasis     float32    `json:"pre_emphasis"`     // Pre-emphasis koeffitsienti
	UseGPU          bool       `json:"use_gpu"`          // GPU ishlatishni yoqish/o‘chirish
	Parallel        bool       `json:"parallel"`         // Parallel hisoblashni yoqish/o‘chirish
	MaxConcurrency  int        `json:"max_concurrency"`  // Maksimal parallel goroutinlar soni
	LowFreq         float32    `json:"low_freq"`         // Mel filtrlar uchun past chastota chegarasi (Hz)
	HighFreq        float32    `json:"high_freq"`        // Mel filtrlar uchun yuqori chastota chegarasi (Hz)
}

// Validate - Konfiguratsiyani tekshirish funksiyasi
func (c *Config) Validate() error {
	if c.SampleRate <= 0 {
		return errors.New("sample rate must be positive")
	}
	if c.FrameLength <= 0 {
		return errors.New("frame length must be positive")
	}
	if c.HopLength <= 0 {
		return errors.New("hop length must be positive")
	}
	if c.NumCoefficients <= 0 {
		return errors.New("number of coefficients must be positive")
	}
	if c.NumFilters <= 0 {
		return errors.New("number of filters must be positive")
	}
	if c.PreEmphasis < 0 || c.PreEmphasis >= 1 {
		return errors.New("pre-emphasis coefficient must be in [0, 1) range")
	}
	if c.MaxConcurrency < 1 {
		return errors.New("max concurrency must be at least 1")
	}
	return nil
}

// DefaultConfig - Standart konfiguratsiyani qaytarish
func DefaultConfig() Config {
	return Config{
		SampleRate:      16000,
		FrameLength:     512,
		HopLength:       256,
		NumCoefficients: 13,
		NumFilters:      26,
		WindowType:      Hamming,
		PreEmphasis:     0.97,
		Parallel:        true,
		MaxConcurrency:  4,
	}
}

// String - Konfiguratsiyani matn sifatida ko‘rish
func (c Config) String() string {
	return fmt.Sprintf(
		"MFCC Config: SampleRate=%d, FrameLength=%d, HopLength=%d, NumCoeffs=%d, NumFilters=%d",
		c.SampleRate, c.FrameLength, c.HopLength, c.NumCoefficients, c.NumFilters,
	)
}

// Processor audio xususiyatlarini chiqarish uchun ishlatiladigan MFCC protsessorini ifodalaydi.
type Processor struct {
	proc *internal.Processor
}

// NewProcessor berilgan konfiguratsiya bilan yangi MFCC protsessorini yaratadi.
func NewProcessor(cfg Config) (*Processor, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	internalCfg := internal.Config{
		SampleRate:      cfg.SampleRate,
		FrameLength:     cfg.FrameLength,
		HopLength:       cfg.HopLength,
		NumCoefficients: cfg.NumCoefficients,
		NumFilters:      cfg.NumFilters,
		WindowType:      internal.WindowType(cfg.WindowType),
		PreEmphasis:     cfg.PreEmphasis,
		UseGPU:          cfg.UseGPU,
		Parallel:        cfg.Parallel,
		MaxConcurrency:  cfg.MaxConcurrency,
		LowFreq:         cfg.LowFreq,
		HighFreq:        cfg.HighFreq,
	}
	proc, err := internal.NewProcessor(internalCfg)
	if err != nil {
		return nil, fmt.Errorf("protsessor yaratishda xatolik: %w", err)
	}
	return &Processor{proc: proc}, nil
}

// Process bitta audio signalidan MFCC xususiyatlarini hisoblaydi.
func (p *Processor) Process(audio []float32) ([][]float32, error) {
	if len(audio) == 0 {
		return nil, errors.New("bo‘sh audio kirishi")
	}
	return p.proc.Process(audio)
}

// normalizeMFCC normalizes MFCC coefficients (zero-mean, unit-variance).
func normalizeMFCC(mfccs [][][]float32) [][][]float32 {
	normalized := make([][][]float32, len(mfccs))
	for i := range mfccs {
		normalized[i] = make([][]float32, len(mfccs[i]))
		for j := range mfccs[i] {
			frame := mfccs[i][j]
			mean := float32(0)
			for _, val := range frame {
				mean += val
			}
			mean /= float32(len(frame))

			variance := float32(0)
			for _, val := range frame {
				diff := val - mean
				variance += diff * diff
			}
			variance /= float32(len(frame))
			stdDev := float32(math.Sqrt(float64(variance)))

			normalized[i][j] = make([]float32, len(frame))
			for k, val := range frame {
				if stdDev > 0 {
					normalized[i][j][k] = (val - mean) / stdDev
				} else {
					normalized[i][j][k] = 0
				}
			}
		}
	}
	return normalized
}

// Update ProcessBatch to include normalization
func (p *Processor) ProcessBatch(audios [][]float32) ([][][]float32, error) {
	if len(audios) == 0 {
		return nil, errors.New("bo‘sh audio partiyasi")
	}

	results := make([][][]float32, len(audios))
	numWorkers := p.proc.Config().MaxConcurrency
	chunkSize := (len(audios) + numWorkers - 1) / numWorkers

	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(audios) {
			end = len(audios)
		}
		go func(start, end int) {
			defer wg.Done()
			for j := start; j < end; j++ {
				mfccs, err := p.Process(audios[j])
				if err != nil {
					continue
				}
				results[j] = mfccs
			}
		}(start, end)
	}

	wg.Wait()
	return normalizeMFCC(results), nil // Add normalization here
}

// Close protsessor resurslarini ozod qiladi.
func (p *Processor) Close() error {
	return p.proc.Close()
}

// Streamer real-vaqt audio uchun MFCC hisoblashini boshqaradi.
type Streamer struct {
	s *internal.Streamer
}

// NewStreamer yangi streaming protsessorini yaratadi.
func (p *Processor) NewStreamer() *Streamer {
	return &Streamer{s: p.proc.NewStreamer()}
}

// Write audio namunalarini streamga yozadi.
func (s *Streamer) Write(data []float32) {
	s.s.Write(data)
}

// Read streamdan MFCC xususiyatlarini oladi.
func (s *Streamer) Read() []float32 {
	return s.s.Read()
}

// Close streamerni to‘xtatadi.
func (s *Streamer) Close() {
	s.s.Close()
}
