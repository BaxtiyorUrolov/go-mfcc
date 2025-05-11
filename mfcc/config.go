package mfcc

import (
	"errors" // Xatolarni boshqarish uchun standart kutubxona
	"fmt"    // Formatlangan chiqish uchun standart kutubxona
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
	MaxConcurrency  int        `json:"max_concurrency"`  // Maksimal parallel ishlaydigan goroutinlar soni
}

// Validate - Konfiguratsiyani tekshirish funksiyasi
func (c *Config) Validate() error {
	if c.SampleRate <= 0 { // Namunalar tezligi musbat bo‘lishi kerak
		return errors.New("sample rate must be positive")
	}
	if c.FrameLength <= 0 { // Ramka uzunligi musbat bo‘lishi kerak
		return errors.New("frame length must be positive")
	}
	if c.HopLength <= 0 { // Qadam uzunligi musbat bo‘lishi kerak
		return errors.New("hop length must be positive")
	}
	if c.NumCoefficients <= 0 { // Koeffitsientlar soni musbat bo‘lishi kerak
		return errors.New("number of coefficients must be positive")
	}
	if c.NumFilters <= 0 { // Filtrlar soni musbat bo‘lishi kerak
		return errors.New("number of filters must be positive")
	}
	if c.PreEmphasis < 0 || c.PreEmphasis >= 1 { // Pre-emphasis [0, 1) oralig‘ida bo‘lishi kerak
		return errors.New("pre-emphasis coefficient must be in [0, 1) range")
	}
	if c.MaxConcurrency < 1 { // Maksimal goroutinlar soni kamida 1 bo‘lishi kerak
		return errors.New("max concurrency must be at least 1")
	}
	return nil
}

// DefaultConfig - Standart konfiguratsiyani qaytarish
func DefaultConfig() Config {
	return Config{
		SampleRate:      16000,   // Standart namunalar tezligi 16 kHz
		FrameLength:     512,     // Standart ramka uzunligi 512 namunalar
		HopLength:       256,     // Standart qadam uzunligi 256 namunalar
		NumCoefficients: 13,      // Standart koeffitsientlar soni 13
		NumFilters:      26,      // Standart filtrlar soni 26
		WindowType:      Hamming, // Standart oyna turi Hamming
		PreEmphasis:     0.97,    // Standart pre-emphasis koeffitsienti 0.97
		Parallel:        true,    // Parallel hisoblash yoqilgan
		MaxConcurrency:  4,       // Maksimal 4 goroutin
	}
}

// String - Konfiguratsiyani matn sifatida ko‘rish
func (c Config) String() string {
	return fmt.Sprintf(
		"MFCC Config: SampleRate=%d, FrameLength=%d, HopLength=%d, NumCoeffs=%d, NumFilters=%d",
		c.SampleRate, c.FrameLength, c.HopLength, c.NumCoefficients, c.NumFilters,
	)
}
