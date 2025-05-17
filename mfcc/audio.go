package mfcc

import (
	"fmt"

	"github.com/DylanMeeus/GoAudio/wave"
)

// LoadAudio WAV faylni o‘qib, float32 formatida qaytaradi
func LoadAudio(filename string) ([]float32, int, error) {
	// WAV faylni dekodlash
	wav, err := wave.ReadWaveFile(filename)
	if err != nil {
		return nil, 0, fmt.Errorf("WAV faylni dekodlashda xatolik: %v", err)
	}

	// Frame'larni float32 formatiga o‘tkazish va normalizatsiya qilish
	data := make([]float32, len(wav.Frames))
	for i, frame := range wav.Frames {
		data[i] = float32(frame) / 32768.0 // 16-bit audio uchun normalizatsiya
	}

	return data, wav.WaveFmt.SampleRate, nil
}
