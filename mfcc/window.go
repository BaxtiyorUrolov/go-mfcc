package mfcc

import "math"

// createWindow - Oyna funksiyasini yaratish
// Xotira optimallashtirish: har safar yangi massiv o‘rniga qayta ishlatish mumkin
func createWindow(length int, wType WindowType) []float32 {
	window := make([]float32, length)

	switch wType {
	case Hamming: // Hamming oynasi
		for i := range window {
			window[i] = float32(0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(length-1)))
		}
	case Hanning: // Hanning oynasi
		for i := range window {
			window[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(length-1))))
		}
	case Blackman: // Blackman oynasi
		for i := range window {
			window[i] = float32(0.42 - 0.5*math.Cos(2*math.Pi*float64(i)/float64(length-1)) +
				0.08*math.Cos(4*math.Pi*float64(i)/float64(length-1)))
		}
	case Rect: // To‘rtburchak oynasi
		for i := range window {
			window[i] = 1.0
		}
	default: // Noma’lum tur uchun to‘rtburchak oynasi
		for i := range window {
			window[i] = 1.0
		}
	}

	return window
}

// applyWindow - Oyna funksiyasini signalga qo‘llash
// Xotira optimallashtirish: buffer ni tashqi havzadan olish mumkin
func applyWindow(frame, window, buffer []float32) {
	if len(frame) != len(window) || len(frame) != len(buffer) {
		return
	}
	for i := range frame {
		buffer[i] = frame[i] * window[i]
	}
}
