package mfcc

import (
	"github.com/mjibson/go-dsp/fft"
	"math"
)

// computePowerSpectrum - Power spectrumini hisoblash
// Xotira optimallashtirish: vaqtinchalik buferlarni qayta ishlatish
func computePowerSpectrum(frame []float32) []float32 {
	n := len(frame)
	if n == 0 {
		return nil
	}

	// Kompleks signalni tayyorlash
	complexFrame := make([]complex128, n)
	for i, v := range frame {
		complexFrame[i] = complex(float64(v), 0)
	}

	// FFT ni hisoblash
	fftResult := fft.FFT(complexFrame)

	// Power spectrumini hisoblash
	powerSpectrum := make([]float32, n/2+1)
	for i := range powerSpectrum {
		re := real(fftResult[i])
		im := imag(fftResult[i])
		powerSpectrum[i] = float32(re*re + im*im)
	}

	return powerSpectrum
}

// applyDCT - Diskret kosinus transformatsiyasini qo‘llash
func applyDCT(logMelEnergies []float32, numCoeffs int, dctBuf []float32) []float32 {
	n := len(logMelEnergies)
	if n == 0 || numCoeffs <= 0 {
		return nil
	}

	sqrt2OverN := float32(math.Sqrt(2.0 / float64(n)))

	for k := 0; k < numCoeffs; k++ {
		var sum float32
		for m, val := range logMelEnergies {
			angle := math.Pi * float64(k) * (float64(m) + 0.5) / float64(n)
			sum += val * float32(math.Cos(angle))
		}
		if k == 0 {
			dctBuf[k] = sum * float32(math.Sqrt(1.0/float64(n)))
		} else {
			dctBuf[k] = sum * sqrt2OverN
		}
	}

	return dctBuf[:numCoeffs]
}

// applyLog - Logarifmik shkalaga o‘tkazish
func applyLog(values []float32, logBuf []float32) []float32 {
	if len(values) == 0 {
		return nil
	}

	for i, v := range values {
		logBuf[i] = float32(math.Log(float64(v) + 1e-6))
	}

	return logBuf
}
