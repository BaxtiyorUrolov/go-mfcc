package internal

import (
	"github.com/mjibson/go-dsp/fft"
	"math"
)

// computePowerSpectrum - Power spectrumini hisoblash
// Bu funksiya audio ramkaning chastota spektri quvvatini hisoblaydi, model o‘qitish uchun asosiy xususiyat
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

// applyWindow - Oyna funksiyasini signalga qo‘llash
// Bu funksiya audio ramkasiga oynani (masalan, Hamming yoki Blackman) qo‘llaydi
func applyWindow(frame, window, buffer []float32) {
	if len(frame) != len(window) || len(frame) != len(buffer) {
		return
	}
	for i := range frame {
		buffer[i] = frame[i] * window[i]
	}
}

// applyMelFilters - Mel filtrlarini qo‘llash
// Bu funksiya power spectrumga Mel filtrlarini qo‘llab, energiya qiymatlarini hisoblaydi
func applyMelFilters(powerSpectrum []float32, filterBanks [][]float32, melBuf []float32) []float32 {
	for i := range melBuf {
		melBuf[i] = 0
	}

	for i, filter := range filterBanks {
		var energy float32
		for j, val := range filter {
			if j < len(powerSpectrum) {
				energy += val * powerSpectrum[j]
			}
		}
		melBuf[i] = energy
	}
	return melBuf
}

// applyLog - Logarifmik shkalaga o‘tkazish
// Bu funksiya Mel energiyalarini logarifmik shkalaga aylantiradi, MFCC uchun muhim qadam
func applyLog(values []float32, logBuf []float32) []float32 {
	if len(values) == 0 {
		return nil
	}

	for i, v := range values {
		logBuf[i] = float32(math.Log(float64(v) + 1e-6)) // Kichik epsilon qo‘shish, log 0 ga qarshi himoya
	}

	return logBuf
}

// applyDCT - Diskret Kosinus Transformatsiyasini qo‘llash
// Bu funksiya log Mel energiyalarini MFCC koeffitsientlariga aylantiradi
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

// computeSpectralCentroid - Spectral Centroid ni hisoblash
// Bu funksiya spektrning og‘irlik markazini hisoblaydi, model o‘qitish uchun xususiyat
func computeSpectralCentroid(powerSpectrum []float32, sampleRate float32) float32 {
	if len(powerSpectrum) == 0 {
		return 0
	}

	var sumFreq, sumPower float32
	for i := 0; i < len(powerSpectrum); i++ {
		freq := float32(i) * sampleRate / float32(2*(len(powerSpectrum)-1))
		sumFreq += freq * powerSpectrum[i]
		sumPower += powerSpectrum[i]
	}

	if sumPower == 0 {
		return 0
	}
	return sumFreq / sumPower
}

// computeSpectralRollOff - Spectral Roll-off ni hisoblash
// Bu funksiya spektr energiyasining 85% ni o‘z ichiga olgan chastotani topadi
func computeSpectralRollOff(powerSpectrum []float32, sampleRate float32, rollOffPercent float32) float32 {
	if len(powerSpectrum) == 0 {
		return 0
	}

	// Umumiy energiyani hisoblash
	var totalEnergy float32
	for _, val := range powerSpectrum {
		totalEnergy += val
	}

	// Roll-off chegarasini hisoblash
	threshold := totalEnergy * rollOffPercent
	var cumulativeEnergy float32
	rollOffFreq := float32(0)
	for i := 0; i < len(powerSpectrum); i++ {
		cumulativeEnergy += powerSpectrum[i]
		if cumulativeEnergy >= threshold {
			rollOffFreq = float32(i) * sampleRate / float32(2*(len(powerSpectrum)-1))
			break
		}
	}

	return rollOffFreq
}

// computeEnergy - Ramka energiyasini hisoblash
// Bu funksiya ramkaning umumiy energiyasini hisoblaydi, model uchun foydali xususiyat
func computeEnergy(frame []float32) float32 {
	var energy float32
	for _, val := range frame {
		energy += val * val
	}
	return float32(math.Sqrt(float64(energy / float32(len(frame)))))
}

// computeZCR - Zero-Crossing Rate ni hisoblash
// Bu funksiya ramkaning nol kesishish darajasini hisoblaydi
func computeZCR(frame []float32) float32 {
	var zcr float32
	for i := 1; i < len(frame); i++ {
		if (frame[i-1] < 0 && frame[i] >= 0) || (frame[i-1] >= 0 && frame[i] < 0) {
			zcr++
		}
	}
	return zcr / float32(len(frame)-1)
}

// computePitch - Pitch (fundamental chastota) ni hisoblash
// Bu funksiya autokorrelyatsiya usuli bilan pitch ni hisoblaydi
func computePitch(frame []float32, sampleRate float32) float32 {
	n := len(frame)
	if n < 2 {
		return 0
	}

	// Autokorrelyatsiyani hisoblash
	autocorr := make([]float32, n)
	for lag := 0; lag < n; lag++ {
		var sum float32
		for i := 0; i < n-lag; i++ {
			sum += frame[i] * frame[i+lag]
		}
		autocorr[lag] = sum
	}

	// Maksimal lag ni topish
	minPeriod := int(sampleRate / 400) // 400 Hz - maksimal pitch
	maxPeriod := int(sampleRate / 50)  // 50 Hz - minimal pitch
	if maxPeriod >= n {
		maxPeriod = n - 1
	}
	if minPeriod < 1 {
		minPeriod = 1
	}

	maxCorr := float32(0)
	lagAtMax := 0
	for lag := minPeriod; lag <= maxPeriod; lag++ {
		if autocorr[lag] > maxCorr {
			maxCorr = autocorr[lag]
			lagAtMax = lag
		}
	}

	if lagAtMax == 0 {
		return 0
	}

	return sampleRate / float32(lagAtMax)
}
