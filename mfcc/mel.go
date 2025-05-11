package mfcc

import "math"

// createMelFilterBanks - Mel filtrlar bankini yaratish
func createMelFilterBanks(sampleRate, frameLength, numFilters int, lowFreq, highFreq float32) [][]float32 {
	filterBanks := make([][]float32, numFilters)
	nyquist := float32(sampleRate) / 2.0
	if highFreq == 0 {
		highFreq = nyquist
	}
	// Mel chastotalarini hisoblash
	lowMel := hzToMel(lowFreq)
	highMel := hzToMel(highFreq)
	melPoints := make([]float32, numFilters+2)
	for i := 0; i < len(melPoints); i++ {
		melPoints[i] = lowMel + float32(i)*(highMel-lowMel)/float32(numFilters+1)
	}
	// Mel nuqtalarini Hz ga aylantirish
	hzPoints := make([]float32, len(melPoints))
	for i := range hzPoints {
		hzPoints[i] = melToHz(melPoints[i])
	}
	// Filtr banklarini yaratish
	fftSize := frameLength/2 + 1
	for i := 0; i < numFilters; i++ {
		filterBanks[i] = make([]float32, fftSize)
		for j := 0; j < fftSize; j++ {
			freq := float32(j) * nyquist / float32(fftSize-1)
			if freq >= hzPoints[i] && freq <= hzPoints[i+1] {
				filterBanks[i][j] = (freq - hzPoints[i]) / (hzPoints[i+1] - hzPoints[i])
			} else if freq > hzPoints[i+1] && freq <= hzPoints[i+2] {
				filterBanks[i][j] = (hzPoints[i+2] - freq) / (hzPoints[i+2] - hzPoints[i+1])
			}
		}
	}
	return filterBanks
}

func hzToMel(hz float32) float32 {
	return 2595 * float32(math.Log10(1+float64(hz)/700))
}

func melToHz(mel float32) float32 {
	return 700 * (float32(math.Pow(10, float64(mel)/2595)) - 1)
}

// linSpace - Berilgan oraliqda teng taqsimlangan nuqtalar yaratish
func linSpace(start, end float32, num int) []float32 {
	if num <= 0 {
		return nil
	}

	result := make([]float32, num)
	if num == 1 {
		result[0] = start
		return result
	}

	step := (end - start) / float32(num-1) // Har bir qadamning hajmi
	for i := range result {
		result[i] = start + float32(i)*step // Nuqtalarni hisoblaymiz
	}

	return result
}
