package mfcc

import "math"

// createMelFilterBanks - Mel filtrlar bankini yaratish
func createMelFilterBanks(sampleRate, frameSize, numBanks int) [][]float32 {
	// Hz dan Mel shkalasiga o‘tkazish uchun minimal va maksimal qiymatlarni aniqlaymiz
	minMel := hzToMel(0)                       // Minimal Mel qiymati (0 Hz)
	maxMel := hzToMel(float32(sampleRate) / 2) // Maksimal Mel qiymati (Nyquist chastotasi)

	// Mel nuqtalarini teng oraliqda yaratamiz
	melPoints := linSpace(minMel, maxMel, numBanks+2)

	// Mel nuqtalarini Hz ga qaytarib, chastota nuqtalarini hisoblaymiz
	hzPoints := make([]float32, len(melPoints))
	for i, mel := range melPoints {
		hzPoints[i] = melToHz(mel) // Har bir Mel qiymatini Hz ga o‘tkazamiz
	}

	// Bin nuqtalarini hisoblash (spektr binlari indekslari)
	binPoints := make([]int, len(hzPoints))
	for i, hz := range hzPoints {
		binPoints[i] = int(math.Floor(float64(hz) * float64(frameSize) / float64(sampleRate)))
	}

	// Filter banklarni yaratish (uchburchak filtrlar)
	filters := make([][]float32, numBanks)
	for i := 0; i < numBanks; i++ {
		filters[i] = make([]float32, frameSize/2+1)
		startBin := binPoints[i]
		peakBin := binPoints[i+1]
		endBin := binPoints[i+2]

		// Chap qismdagi uchburchak filtr
		for j := startBin; j <= peakBin; j++ {
			if j < len(filters[i]) {
				filters[i][j] = float32(j-startBin) / float32(peakBin-startBin)
			}
		}

		// O‘ng qismdagi uchburchak filtr
		for j := peakBin; j <= endBin; j++ {
			if j < len(filters[i]) {
				filters[i][j] = float32(endBin-j) / float32(endBin-peakBin)
			}
		}
	}

	return filters // Tayyor filtrlar bankini qaytaramiz
}

// hzToMel - Hz ni Mel shkalasiga o‘tkazish
func hzToMel(hz float32) float32 {
	return 2595 * float32(math.Log10(1+float64(hz)/700)) // Standart Mel formula
}

// melToHz - Mel ni Hz shkalasiga o‘tkazish
func melToHz(mel float32) float32 {
	return 700 * (float32(math.Pow(10, float64(mel)/2595)) - 1) // Teskari Mel formula
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
