package mfcc

import (
	"encoding/csv"
	"fmt"
	"os"

	"github.com/BaxtiyorUrolov/go-mfcc/internal"
)

// ExportToCSV - Xususiyatlarni CSV faylga eksport qilish
// Bu funksiya model o‘qitish uchun ma’lumotlarni saqlaydi
func ExportToCSV(features [][]internal.FrameFeatures, labels []string, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("CSV faylni yaratishda xatolik: %v", err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	// Sarlavhalar: barcha xususiyatlar va yorliq
	headers := []string{"file_id", "frame_id", "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "zcr", "pitch", "spectral_centroid", "spectral_rolloff", "energy", "label"}
	if err := writer.Write(headers); err != nil {
		return fmt.Errorf("sarlavhalarni yozishda xatolik: %v", err)
	}

	for i, featureSet := range features {
		for j, f := range featureSet {
			record := make([]string, len(headers))
			record[0] = fmt.Sprintf("%d", i) // Fayl ID
			record[1] = fmt.Sprintf("%d", j) // Ramka ID
			for k, val := range f.MFCC[:13] {
				record[k+2] = fmt.Sprintf("%f", val)
			}
			record[15] = fmt.Sprintf("%f", f.ZCR)
			record[16] = fmt.Sprintf("%f", f.Pitch)
			record[17] = fmt.Sprintf("%f", f.SpectralCentroid)
			record[18] = fmt.Sprintf("%f", f.SpectralRollOff)
			record[19] = fmt.Sprintf("%f", f.Energy)
			if i < len(labels) {
				record[20] = labels[i]
			} else {
				record[20] = "unknown"
			}
			if err := writer.Write(record); err != nil {
				return fmt.Errorf("yozuvni yozishda xatolik: %v", err)
			}
		}
	}
	writer.Flush()
	return writer.Error()
}
