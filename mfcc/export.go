package mfcc

import (
	"encoding/csv"
	"fmt"
	"os"
)

func ExportToCSV(mfccs [][][]float32, labels []string, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	headers := []string{"file", "mfcc_0", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "label"}
	if err := writer.Write(headers); err != nil {
		return err
	}

	for i, mfcc := range mfccs {
		for j, frame := range mfcc {
			record := make([]string, 15)
			record[0] = fmt.Sprintf("file_%d_frame_%d", i, j)
			for k, val := range frame[:13] {
				record[k+1] = fmt.Sprintf("%f", val)
			}
			record[14] = labels[i]
			if err := writer.Write(record); err != nil {
				return err
			}
		}
	}
	writer.Flush()
	return writer.Error()
}
