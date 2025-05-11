package main

import (
	"encoding/binary"
	"fmt"
	"go-mfcc/mfcc"
	"os"
)

// readWAV - WAV faylni o‘qish (oddiy versiya)
func readWAV(filename string) ([]float32, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	// WAV headerdan ma’lumotlarni o‘qish
	var header [44]byte
	_, err = file.Read(header[:])
	if err != nil {
		return nil, 0, err
	}

	// Sample rate (24-27 baytlar)
	sampleRate := int(binary.LittleEndian.Uint32(header[24:28]))

	// Ma’lumotlar blokini o‘qish
	file.Seek(44, 0)
	data := make([]int16, 0)
	for {
		var sample int16
		err = binary.Read(file, binary.LittleEndian, &sample)
		if err != nil {
			break
		}
		data = append(data, sample)
	}

	// int16 ni float32 ga aylantirish
	audio := make([]float32, len(data))
	for i, sample := range data {
		audio[i] = float32(sample) / 32768.0 // Normalizatsiya
	}

	return audio, sampleRate, nil
}

func main() {
	// Sozlamalarni aniqlash
	cfg := mfcc.Config{
		SampleRate:      16000,
		FrameLength:     512,
		HopLength:       160,
		NumFilters:      26,
		NumCoefficients: 13,
		WindowType:      "hamming",
		UseGPU:          true,
		MaxConcurrency:  4,
		PreEmphasis:     0.97,
		Parallel:        false,
	}

	// WAV faylni o‘qish
	audio, sampleRate, err := readWAV("audio.wav")
	if err != nil {
		fmt.Printf("WAV faylni o‘qishda xatolik: %v\n", err)
		return
	}
	cfg.SampleRate = sampleRate

	// Processor ni yaratish
	processor, err := mfcc.NewProcessor(cfg)
	if err != nil {
		fmt.Printf("Processor yaratishda xatolik: %v\n", err)
		return
	}
	defer processor.Close()

	// MFCC ni hisoblash
	mfccs, err := processor.Process(audio)
	if err != nil {
		fmt.Printf("MFCC hisoblashda xatolik: %v\n", err)
		return
	}

	// Natijalarni chiqarish
	fmt.Println("MFCC natijalari:")
	for i, frame := range mfccs {
		fmt.Printf("Ramka %d: %v\n", i, frame)
	}
}
