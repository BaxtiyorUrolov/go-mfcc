package main

import (
	"encoding/binary"
	"fmt"
	"github.com/BaxtiyorUrolov/go-mfcc/mfcc"
	"os"
)

// readWAV - WAV faylni o‘qish
func readWAV(filename string) ([]float32, int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, 0, err
	}
	defer file.Close()

	var header [44]byte
	if _, err := file.Read(header[:]); err != nil {
		return nil, 0, err
	}

	sampleRate := int(binary.LittleEndian.Uint32(header[24:28]))

	file.Seek(44, 0)
	data := make([]int16, 0, 1024) // Boshlang‘ich hajmni optimallashtirish
	for {
		var sample int16
		err = binary.Read(file, binary.LittleEndian, &sample)
		if err != nil {
			break
		}
		data = append(data, sample)
	}

	audio := make([]float32, len(data))
	for i, sample := range data {
		audio[i] = float32(sample) / 32768.0
	}

	return audio, sampleRate, nil
}

func main() {
	cfg := mfcc.DefaultConfig()
	cfg.UseGPU = true
	cfg.MaxConcurrency = 4
	cfg.Parallel = true

	audio, sampleRate, err := readWAV("audio.wav")
	if err != nil {
		fmt.Printf("WAV faylni o‘qishda xatolik: %v\n", err)
		return
	}
	cfg.SampleRate = sampleRate

	processor, err := mfcc.NewProcessor(cfg)
	if err != nil {
		fmt.Printf("Processor yaratishda xatolik: %v\n", err)
		return
	}
	defer processor.Close()

	mfccs, err := processor.Process(audio)
	if err != nil {
		fmt.Printf("MFCC hisoblashda xatolik: %v\n", err)
		return
	}

	fmt.Println("MFCC natijalari:")
	for i, frame := range mfccs[:5] { // Faqat 5 ta ramkani chiqarish
		fmt.Printf("Ramka %d: %v\n", i, frame)
	}
}
