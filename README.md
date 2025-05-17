# go-mfcc - MFCC Hisoblash Paketi

`go-mfcc` — bu audio signallardan **Mel-Frequency Cepstral Coefficients (MFCC)** xususiyatlarini olish uchun mo‘ljallangan Go dasturlash tilidagi kutubxona. Ushbu kutubxona mashinaviy o‘qitish (Machine Learning) modellarida audio ma’lumotlarni qayta ishlash uchun juda qulay vosita bo‘lib, katta hajmdagi audio fayllarni (minglab fayllar) samarali qayta ishlash uchun optimallashtirilgan. Kutubxona CPU va GPU (CUDA orqali) hisoblashni, parallel ishlov berishni va real vaqtda oqimni qo‘llab-quvvatlaydi.

## Xususiyatlari

- **MFCC Hisoblash**: Moslashuvchan parametrlar bilan MFCC xususiyatlarini olish.
- **Oyna Funksiyalari**: Hamming, Hanning, Blackman va boshqa oyna turlarini qo‘llab-quvvatlash.
- **Pre-Emphasis**: Audio signalning yuqori chastotalarini kuchaytirish filtri.
- **Qo‘shimcha Xususiyatlar**: Zero-Crossing Rate (ZCR), Pitch, Spectral Centroid, Spectral Roll-off va Energy.
- **Audio Faylni O‘qish**: `DylanMeeus/GoAudio` yordamida WAV formatdagi audio fayllarni oson o‘qish.
- **GPU Tezlashtirish**: CUDA yordamida GPU’da tezkor hisoblash.
- **Parallel Hisoblash**: Ko‘p yadroli protsessorlarda samarali ishlash.
- **Real Vaqtda Oqim**: Audio ma’lumotlarini real vaqtda qayta ishlash.
- **Xotira Optimallashtirish**: Xotira havzasi orqali samarali xotira boshqaruvi.
- **CSV Eksport**: Hisoblangan xususiyatlarni CSV formatida saqlash (ML datasetlari uchun qulay).

## O‘rnatish

Ushbu kutubxonani o‘rnatish uchun quyidagi qadamlarni bajaring:

1. **Go O‘rnatish**  
   Go 1.18 yoki undan yuqori versiyasini o‘rnating. Uni [rasmiy sayt](https://golang.org/dl/)dan yuklab olishingiz mumkin.

2. **Kutubxonani Yuklab Olish**  
   Terminalda quyidagi buyruqlarni ishga tushuring:
   ```bash
   go get github.com/BaxtiyorUrolov/go-mfcc
   go get github.com/DylanMeeus/GoAudio/wave
   ```

3. **GPU Qo‘llab-Quvvatlash (Ixtiyoriy)**  
   Agar GPU’da hisoblashni xohlasangiz:
   - **CUDA Toolkit** o‘rnating (masalan, 12.8 versiyasi). Yuklab olish: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
   - Muhit o‘zgaruvchilarini sozlang:
     ```bash
     export PATH=/usr/local/cuda-12.8/bin:$PATH
     export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
     ```
   - CUDA kernelni kompilyatsiya qiling:
     ```bash
     nvcc -c internal/kernels.cu -o kernels.o
     ```
   - Eslatma: GPU ishlatish uchun C kompilyatori (masalan, gcc) o‘rnatilgan bo‘lishi kerak.

4. **Oddiy CPU Ishlatish**  
   Agar GPU kerak bo‘lmasa, `Config`da `UseGPU: false` sozlamasini qo‘llang va yuqoridagi CUDA qadamlarini o‘tkazib yuboring.

## Foydalanish Misollari

### 1. Bitta Audio Faylni Qayta Ishlash

`DylanMeeus/GoAudio` yordamida WAV faylni o‘qib, MFCC xususiyatlarini hisoblash:

```go
package main

import (
	"fmt"
	"github.com/BaxtiyorUrolov/go-mfcc/mfcc"
)

func main() {
	// Standart sozlamalarni olish
	cfg := mfcc.DefaultConfig()

	// Protsessor yaratish
	processor, err := mfcc.NewProcessor(cfg)
	if err != nil {
		fmt.Println("Protsessor yaratishda xatolik:", err)
		return
	}
	defer processor.Close()

	// Audio faylni o‘qish
	audio, sampleRate, err := mfcc.LoadAudio("path/to/audio.wav")
	if err != nil {
		fmt.Println("Audio faylni o‘qishda xatolik:", err)
		return
	}

	// SampleRate ni konfiguratsiyaga moslashtirish
	cfg.SampleRate = sampleRate

	// MFCC hisoblash
	mfccs, err := processor.Process(audio)
	if err != nil {
		fmt.Println("MFCC hisoblashda xatolik:", err)
		return
	}

	// Natijalarni chiqarish (birinchi 5 ramka)
	fmt.Println("MFCC natijalari (birinchi 5 ramka):")
	for i, frame := range mfccs[:5] {
		fmt.Printf("Ramka %d: %v\n", i, frame)
	}
}
```

### 2. Bir Necha Audio Fayllarni Parallel Qayta Ishlash

Katta datasetlarni qayta ishlash uchun:

```go
package main

import (
	"fmt"
	"github.com/BaxtiyorUrolov/go-mfcc/mfcc"
)

func main() {
	cfg := mfcc.DefaultConfig()
	cfg.Parallel = true // Parallel ishlashni yoqish
	processor, err := mfcc.NewProcessor(cfg)
	if err != nil {
		fmt.Println("Protsessor yaratishda xatolik:", err)
		return
	}
	defer processor.Close()

	// Bir nechta audio fayllarni o‘qish
	audioFiles := []string{"audio1.wav", "audio2.wav", "audio3.wav"}
	audios := make([][]float32, len(audioFiles))
	for i, file := range audioFiles {
		audio, _, err := mfcc.LoadAudio(file)
		if err != nil {
			fmt.Printf("%s faylni o‘qishda xatolik: %v\n", file, err)
			continue
		}
		audios[i] = audio
	}

	// Batch orqali MFCC hisoblash
	results, err := processor.ProcessBatch(audios)
	if err != nil {
		fmt.Println("Batch qayta ishlashda xatolik:", err)
		return
	}

	// Natijalarni chiqarish
	for i, mfccs := range results {
		fmt.Printf("%s fayl uchun MFCC natijalari (birinchi 5 ramka):\n", audioFiles[i])
		for j, frame := range mfccs[:5] {
			fmt.Printf("Ramka %d: %v\n", j, frame)
		}
	}
}
```

### 3. Real Vaqtda Oqim

Audio oqimini real vaqtda qayta ishlash:

```go
package main

import (
	"fmt"
	"github.com/BaxtiyorUrolov/go-mfcc/mfcc"
)

func main() {
	cfg := mfcc.DefaultConfig()
	processor, err := mfcc.NewProcessor(cfg)
	if err != nil {
		fmt.Println("Protsessor yaratishda xatolik:", err)
		return
	}
	defer processor.Close()

	// Oqim protsessorini yaratish
	streamer := processor.NewStreamer()
	defer streamer.Close()

	// Audio qismini o‘qish va yozish
	audio, _, err := mfcc.LoadAudio("path/to/audio.wav")
	if err != nil {
		fmt.Println("Audio faylni o‘qishda xatolik:", err)
		return
	}
	go func() {
		chunkSize := 256
		for i := 0; i < len(audio); i += chunkSize {
			end := i + chunkSize
			if end > len(audio) {
				end = len(audio)
			}
			chunk := audio[i:end]
			streamer.Write(chunk)
		}
	}()

	// MFCC natijalarini olish
	for {
		mfcc := streamer.Read()
		if mfcc == nil {
			break
		}
		fmt.Println("Real vaqtda MFCC:", mfcc)
	}
}
```

### 4. Xususiyatlarni CSV ga Eksport Qilish

Hisoblangan xususiyatlarni CSV faylga saqlash:

```go
package main

import (
	"fmt"
	"github.com/BaxtiyorUrolov/go-mfcc/mfcc"
	"github.com/BaxtiyorUrolov/go-mfcc/internal"
)

func main() {
	cfg := mfcc.DefaultConfig()
	processor, err := mfcc.NewProcessor(cfg)
	if err != nil {
		fmt.Println("Protsessor yaratishda xatolik:", err)
		return
	}
	defer processor.Close()

	// Audio faylni o‘qish
	audio, _, err := mfcc.LoadAudio("path/to/audio.wav")
	if err != nil {
		fmt.Println("Audio faylni o‘qishda xatolik:", err)
		return
	}

	// Xususiyatlarni hisoblash
	features, err := processor.proc.Process(audio)
	if err != nil {
		fmt.Println("Xususiyatlarni hisoblashda xatolik:", err)
		return
	}

	// Yorliqlar bilan CSV ga eksport qilish
	labels := []string{"sinf1"}
	err = mfcc.ExportToCSV([][]internal.FrameFeatures{features}, labels, "xususiyatlar.csv")
	if err != nil {
		fmt.Println("CSV ga eksport qilishda xatolik:", err)
	}
}
```

## Sozlamalar (Configuration Options)

`Config` tuzilmasi orqali quyidagi parametrlarni moslashtirish mumkin:

- **`SampleRate`**: Audio sampling tezligi (Hz, masalan, 44100).
- **`FrameLength`**: Har bir ramkaning uzunligi (namunalar soni).
- **`HopLength`**: Ramkalar orasidagi qadam uzunligi (overlapni nazorat qiladi).
- **`NumCoefficients`**: Qaytariladigan MFCC koeffitsientlari soni.
- **`NumFilters`**: Mel filtrlar soni.
- **`WindowType`**: Oyna funksiyasi turi ("hamming", "hanning", "blackman", "rect").
- **`PreEmphasis`**: Pre-emphasis koeffitsienti (0.0 dan 1.0 gacha).
- **`UseGPU`**: GPU hisoblashni yoqish/o‘chirish (true/false).
- **`Parallel`**: Parallel hisoblashni yoqish/o‘chirish (true/false).
- **`MaxConcurrency`**: Parallel hisoblash uchun maksimal goroutinlar soni.
- **`LowFreq`**: Mel filtrlar uchun past chastota chegarasi (Hz).
- **`HighFreq`**: Mel filtrlar uchun yuqori chastota chegarasi (Hz).

Standart sozlamalarni olish uchun `mfcc.DefaultConfig()` funksiyasidan foydalaning.

## Loyiha Tuzilishi

```
go-mfcc/
├── go.mod              # Go modul fayli
├── go.sum              # Go dependency fayli
├── internal/           # Ichki modullar
│   ├── config.go       # Sozlamalar logikasi
│   ├── core.go         # Asosiy hisoblash funksiyalari
│   ├── gpu.go          # GPU qo‘llab-quvvatlash
│   ├── kernels.cu      # CUDA kernel kodi
│   ├── mel.go          # Mel filtr logikasi
│   ├── memory.go       # Xotira boshqaruvi
│   ├── processor.go    # Audio qayta ishlash
│   ├── stream.go       # Oqim logikasi
│   ├── transform.go    # Transformatsiya funksiyalari
│   └── window.go       # Oyna funksiyalari
├── kernels.o           # Kompilyatsiya qilingan CUDA kernel
├── mfcc/               # Asosiy paket
│   ├── audio.go        # Audio fayllarni o‘qish funksiyalari (DylanMeeus/GoAudio)
│   ├── export.go       # Eksport funksiyalari (masalan, CSV)
│   ├── mfcc.go         # MFCC hisoblash logikasi
│   └── processor_test.go # Test fayllari
└── README.md           # Ushbu hujjat
```

## Hissadorlik

Agar ushbu loyihaga hissa qo‘shmoqchi bo‘lsangiz:
1. Repozitoriyani fork qiling.
2. Yangi branch yarating (`git checkout -b feature/yangi-xususiyat`).
3. O‘zgartirishlaringizni kiriting va commit qiling.
4. Pull request jo‘nating.

Kodingiz testlardan o‘tganligiga va loyiha kodlash standartlariga mos kelishiga ishonch hosil qiling.

## Aloqa

Agar savollar yoki takliflar bo‘lsa, loyiha muallifi bilan quyidagi usullar orqali bog‘laning:
- **Telegram kanali**: [https://t.me/UrolovBaxtiyor](https://t.me/UrolovBaxtiyor)
- **LinkedIn**: [https://www.linkedin.com/in/BaxtiyorUrolov](https://www.linkedin.com/in/BaxtiyorUrolov)
- **GitHub**: [BaxtiyorUrolov](https://github.com/BaxtiyorUrolov)