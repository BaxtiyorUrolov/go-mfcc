# go-mfcc - MFCC Hisoblash Paketi

`go-mfcc` - bu audio signallardan Mel-Frequency Cepstral Coefficients (MFCC) xususiyatlarini olish uchun mo‘ljallangan Go dasturlash tilidagi paket. Ushbu paket mashinaviy o‘qitish (ML) modellarida audio ma’lumotlar bilan ishlash uchun qulay vosita sifatida ishlatilishi mumkin. Paket katta hajmdagi audio fayllarni (minglab fayllar) samarali qayta ishlash uchun optimallashtirilgan.

## Xususiyatlari
- **CPU va GPU qo‘llab-quvvatlash**: MFCC hisoblashni CPU yoki GPU (CUDA) da amalga oshirish imkoniyati.
- **Parallel hisoblash**: Ko‘p yadroli protsessorlarda tez ishlash uchun parallel goroutinlar.
- **Real vaqtda oqim**: Audio ma’lumotlarini real vaqtda qayta ishlash imkoniyati.
- **Xotira optimallashtirish**: Xotira havzasi (memory pool) orqali xotira ajratishni minimallashtirish.
- **Katta hajmdagi ma’lumotlar bilan ishlash**: Minglab audio fayllarni parallel qayta ishlash imkoniyati (`ProcessBatch` funksiyasi orqali).
- **Moslashuvchan konfiguratsiya**: Sample rate, frame uzunligi, filtrlar soni va boshqa parametrlarni sozlash imkoniyati.

## Loyiha tuzilishi
Loyiha quyidagi fayllardan iborat:
```
go-mfcc/
├── go.mod              # Go moduli fayli
├── kernels.o           # Kompilyatsiya qilingan kernel ob'ekti
└── mfcc/               # MFCC paketi
    ├── config.go       # Konfiguratsiya tuzilmasi va validatsiya
    ├── core.go         # Asosiy protsessor tuzilmasi
    ├── gpu.go          # GPU hisoblashlari
    ├── kernels.cu      # CUDA kernel fayli (GPU hisoblashlari uchun)
    ├── mel.go          # Mel filtrlar bankini yaratish
    ├── memory.go       # Xotira havzasi
    ├── processor.go    # MFCC hisoblash logikasi
    ├── stream.go       # Real vaqtda oqim logikasi
    ├── transform.go    # Transformatsiyalar (FFT, DCT, log)
    └── window.go       # Oyna funksiyalari
```

**Eslatma**: Bu paket faqat kod kutubxonasi sifatida ishlatilishi uchun mo‘ljallangan. Namuna `main.go` fayli loyiha tarkibida yo‘q, foydalanuvchilar o‘z loyihalarida kerakli funksiyalarni chaqirib ishlatishi mumkin.

## O‘rnatish bo‘yicha ko‘rsatmalar (O‘zbek tilida)

### 1. Talablar
Ushbu paketni ishlatish uchun quyidagi dasturlar o‘rnatilgan bo‘lishi kerak:
- **Go 1.20 yoki undan yuqori versiyasi**: Go dasturlash tilini o‘rnating ([rasmiy sayt](https://golang.org/dl/)).
- **CUDA va cuFFT (agar GPU ishlatmoqchi bo‘lsangiz)**: CUDA Toolkit 12.8 va cuFFT kutubxonasi talab qilinadi.
- **nvcc kompilyatori**: CUDA fayllarini kompilyatsiya qilish uchun.

### 2. Paketni o‘rnatish
1. O‘z loyihangizda `go-mfcc` paketini import qilish uchun quyidagi buyruqni bajaring:
   ```bash
   go get github.com/BaxtiyorUrolov/go-mfcc
   ```
   Bu buyruq `go-mfcc` paketini loyihangizga qo‘shadi va qaramliklarni (`github.com/mjibson/go-dsp`) avtomatik yuklaydi.

2. Agar loyiha tuzilishini ko‘rish yoki o‘zgartirish kerak bo‘lsa, loyihani klon qiling:
   ```bash
   git clone https://github.com/BaxtiyorUrolov/go-mfcc.git
   cd go-mfcc
   ```

### 3. CUDA va cuFFT o‘rnatish (GPU uchun)
Agar GPU hisoblashlaridan foydalanmoqchi bo‘lsangiz, quyidagi qadamlarni bajaring:
1. **CUDA Toolkit o‘rnatish**:
    - NVIDIA rasmiy saytidan CUDA Toolkit 12.8 ni yuklab oling: [CUDA Downloads](https://developer.nvidia.com/cuda-downloads).
    - O‘rnatish bo‘yicha ko‘rsatmalarga rioya qiling:
      ```bash
      sudo apt-get install cuda-12-8
      ```
    - O‘rnatilganligini tekshirish uchun:
      ```bash
      nvcc --version
      ```
2. **Muhit o‘zgaruvchilarini sozlash**:
    - CUDA kutubxonalari yo‘lini tizim o‘zgaruvchilariga qo‘shing:
      ```bash
      export PATH=/usr/local/cuda-12.8/bin:$PATH
      export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
      ```
    - Bu o‘zgarishlarni doimiy qilish uchun `~/.bashrc` fayliga qo‘shing:
      ```bash
      echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
      echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
      source ~/.bashrc
      ```
3. **cuFFT kutubxonasini tekshirish**:
    - cuFFT odatda CUDA bilan birga o‘rnatiladi. Uning mavjudligini `/usr/local/cuda-12.8/lib64` ichida `libcufft.so` fayli orqali tekshiring.

### 4. CUDA kernel faylini kompilyatsiya qilish
Agar GPU hisoblashlaridan foydalanmoqchi bo‘lsangiz, `mfcc/kernels.cu` faylini kompilyatsiya qilishingiz kerak:
1. `go-mfcc` loyihasiga o‘ting:
   ```bash
   cd $GOPATH/pkg/mod/github.com/!baxtiyor!urolov/go-mfcc@vX.X.X
   ```
   (Bu yerda `vX.X.X` - o‘rnatilgan versiya, masalan, `v0.0.1`.)
2. `kernels.cu` faylini kompilyatsiya qiling:
   ```bash
   nvcc -c mfcc/kernels.cu -o kernels.o
   ```
3. Kompilyatsiya muvaffaqiyatli bo‘lsa, `kernels.o` fayli loyiha ildizida paydo bo‘ladi.

**Eslatma**: Agar GPU ishlatishni rejalashtirmasangiz, bu qadamni o‘tkazib yuborishingiz mumkin. Bunday holda `Config.UseGPU = false` sozlamasini ishlatish kifoya.

## Foydalanish (O‘zbek tilida)

### 1. Paketni import qilish
O‘z loyihangizda `go-mfcc` paketini quyidagicha import qiling:
```go
import "github.com/BaxtiyorUrolov/go-mfcc/mfcc"
```

### 2. Asosiy foydalanish - Bitta audio fayl uchun
Bitta audio fayldan MFCC xususiyatlarini olish uchun `Process` metodidan foydalaning.

#### Namuna kod:
```go
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
    data := make([]int16, 0, 1024)
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
    // Audio faylni o‘qish
    audio, sampleRate, err := readWAV("path/to/audio.wav")
    if err != nil {
        fmt.Printf("WAV faylni o‘qishda xatolik: %v\n", err)
        return
    }

    // Konfiguratsiyani sozlash
    cfg := mfcc.DefaultConfig()
    cfg.SampleRate = sampleRate
    cfg.UseGPU = true
    cfg.MaxConcurrency = 4
    cfg.Parallel = true

    // Protsessor yaratish
    processor, err := mfcc.NewProcessor(cfg)
    if err != nil {
        fmt.Printf("Processor yaratishda xatolik: %v\n", err)
        return
    }
    defer processor.Close()

    // MFCC hisoblash
    mfccs, err := processor.Process(audio)
    if err != nil {
        fmt.Printf("MFCC hisoblashda xatolik: %v\n", err)
        return
    }

    // Natijalarni chiqarish
    fmt.Println("MFCC natijalari (birinchi 5 ramka):")
    for i, frame := range mfccs[:5] {
        fmt.Printf("Ramka %d: %v\n", i, frame)
    }
}
```

### 3. Katta hajmdagi audio fayllarni qayta ishlash
Agar minglab audio fayllarni qayta ishlash kerak bo‘lsa, `ProcessBatch` metodidan foydalaning. Bu metod bir nechta audio signallarni parallel ravishda qayta ishlaydi.

#### Namuna kod:
```go
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
    data := make([]int16, 0, 1024)
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
    // Bir nechta audio faylni o‘qish
    audioFiles := []string{
        "path/to/audio1.wav",
        "path/to/audio2.wav",
        "path/to/audio3.wav",
        // ... minglab fayllar
    }
    audios := make([][]float32, len(audioFiles))
    var sampleRate int
    for i, file := range audioFiles {
        audio, sr, err := readWAV(file)
        if err != nil {
            fmt.Printf("Fayl %s ni o‘qishda xatolik: %v\n", file, err)
            continue
        }
        audios[i] = audio
        sampleRate = sr // Barcha fayllar bir xil sample rate ga ega bo‘lishi kerak
    }

    // Konfiguratsiyani sozlash
    cfg := mfcc.DefaultConfig()
    cfg.SampleRate = sampleRate
    cfg.UseGPU = true
    cfg.MaxConcurrency = 8 // Protsessor yadrolari soniga qarab sozlashingiz mumkin
    cfg.Parallel = true

    // Protsessor yaratish
    processor, err := mfcc.NewProcessor(cfg)
    if err != nil {
        fmt.Printf("Processor yaratishda xatolik: %v\n", err)
        return
    }
    defer processor.Close()

    // Batch orqali MFCC hisoblash
    batchMFCCs, err := processor.ProcessBatch(audios)
    if err != nil {
        fmt.Printf("Batch MFCC hisoblashda xatolik: %v\n", err)
        return
    }

    // Natijalarni chiqarish (har bir fayl uchun birinchi 5 ramka)
    for i, mfccs := range batchMFCCs {
        if mfccs == nil {
            continue
        }
        fmt.Printf("Fayl %s uchun MFCC natijalari (birinchi 5 ramka):\n", audioFiles[i])
        for j, frame := range mfccs[:5] {
            fmt.Printf("Ramka %d: %v\n", j, frame)
        }
    }
}
```

### 4. Streaming rejimida foydalanish
Agar audio ma’lumotlarini real vaqtda qayta ishlash kerak bo‘lsa, `Streamer` dan foydalaning:
```go
streamer := processor.NewStreamer()
defer streamer.Close()
streamer.Write(audioChunk) // Audio qismini yozish
mfcc := streamer.Read()    // MFCC natijasini olish
```

### 5. Konfiguratsiyani sozlash
`Config` tuzilmasi orqali hisoblash parametrlarni o‘zgartirish mumkin:
- `SampleRate`: Audio sample rate (masalan, 16000 Hz).
- `FrameLength`: Har bir ramkaning uzunligi (standart: 512).
- `HopLength`: Ramkalar orasidagi qadam uzunligi (standart: 256).
- `NumCoefficients`: MFCC koeffitsientlari soni (standart: 13).
- `NumFilters`: Mel filtrlar soni (standart: 26).
- `UseGPU`: GPU hisoblashni yoqish/o‘chirish.
- `Parallel`: Parallel hisoblashni yoqish/o‘chirish.
- `MaxConcurrency`: Parallel hisoblash uchun ishchi goroutinlar soni.

## Xatolarni bartaraf qilish
- **"nvcc topilmadi" xatosi**: CUDA to‘g‘ri o‘rnatilganligini va `nvcc` yo‘lining tizim o‘zgaruvchilarida ekanligini tekshiring.
- **"libcufft.so topilmadi" xatosi**: `LD_LIBRARY_PATH` o‘zgaruvchisida `/usr/local/cuda-12.8/lib64` yo‘li qo‘shilganligini tekshiring.
- **Xotira yetishmovchiligi**: Agar minglab fayllarni qayta ishlayotgan bo‘lsangiz, `MaxConcurrency` ni protsessor yadrolari soniga mos ravishda kamaytiring.

## Litsenziya
Ushbu paket MIT litsenziyasi ostida tarqatiladi. Batafsil ma’lumot uchun `LICENSE` faylini ko‘ring.

## Aloqa
Agar savollar yoki takliflar bo‘lsa, loyiha muallifi bilan quyidagi usullar orqali bog‘laning:
- **Telegram kanali**: [https://t.me/UrolovBaxtiyor](https://t.me/UrolovBaxtiyor)
- **LinkedIn**: [https://www.linkedin.com/in/BaxtiyorUrolov](https://www.linkedin.com/in/BaxtiyorUrolov)
- **GitHub**: [BaxtiyorUrolov](https://github.com/BaxtiyorUrolov)