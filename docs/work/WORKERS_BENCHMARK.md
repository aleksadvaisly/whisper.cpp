# Hybrid Mode: Porównanie Konfiguracji Workers

## Test Audio
- **Plik:** 20251002_x34-JKlSjgo_Czy_da_si_by_dobrym_RODZICEM_i_jednoczenie_dobrym_.wav
- **Długość:** 28:57 (1737 sekund)
- **Model:** ggml-large-v3-turbo-q5_0
- **Język:** Polski
- **Hardware:** Apple M1

## Wyniki Testów

### Czas Całkowity

| Konfiguracja | Czas | Speedup | vs Baseline | vs Poprzedni |
|--------------|------|---------|-------------|--------------|
| Metal-only | 4:34.61 (274.6s) | 1.00x | baseline | - |
| **Hybrid 3w (2M+1C)** | **3:21.06 (201.0s)** | **1.37x** | **+36.6%** | - |
| Hybrid 5w (3M+2C) | 3:13.36 (193.4s) | 1.42x | +42.0% | +3.9% |

### Encode Time (Paralelizacja)

| Konfiguracja | Encode Time | Runs | Per Run | Redukcja |
|--------------|-------------|------|---------|----------|
| Metal-only | 125.3s | 73 | 1716 ms | baseline |
| **Hybrid 3w (2M+1C)** | **50.8s** | **80** | **635 ms** | **-59%** |
| Hybrid 5w (3M+2C) | 35.8s | 76 | 471 ms | -71% |

### Batch Decode Time (Bottleneck)

| Konfiguracja | Total | Runs | Per Run | Wzrost |
|--------------|-------|------|---------|--------|
| Metal-only | 98.2s | 55363 | 1.77 ms | baseline |
| **Hybrid 3w (2M+1C)** | **373.5s** | **44519** | **8.39 ms** | **+280%** |
| Hybrid 5w (3M+2C) | 660.6s | 46177 | 14.31 ms | +573% |

## Analiza

### ✅ Encode Time - Świetna Paralelizacja
- 3 workers: **-59%** czasu encode
- 5 workers: **-71%** czasu encode
- Prawie liniowa skalowalność

### ⚠️ Batch Decode - Problem Synchronizacji
- Drastyczny wzrost czasu per run przy więcej workers
- Prawdopodobnie: lock contention lub synchronizacja między threads
- Im więcej workers → tym większy overhead

### 💡 Diminishing Returns
- 3w → 5w: tylko **+3.9%** przyspieszenia całkowitego
- 5w: **+2 workers** ale **7x większy** batch decode overhead
- Nie opłaca się dla typowego użytkownika

## Rekomendacja

### 🏆 Domyślnie: 2 Metal + 1 CPU (3 workers)
**Powody:**
- ✅ Solidny **1.37x speedup** (37% szybciej)
- ✅ Najlepszy stosunek wydajność/złożoność
- ✅ Umiarkowany batch decode overhead (+280% ale tolerowalne)
- ✅ Działa dobrze na standardowych maszynach
- ✅ Oszczędza ~73 sekundy na 29-minutowym audio

### 🔧 Opcjonalnie: 3 Metal + 2 CPU (5 workers)
**Kiedy używać:**
- Gdy masz wielordzeniowy CPU (8+ cores)
- Potrzebujesz absolutnie maksymalnej wydajności
- Akceptujesz większy batch decode overhead
- Dla power users

**Jak włączyć:**
```cpp
// src/whisper.cpp:144-145
#define HYBRID_METAL_WORKERS 2  // było 1
#define HYBRID_CPU_WORKERS   2  // było 1
```

## Wnioski

1. **Hybrid mode działa** - solidne przyspieszenie dla długich audio
2. **3 workers jest sweet spot** - najlepszy stosunek korzyści/kosztów
3. **5 workers ma diminishing returns** - marginalnie lepiej ale większa złożoność
4. **Batch decode jest bottleneckiem** - wymaga optymalizacji w przyszłości
5. **DEFINE ułatwiają eksperymentowanie** - jedna zmiana zamiast chodzenia po kodzie

## Następne Kroki Optymalizacji

1. **Zbadać batch decode bottleneck**
   - Profiling pod kątem lock contention
   - Optymalizacja synchronizacji między threads
   
2. **Dynamiczny dobór workers**
   - Wykrywanie liczby rdzeni CPU
   - Automatyczny wybór optymalnej konfiguracji
   
3. **Memory pooling**
   - Zmniejszyć overhead alokacji pamięci
   - Shared buffer dla read-only danych

4. **Adaptacyjna strategia**
   - Krótkie audio (<5min): Metal-only
   - Średnie audio (5-30min): 2M+1C
   - Długie audio (>30min): 3M+2C
