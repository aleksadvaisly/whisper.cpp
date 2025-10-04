# Whisper.cpp - Logging Architecture Change

**Data:** 2025-10-04
**Typ:** Propozycja zmiany architektury logowania w CLI

---

## Obecny Stan (Odkrycie z Testów)

### Jak Powstały Pliki `.log` Podczas Testów

Podczas testowania hybrid mode użyłem komendy `tee` do przechwytywania pełnego outputu:

```bash
time whisper-cli -m model.bin -l pl -f audio.wav -otxt -of gpu_only 2>&1 | tee gpu_only_run.log | tail -20
```

**Co się dzieje:**
1. `2>&1` - Łączy stderr i stdout w jeden strumień
2. `| tee gpu_only_run.log` - Zapisuje cały output do pliku ORAZ przepuszcza dalej
3. `| tail -20` - Wyświetla tylko ostatnie 20 linii na ekranie

**Wynik:**
- `gpu_only_run.log` (77 KB) - WSZYSTKO: logi + transkrypcja
- `gpu_only.txt` (29 KB) - tylko czysta transkrypcja (dzięki `-otxt -of`)

### Problem z Obecnym Podejściem

```
❌ Logi i output są zmieszane na stdout/stderr
❌ Trzeba używać zewnętrznych narzędzi (tee) do separacji
❌ Brak prostego sposobu na "tylko logi" lub "tylko output"
❌ Output zawiera timestampy i newliny (trudny do parsowania)
```

---

## Nowa Koncepcja: Rozdzielenie Logów od Outputu

### Zasada 1: Logi Domyślnie Wyłączone

**Logi nigdy nie idą razem z tekstem.**

Domyślnie:
```bash
./whisper-cli -m model.bin -f audio.wav
# Output: TYLKO czysta transkrypcja na stdout
# Logi: BRAK
```

### Zasada 2: Logi Trzeba Jawnie Włączyć

**UWAGA: Flagi się NIE wykluczają!** Można włączyć kilka jednocześnie.

| Flaga | Gdzie idą logi | Przykład |
|-------|---------------|----------|
| `--log-file <path>` | Do pliku (ścieżka relatywna lub absolutna) | `--log-file debug.log` |
| `--log-stderr` | Na stderr | `--log-stderr` |
| `--log-stdout` | Na stdout | `--log-stdout` |

**Przykłady kombinacji:**

```bash
# Tylko logi do pliku
./whisper-cli -m model.bin -f audio.wav --log-file transcription.log

# Logi do pliku I na stderr jednocześnie
./whisper-cli -m model.bin -f audio.wav --log-file debug.log --log-stderr

# Logi wszędzie (plik + stderr + stdout)
./whisper-cli -m model.bin -f audio.wav --log-file all.log --log-stderr --log-stdout
```

### Zasada 3: Output Zawsze na stdout (Domyślnie)

**Format Outputu:**
- Join wszystkich segmentów po spacji `" "`
- Smart line wrapping: Entery tylko na pełnym końcu słowa
- **Reguła:** Jeśli następne słowo przekroczy ~1024 znaki, enterujemy PRZED tym słowem

**Przykład:**

```
To jest pierwsza część transkrypcji która zawiera bardzo dużo tekstu i chcemy
żeby był formatowany w sposób czytelny dla człowieka więc łamiemy linie co
około tysiąc znaków ale tylko pomiędzy słowami nigdy w środku słowa bo to
byłoby nieczytelne i trudne do przetwarzania przez inne programy które mogą
chcieć to dalej parsować.
```

**Bez timestampów, bez `[00:00:00.000 --> 00:00:05.000]`**

### Zasada 4: Kontrola Outputu (Opcjonalna)

**UWAGA: Flagi się NIE wykluczają!**

| Flaga | Gdzie idzie output | Format | Przykład |
|-------|-------------------|--------|----------|
| `--out-file <path>` | Do pliku (bez rozszerzenia) | Domyślnie `.txt` (bez timestampów) | `--out-file result` → `result.txt` |
| `--out-stdout` | Na stdout | Smart wrapping | (domyślne zachowanie) |

**Przykłady:**

```bash
# Domyślnie: output na stdout, brak logów
./whisper-cli -m model.bin -f audio.wav

# Output do pliku, logi na stderr
./whisper-cli -m model.bin -f audio.wav --out-file result --log-stderr

# Output NA STDOUT I DO PLIKU jednocześnie
./whisper-cli -m model.bin -f audio.wav --out-file result.txt --out-stdout

# Tylko logi (bez outputu na ekran)
./whisper-cli -m model.bin -f audio.wav --out-file result.txt --log-stderr
```

---

## Formaty Outputu

### Domyślny Format (TXT - bez timestampów)

```bash
./whisper-cli -m model.bin -f audio.wav --out-file result
# Tworzy: result.txt
```

**Zawartość:**
```
To jest transkrypcja audio która jest formatowana jako ciągły tekst z smart
line wrapping co około tysiąc znaków pomiędzy słowami dla czytelności.
```

### Inne Formaty (Istniejące Flagi)

```bash
# SRT (z timestampami)
./whisper-cli -m model.bin -f audio.wav --out-file result -osrt
# Tworzy: result.srt

# VTT (z timestampami)
./whisper-cli -m model.bin -f audio.wav --out-file result -ovtt
# Tworzy: result.vtt

# JSON (z metadanymi)
./whisper-cli -m model.bin -f audio.wav --out-file result -oj
# Tworzy: result.json
```

---

## Porównanie: Przed vs Po

### PRZED (Obecny Stan)

```bash
./whisper-cli -m model.bin -f audio.wav
```

**Output na stdout:**
```
whisper_init_from_file: loading model...
ggml_metal_init: allocating...
[00:00:00.000 --> 00:00:05.000]  To jest pierwsza część
[00:00:05.000 --> 00:00:10.000]  To jest druga część
whisper_print_timings: total time = 123.45 ms
```

❌ Logi zmieszane z transkrypcją
❌ Timestampy w każdej linii
❌ Trudne do parsowania

### PO (Nowa Koncepcja)

```bash
./whisper-cli -m model.bin -f audio.wav
```

**Output na stdout:**
```
To jest pierwsza część To jest druga część
```

✅ Tylko czysta transkrypcja
✅ Bez logów
✅ Smart formatting

**Z logami:**
```bash
./whisper-cli -m model.bin -f audio.wav --log-stderr 2>debug.log
```

**Output na stdout:**
```
To jest pierwsza część To jest druga część
```

**Plik debug.log:**
```
whisper_init_from_file: loading model...
ggml_metal_init: allocating...
whisper_print_timings: total time = 123.45 ms
```

✅ Logi oddzielone
✅ Łatwe przekierowanie
✅ Czytelny output

---

## Przypadki Użycia

### 1. Cicha Transkrypcja (Tylko Wynik)

```bash
./whisper-cli -m model.bin -f audio.wav > result.txt
# Brak logów, tylko czysta transkrypcja w pliku
```

### 2. Debugowanie (Tylko Logi)

```bash
./whisper-cli -m model.bin -f audio.wav --out-file result.txt --log-stderr
# Transkrypcja w result.txt, logi na ekran dla debugowania
```

### 3. Pełny Zapis (Output + Logi)

```bash
./whisper-cli -m model.bin -f audio.wav --out-file result.txt --log-file debug.log
# Transkrypcja w result.txt, logi w debug.log
```

### 4. Pipeline Processing

```bash
./whisper-cli -m model.bin -f audio.wav | jq -R 'split(" ")'
# Czysty output idealny do pipe'owania do innych narzędzi
```

### 5. Hybrid Mode z Logami

```bash
./whisper-cli -m model.bin -f audio.wav --hybrid --log-file hybrid.log
# Output: czysta transkrypcja
# hybrid.log: "Hybrid mode enabled: 58 chunks...", timing stats, etc.
```

---

## Szczegóły Techniczne

### Smart Line Wrapping Algorithm

```
1. Kolekcjonuj słowa z segmentów (bez timestampów)
2. Dodawaj słowa do bieżącej linii
3. Przed dodaniem słowa:
   IF (długość_linii + " " + długość_słowa) > 1024:
       Wypisz bieżącą linię
       Rozpocznij nową linię od tego słowa
   ELSE:
       Dodaj " " + słowo do bieżącej linii
4. Na końcu wypisz ostatnią linię
```

**Przykład:**

```
Input segments:
[00:00:00] "To jest"
[00:00:01] "przykładowa"
[00:00:02] "transkrypcja"

Output (jedna linia):
To jest przykładowa transkrypcja
```

```
Input segments (długi tekst):
[00:00:00] "Lorem ipsum dolor sit amet..." (500 znaków)
[00:00:10] "consectetur adipiscing elit..." (600 znaków)

Output (dwie linie):
Lorem ipsum dolor sit amet...
consectetur adipiscing elit...
```

### Kompatybilność Wsteczna

**Opcja A: Soft Migration (Zalecana)**
- Zachowaj obecne zachowanie jako deprecated
- Dodaj nowe flagi
- W dokumentacji zachęcaj do nowych flag
- Za 6 miesięcy: ustaw nowe zachowanie jako domyślne

**Opcja B: Hard Break**
- Natychmiast zmień domyślne zachowanie
- Dodaj flagę `--legacy-output` dla starego zachowania
- Może zepsuć istniejące skrypty użytkowników

**Rekomendacja:** Opcja A z ostrzeżeniami deprecation.

---

## Implementacja

### Pliki do Modyfikacji

1. `examples/cli/cli.cpp`
   - Dodać nowe flagi: `--log-file`, `--log-stderr`, `--log-stdout`
   - Dodać flagę: `--out-file`, `--out-stdout`
   - Zmienić logikę outputu (smart wrapping)

2. `src/whisper.cpp`
   - Dodać callback do logów z kontrolą docelowego strumienia
   - Oddzielić `whisper_print_timings()` od outputu transkrypcji

3. `include/whisper.h`
   - Dodać struktury dla log callbacks z konfiguracją

### Przykładowa Struktura

```cpp
struct whisper_log_params {
    bool enabled;
    FILE* file;        // NULL jeśli --log-file nie podano
    bool to_stderr;    // true jeśli --log-stderr
    bool to_stdout;    // true jeśli --log-stdout
};

struct whisper_output_params {
    FILE* file;        // NULL jeśli --out-file nie podano
    bool to_stdout;    // true jeśli --out-stdout
    int max_line_length; // 1024 dla smart wrapping
};
```

---

## Migracja dla Użytkowników

### Skrypty Używające `tee`

**Przed:**
```bash
./whisper-cli -m model.bin -f audio.wav 2>&1 | tee full.log | grep -v "whisper_"
```

**Po:**
```bash
./whisper-cli -m model.bin -f audio.wav --log-file full.log
# Output czysty automatycznie, logi w pliku
```

### Skrypty Parsujące Timestampy

**Przed:**
```bash
./whisper-cli -m model.bin -f audio.wav | grep "\[00:" > timestamped.txt
```

**Po (jeśli timestampy potrzebne):**
```bash
./whisper-cli -m model.bin -f audio.wav --out-file result -osrt
# Użyj formatu SRT dla timestampów
```

---

## Korzyści

✅ **Czytelność:** Output nie jest zaśmiecony logami
✅ **Prostota:** Nie trzeba używać `tee`, `grep`, `sed` do filtrowania
✅ **Pipe-friendly:** Czysty output idealny do pipe'owania
✅ **Debugowanie:** Logi łatwo przekierować do osobnego pliku
✅ **Skalowalność:** Różne kombinacje flag dla różnych use-case'ów
✅ **Hybrid Mode:** Logi hybridu nie mieszają się z wynikiem

---

## Status

**Typ:** 📋 Propozycja
**Priorytet:** Medium
**Effort:** ~2-3 dni implementacji + testy
**Breaking Change:** Tak (jeśli hard migration) / Nie (jeśli soft migration)

**Następne Kroki:**
1. Dyskusja o podejściu (soft vs hard migration)
2. Implementacja flag logowania
3. Implementacja smart wrapping
4. Testy z różnymi kombinacjami flag
5. Dokumentacja użytkownika
6. Migration guide dla istniejących skryptów

---

**Autor:** Aleksander
**Data Utworzenia:** 2025-10-04
**Ostatnia Aktualizacja:** 2025-10-04
