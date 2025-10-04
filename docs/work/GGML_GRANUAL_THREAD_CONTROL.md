# GGML Granular Thread Control Plan

## 1. Cel
- Rozdzielić wykonanie Whispera tak, aby encoder (ciężkie operacje MAC) mógł skalować się na GPU/Metal w wielu wątkach.
- Przenieść decoder (sekwencyjne tokeny) na dedykowany wątek/CPU – uniknąć kolejkowania drobnych zadań na GPU.
- Zbadać realny sufit encoder’a; obecnie wyniki maskuje wąskie gardło dekodera.

## 2. Stan obecny
- `whisper_full_with_state()` realizuje cały pipeline (PCM→Mel→Encoder→Decoder) w jednej funkcji (`src/whisper.cpp:6935+`).
- Hybrydowy tryb dzieli audio na chunk’i, ale każdy worker wykonuje **pełny** pipeline, więc dekoder wciąż trafia na ten sam backend Metal (`src/whisper.cpp:8050-8093`).
- Scheduler GGML sam decyduje o backendzie na podstawie priorytetów, brak API do przypięcia fragmentów grafu na CPU (`ggml/src/ggml-backend.cpp:968-1240`).

## 3. Zmiany w whisper.cpp (warstwa aplikacyjna)
1. **Rozcięcie pipeline’u:**
   - Wydzielić `whisper_encode_stage(state, samples_chunk)` kończące na tensorze encoder’a.
   - Wydzielić `whisper_decode_stage(state, encoder_out, ctx)` – konsumuje wynik encoder’a.
   - Zachować logikę VAD i scalania wyników (linie 8000+).
2. **Kolejka między etapami:**
   - Encode workers (GPU) produkują paczki `encoder_out + metadata` do lock-free kolejki.
   - Dedykowany wątek decode (CPU) pobiera paczki i uruchamia `whisper_decode_stage` sekwencyjnie.
3. **Stany i pamięć:**
   - Oddzielne `whisper_state` dla encode (backend GPU) i decode (backend CPU).
   - Dekoder utrzymuje własny KV cache; encode przekazuje tylko wyniki encoder’a.
4. **API:**
   - Nowy wrapper (np. `whisper_full_pipeline()`) zastąpi hybrydę.
   - Zachować kompatybilność z CLI i callbackami.

## 4. Rozszerzenia w GGML
1. **Preferencje backendu:**
   - Dodać hinty typu `ggml_tensor_set_backend_hint(tensor, GGML_BACKEND_DEVICE_CPU)`.
   - Scheduler ma respektować hinty i nie „przeskakiwać” na GPU, jeśli zablokowano CPU.
2. **Wymiana tensorów:**
   - API do kopiowania tensorów między backendami (GPU→CPU) bez pełnej re-inicjalizacji (np. `ggml_backend_tensor_copy_between`).
   - Możliwy mechanizm „external tensor handle” – dzielenie referencji między schedulerami.
3. **Pipeline w GGML (opcjonalnie):**
   - Asynchroniczne uruchamianie subgrafów (`ggml_backend_sched_compute_async`).
   - Mechanizm producer/consumer zarządzający kolejką subgrafów.

## 5. Etapy wdrożenia
1. **Prototyp w whisper.cpp:**
   - Encode na 1 GPU workerze, decode na 1 CPU wątku.
   - Ręcznie kopiować wynik encoder’a do CPU.
   - Zweryfikować, że wynik = `whisper_full_with_state`.
2. **Integracja z GGML:**
   - Dodać hinty backendów i użyć ich w prototypie.
   - Uniknąć ręcznych kopii poprzez API GGML.
3. **Optymalizacja pipeline’u:**
   - Zbadać liczbę wątków encode (4–6?) przy jednym dekoderze.
   - Jeśli decode nie nadąża: rozważyć drugi wątek decode lub batching chunków.

## 6. Testy
- **Funkcjonalne:** porównanie transkrypcji/timestampów z obecnym pipeline’em.
- **Wydajnościowe:** czasy encode/decode, wykorzystanie CPU/GPU, głębokość kolejki.
- **Stabilność:** różne modele (tiny ↔ large), różne języki, tryby beam search/grammar.

## 7. Ryzyka
- Złożoność utrzymania wielu stanów i kolejki.
- Możliwe regresje jakości (błędy przy scalaniu chunków).
- Większe zużycie pamięci (oddzielne stany encode/decode).
- Zmiany w GGML trzeba uzgodnić z maintainerami.

---
**Następny krok:** przygotować prototyp pipeline’u w `whisper.cpp` oraz propozycję zmian w GGML (hinty backendu/kopiowanie). Wspólnie zweryfikować z maintainerami GGML, czy taka koncepcja jest akceptowalna.

## 8. Podpięcie lokalnego forka GGML
Aby testować zmiany w `/Users/aleksander/Documents/projects3/ggml` bez modyfikowania submodułu w repo:

1. **Zmień include w CMake**  
   - Usuń (lub zignoruj) wbudowany katalog `ggml/` w `whisper.cpp`.
   - Na potrzeby developmentu możesz utworzyć symlink:
     ```bash
     mv ggml ggml.orig       # zachowaj kopię
     ln -s /Users/aleksander/Documents/projects3/ggml ggml
     ```
     CMake wciąż wywoła `add_subdirectory(ggml)` – trafi wtedy w lokalny fork.

2. **Alternatywa: tryb WHISPER_USE_SYSTEM_GGML**  
   - Zbuduj i zainstaluj forka GGML:
     ```bash
     cd /Users/aleksander/Documents/projects3/ggml
     cmake -B build -DGGML_METAL=ON # inne flagi wg potrzeby
     cmake --build build --target install
     ```
   - W `whisper.cpp` wołaj CMake z opcją `-DWHISPER_USE_SYSTEM_GGML=ON` oraz wskaż `CMAKE_PREFIX_PATH` na katalog instalacyjny GGML:
     ```bash
     cmake -B build -DWHISPER_USE_SYSTEM_GGML=ON \
           -DCMAKE_PREFIX_PATH="/usr/local/lib/cmake/ggml" # lub katalog z install
     ```
   - CMake użyje wówczas `find_package(ggml)` i nie będzie kompilował wbudowanego submodułu.

> Dla prac eksperymentalnych najprostszy jest symlink – wszystkie zmiany w forkuggml będą od razu widoczne podczas budowy `whisper.cpp`.
