#!/bin/bash

WHISPER_BIN="/Users/aleksander/Documents/projects3/whisper.cpp/build/bin/whisper-cli"
MODEL_PATH="/Users/aleksander/Documents/projects3/whisper.cpp/models/ggml-large-v3-turbo-q5_0.bin"
TARGET_DIR="/Users/aleksander/Documents/projects3/youfetch/docs/work/marketing"
TEMP_DIR="/tmp/whisper_batch_$$"

if [ ! -f "$WHISPER_BIN" ]; then
    echo "ERROR: whisper-cli binary not found at $WHISPER_BIN"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file not found at $MODEL_PATH"
    exit 1
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Target directory not found at $TARGET_DIR"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "ERROR: ffmpeg not found. Please install ffmpeg."
    exit 1
fi

mkdir -p "$TEMP_DIR"
trap "rm -rf $TEMP_DIR" EXIT

processed=0
skipped=0
failed=0
current=0

total_files=$(ls -1 "$TARGET_DIR"/*.mp3 2>/dev/null | wc -l | tr -d ' ')

echo "Starting batch transcription..."
echo "Target directory: $TARGET_DIR"
echo "Total MP3 files: $total_files"
echo ""

for mp3_file in "$TARGET_DIR"/*.mp3; do
    if [ ! -f "$mp3_file" ]; then
        continue
    fi

    ((current++))

    base_name=$(basename "$mp3_file" .mp3)
    txt_file="$TARGET_DIR/${base_name}.txt"

    if [ -s "$txt_file" ]; then
        echo "[SKIP] [$current/$total_files] $base_name - TXT already exists and is not empty"
        ((skipped++))
        continue
    fi

    if [ ! -s "$mp3_file" ]; then
        echo "[SKIP] [$current/$total_files] $base_name - MP3 file is empty"
        ((skipped++))
        continue
    fi

    file_size_mb=$(echo "scale=1; $(stat -f %z "$mp3_file") / 1048576" | bc)
    duration_sec=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$mp3_file" 2>/dev/null | cut -d. -f1)
    duration_min=$(echo "scale=1; $duration_sec / 60" | bc)
    echo "[PROCESSING] [$current/$total_files] $base_name (${file_size_mb}MB, ${duration_min}min)"
    start_time=$(date +%s)

    wav_file="$TEMP_DIR/${base_name}.wav"

    if ! ffmpeg -i "$mp3_file" -ar 16000 -ac 1 -c:a pcm_s16le "$wav_file" -y 2>&1 | grep -q "Output"; then
        echo "[FAILED] $base_name - MP3 to WAV conversion failed"
        ((failed++))
        echo ""
        continue
    fi

    if "$WHISPER_BIN" -m "$MODEL_PATH" -f "$wav_file" -l auto -otxt -of "$TARGET_DIR/$base_name" 2>&1 | tail -5; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        minutes=$(echo "scale=2; $duration / 60" | bc)

        if [ -f "$txt_file" ]; then
            echo "[SUCCESS] [$current/$total_files] $base_name - Transcribed in ${minutes} minutes"
            ((processed++))
        else
            echo "[FAILED] [$current/$total_files] $base_name - Transcription completed but TXT file not found"
            ((failed++))
        fi
    else
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        minutes=$(echo "scale=2; $duration / 60" | bc)
        echo "[FAILED] [$current/$total_files] $base_name - Transcription error (${minutes} minutes)"
        ((failed++))
    fi

    rm -f "$wav_file"
    echo ""
done

echo "=========================================="
echo "Batch transcription completed"
echo "Processed: $processed"
echo "Skipped: $skipped"
echo "Failed: $failed"
echo "=========================================="
