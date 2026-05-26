#!/usr/bin/env bash
set -euo pipefail

# Edit the variables below for each run.
# MODE: 1 = upload, 0 = download
MODE=0
# Absolute file path to upload or to derive the download filename.
FILE_PATH="/data1/linzengrong/Code/DiffSynth-Studio/Ckpt/3_15_real_robot_history1_480p/epoch-149/epoch-149.safetensors"
# Fixed root directory for downloads (only used when MODE=0).
DOWNLOAD_ROOT="/data1/linzengrong/Code/DiffSynth-Studio/Ckpt"
# Delay before upload or download. Set both to 0 to skip.
DELAY_HOURS=0
DELAY_MINUTES=0

FILE_NAME="$(basename "$FILE_PATH")"
TARGET_DIR=""

mode_label() {
  case "$MODE" in
    1) echo "upload" ;;
    0) echo "download" ;;
    *) echo "unknown" ;;
  esac
}

print_run_info() {
  echo "MODE=${MODE} ($(mode_label))"
  case "$MODE" in
    1)
      echo "FILE_PATH=${FILE_PATH}"
      ;;
    0)
      echo "FILE_PATH=${FILE_PATH}"
      echo "DOWNLOAD_ROOT=${DOWNLOAD_ROOT}"
      echo "TARGET_DIR=${TARGET_DIR}"
      ;;
    *)
      ;;
  esac
  echo "DELAY_HOURS=${DELAY_HOURS}"
  echo "DELAY_MINUTES=${DELAY_MINUTES}"
}

prepare_download_target() {
  local file_dir rel_dir

  if [ -z "$FILE_PATH" ] || [ "$FILE_PATH" = "<path/to/your/file>" ]; then
    echo "Please set FILE_PATH to a valid .safetensors file path."
    exit 1
  fi

  if [ -z "$FILE_NAME" ]; then
    echo "Failed to derive file name from FILE_PATH: $FILE_PATH"
    exit 1
  fi

  case "$FILE_NAME" in
    *.safetensors) ;;
    *)
      echo "FILE_PATH must end with .safetensors: $FILE_PATH"
      exit 1
      ;;
  esac

  file_dir="$(dirname "$FILE_PATH")"
  case "$file_dir" in
    "$DOWNLOAD_ROOT")
      rel_dir=""
      ;;
    "$DOWNLOAD_ROOT"/*)
      rel_dir="${file_dir#"$DOWNLOAD_ROOT"/}"
      ;;
    *)
      echo "Download mode requires FILE_PATH under DOWNLOAD_ROOT."
      echo "FILE_PATH=${FILE_PATH}"
      echo "DOWNLOAD_ROOT=${DOWNLOAD_ROOT}"
      exit 1
      ;;
  esac

  if [ -n "$rel_dir" ]; then
    TARGET_DIR="${DOWNLOAD_ROOT}/${rel_dir}"
  else
    TARGET_DIR="${DOWNLOAD_ROOT}"
  fi
}

sleep_if_needed() {
  if [ "${DELAY_HOURS}" -gt 0 ] || [ "${DELAY_MINUTES}" -gt 0 ]; then
    echo "Sleeping for ${DELAY_HOURS}h ${DELAY_MINUTES}m..."
    if [ "${DELAY_HOURS}" -gt 0 ]; then
      sleep "${DELAY_HOURS}h"
    fi
    if [ "${DELAY_MINUTES}" -gt 0 ]; then
      sleep "${DELAY_MINUTES}m"
    fi
  fi
}

if [ "$MODE" -eq 0 ]; then
  prepare_download_target
fi

print_run_info
sleep_if_needed

modelscope login --token ms-5a279368-ebad-4332-b759-6018c2fcf6e5

case "$MODE" in
  1)
    if [ ! -f "$FILE_PATH" ]; then
      echo "Upload file not found: $FILE_PATH"
      exit 1
    fi
    modelscope upload zzrzzr/wan_pretrain "$FILE_PATH" --repo-type dataset
    ;;
  0)
    mkdir -p "$TARGET_DIR"
    modelscope download zzrzzr/wan_pretrain "$FILE_NAME" --repo-type dataset --local_dir "$TARGET_DIR"
    ;;
  *)
    echo "Invalid mode: $MODE (use 1 for upload, 0 for download)"
    exit 1
    ;;
esac
