#!/usr/bin/env bash

set -euo pipefail

readonly WIDTH=848
readonly HEIGHT=480

usage() {
    printf 'Usage: %s INPUT_VIDEO [OUTPUT_VIDEO]\n' "$(basename "$0")" >&2
}

if (( $# < 1 || $# > 2 )); then
    usage
    exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    printf 'Error: ffmpeg is required but was not found in PATH.\n' >&2
    exit 127
fi

input=$1
if [[ ! -f "$input" ]]; then
    printf 'Error: input video does not exist: %s\n' "$input" >&2
    exit 1
fi

input_dir=$(dirname "$input")
input_name=$(basename "$input")
if [[ "$input_name" != *.* || "$input_name" == .* ]]; then
    printf 'Error: input video must have a filename extension: %s\n' "$input" >&2
    exit 1
fi

extension=${input_name##*.}
stem=${input_name%.*}
output=${2:-"${input_dir}/${stem}-w${WIDTH}h${HEIGHT}.${extension}"}

if [[ "$input" == "$output" ]]; then
    printf 'Error: output path must differ from input path.\n' >&2
    exit 1
fi

if [[ -e "$output" ]]; then
    printf 'Error: output already exists: %s\n' "$output" >&2
    exit 1
fi

output_dir=$(dirname "$output")
output_name=$(basename "$output")
mkdir -p "$output_dir"

temporary_directory=$(mktemp -d "${output_dir}/.${output_name}.part.XXXXXX")
temporary_output="${temporary_directory}/${output_name}"
cleanup() {
    if [[ -f "$temporary_output" ]]; then
        rm -f -- "$temporary_output"
    fi
    rmdir -- "$temporary_directory" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

ffmpeg \
    -hide_banner \
    -i "$input" \
    -map 0:v:0 \
    -map '0:a?' \
    -map_metadata 0 \
    -vf "scale=${WIDTH}:${HEIGHT}:flags=lanczos,setsar=1" \
    -c:v libx264 \
    -preset fast \
    -crf 18 \
    -pix_fmt yuv420p \
    -c:a copy \
    -movflags +faststart \
    "$temporary_output"

mv -- "$temporary_output" "$output"
rmdir -- "$temporary_directory"
trap - EXIT INT TERM

printf 'Created %s\n' "$output"
