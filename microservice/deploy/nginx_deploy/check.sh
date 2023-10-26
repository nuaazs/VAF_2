#!/bin/bash

while read -r expected_md5 file_path; do
    actual_md5=$(md5sum "$file_path" | awk '{print $1}')
    if [ "$actual_md5" != "$expected_md5" ]; then
        echo "[ERROR] Checksum mismatch: $file_path (expected: $expected_md5, actual: $actual_md5)"
    else
        echo "[INFO] Checksum OK: $file_path"
    fi
done < md5.txt

