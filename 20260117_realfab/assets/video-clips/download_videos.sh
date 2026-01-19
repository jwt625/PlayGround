#!/bin/bash
# Download videos from sources.md with rate limit between downloads
# Video-only mode, breaks immediately on 403/rate limit

cd "$(dirname "$0")"

# Array of YouTube URLs to download (remaining videos)
urls=(
    "https://www.youtube.com/watch?v=xACB3-FSTmg"   # RF GaN Experience Fab Tour
    "https://www.youtube.com/watch?v=h_zgURwr6nA"   # Unveiling High NA EUV ASML
    "https://www.youtube.com/watch?v=WKHKy89QaV0"   # Inside Micron Taiwan
    "https://www.youtube.com/watch?v=hdxjWJ2c0ao"   # Nordson ASYMTEK NexJet
    "https://www.youtube.com/watch?v=pmdibqc7-Hc"   # Siltronic insights
    "https://www.youtube.com/watch?v=Fx3XwzGwQSY"   # SLS Powders Explained
    "https://www.youtube.com/watch?v=fafBpOJv3jM"   # SparkNano Omega
    "https://www.youtube.com/watch?v=ZSSl1O8-YZc"   # EHLA Extreme High-speed Laser
    "https://www.youtube.com/watch?v=LW2pgvmz5Rw"   # TruLaser Cell 7040 Overview
    "https://www.youtube.com/watch?v=cr40F4l56Jc"   # Makera Z1 Unicorn
    "https://www.youtube.com/watch?v=wThtfAtB5U8"   # Microscale 3D printing spaceship
    "https://www.youtube.com/watch?v=ry_Vuzw-ASM"   # 5-Axis Cutting Silicon Carbide
    "https://www.youtube.com/watch?v=_qKsRRZNh74"   # Eplus3D Metal 3D Printers Dual Lasers
)

total=${#urls[@]}
count=0

for url in "${urls[@]}"; do
    count=$((count + 1))
    echo "========================================"
    echo "Downloading $count of $total: $url"
    echo "========================================"
    
    yt-dlp --fragment-retries 0 --retries 0 --sleep-requests 2 \
           -f "bestvideo" -o "%(title)s.%(ext)s" "$url" 2>&1 | tee /tmp/ytdlp_output.txt

    # Check for 403 or rate limit errors
    if grep -q "HTTP Error 403\|rate limit\|Too Many Requests" /tmp/ytdlp_output.txt; then
        echo "========================================"
        echo "RATE LIMITED OR 403 ERROR - STOPPING"
        echo "========================================"
        exit 1
    fi

    if [ $count -lt $total ]; then
        echo "Waiting 120 seconds before next download..."
        sleep 120
    fi
done

echo "========================================"
echo "All downloads complete!"
echo "========================================"

