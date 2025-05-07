import youtubeDl from 'youtube-dl-exec';
import ffmpeg from 'fluent-ffmpeg';
import { PassThrough } from 'stream';
import { writeFile } from 'fs/promises';
import path from 'path';
import { spawn } from 'child_process';

// Create a yt-dlp instance
const ytdlp = youtubeDl.create('yt-dlp');

async function getVideoDuration(videoUrl: string): Promise<number> {
  const info = await ytdlp(videoUrl, {
    dumpJson: true,
    noWarnings: true,
    noCheckCertificates: true,
    preferFreeFormats: true
  }) as any;

  return info.duration;
}

async function extractFrame(
  videoUrl: string,
  timestampSec: number
): Promise<Buffer> {
  // Use yt-dlp to get the video stream
  const ytProcess = spawn('yt-dlp', [
    videoUrl,
    '-f', 'best[ext=mp4]',
    '-o', '-',  // Output to stdout
    '--no-warnings'
  ]);

  return new Promise<Buffer>((resolve, reject) => {
    const pngStream = new PassThrough();
    const chunks: Buffer[] = [];

    ytProcess.on('error', (err) => {
      console.error('yt-dlp process error:', err);
      reject(err);
    });

    ytProcess.stderr.on('data', (data) => {
      console.error('yt-dlp stderr:', data.toString());
    });

    // collect all data from FFmpeg
    pngStream.on('data', (c) => chunks.push(c));
    pngStream.on('end', () => resolve(Buffer.concat(chunks)));
    pngStream.on('error', (err) => {
      console.error('PNG stream error:', err);
      reject(err);
    });

    ffmpeg(ytProcess.stdout)
      .inputOptions([`-ss ${timestampSec}`])
      .outputOptions(['-frames:v 1', '-f image2pipe', '-vcodec png'])
      .on('error', (err) => {
        console.error('FFmpeg error:', err);
        reject(err);
      })
      .pipe(pngStream, { end: true });
  });
}

async function main() {
  const videoId = 'vspD719IM0E';
  const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;

  try {
    console.log('Getting video info...');
    const durationSec = await getVideoDuration(videoUrl);
    const totalMinutes = Math.floor(durationSec / 60);
    console.log(`Video duration: ${totalMinutes} minutes`);

    for (let m = 0; m <= totalMinutes; ++m) {
      const t = m * 60; // seconds
      process.stdout.write(`Extracting minute ${m}/${totalMinutes}... `);
      try {
        const pngBuf = await extractFrame(videoUrl, t);
        const filename = path.resolve(
          process.cwd(),
          `frame_${String(m).padStart(2, '0')}m.png`
        );
        await writeFile(filename, pngBuf);
        console.log('saved to', filename);
      } catch (err) {
        console.error('Failed at minute', m, err);
      }
    }
  } catch (err) {
    console.error('Fatal error:', err);
    process.exit(1);
  }
}

main();
