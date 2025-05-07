"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const ytdl_core_1 = __importDefault(require("ytdl-core"));
const fluent_ffmpeg_1 = __importDefault(require("fluent-ffmpeg"));
const stream_1 = require("stream");
const promises_1 = require("fs/promises");
const path_1 = __importDefault(require("path"));
async function getVideoDuration(videoUrl) {
    const info = await ytdl_core_1.default.getInfo(videoUrl);
    return parseFloat(info.videoDetails.lengthSeconds);
}
async function extractFrame(videoUrl, timestampSec) {
    const videoStream = (0, ytdl_core_1.default)(videoUrl, { quality: 'highest' });
    return new Promise((resolve, reject) => {
        const pngStream = new stream_1.PassThrough();
        const chunks = [];
        // collect all data from FFmpeg
        pngStream.on('data', (c) => chunks.push(c));
        pngStream.on('end', () => resolve(Buffer.concat(chunks)));
        pngStream.on('error', reject);
        (0, fluent_ffmpeg_1.default)(videoStream)
            .inputOptions([`-ss ${timestampSec}`])
            .outputOptions(['-frames:v 1', '-f image2pipe', '-vcodec png'])
            .on('error', reject)
            .pipe(pngStream, { end: true });
    });
}
async function main() {
    const videoId = 'vspD719IM0E';
    const videoUrl = `https://www.youtube.com/watch?v=${videoId}`;
    // 1) figure out how long the video is
    const durationSec = await getVideoDuration(videoUrl);
    const totalMinutes = Math.floor(durationSec / 60);
    // 2) loop one frame per minute
    for (let m = 0; m <= totalMinutes; ++m) {
        const t = m * 60; // seconds
        process.stdout.write(`Extracting minute ${m}… `);
        try {
            const pngBuf = await extractFrame(videoUrl, t);
            const filename = path_1.default.resolve(__dirname, `frame_${String(m).padStart(2, '0')}m.png`);
            await (0, promises_1.writeFile)(filename, pngBuf);
            console.log('saved to', filename);
        }
        catch (err) {
            console.error('failed at minute', m, err);
        }
    }
}
main().catch((err) => {
    console.error('fatal error:', err);
    process.exit(1);
});
