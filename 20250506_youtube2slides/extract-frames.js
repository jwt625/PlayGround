"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g = Object.create((typeof Iterator === "function" ? Iterator : Object).prototype);
    return g.next = verb(0), g["throw"] = verb(1), g["return"] = verb(2), typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (g && (g = 0, op[0] && (_ = 0)), _) try {
            if (f = 1, y && (t = op[0] & 2 ? y["return"] : op[0] ? y["throw"] || ((t = y["return"]) && t.call(y), 0) : y.next) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [op[0] & 2, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
var ytdl_core_1 = require("ytdl-core");
var fluent_ffmpeg_1 = require("fluent-ffmpeg");
var stream_1 = require("stream");
var promises_1 = require("fs/promises");
var path_1 = require("path");
function getVideoDuration(videoUrl) {
    return __awaiter(this, void 0, void 0, function () {
        var info;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, ytdl_core_1.default.getInfo(videoUrl)];
                case 1:
                    info = _a.sent();
                    return [2 /*return*/, parseFloat(info.videoDetails.lengthSeconds)];
            }
        });
    });
}
function extractFrame(videoUrl, timestampSec) {
    return __awaiter(this, void 0, void 0, function () {
        var videoStream;
        return __generator(this, function (_a) {
            videoStream = (0, ytdl_core_1.default)(videoUrl, { quality: 'highest' });
            return [2 /*return*/, new Promise(function (resolve, reject) {
                    var pngStream = new stream_1.PassThrough();
                    var chunks = [];
                    // collect all data from FFmpeg
                    pngStream.on('data', function (c) { return chunks.push(c); });
                    pngStream.on('end', function () { return resolve(Buffer.concat(chunks)); });
                    pngStream.on('error', reject);
                    (0, fluent_ffmpeg_1.default)(videoStream)
                        .inputOptions(["-ss ".concat(timestampSec)])
                        .outputOptions(['-frames:v 1', '-f image2pipe', '-vcodec png'])
                        .on('error', reject)
                        .pipe(pngStream, { end: true });
                })];
        });
    });
}
function main() {
    return __awaiter(this, void 0, void 0, function () {
        var videoId, videoUrl, durationSec, totalMinutes, m, t, pngBuf, filename, err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    videoId = 'vspD719IM0E';
                    videoUrl = "https://www.youtube.com/watch?v=".concat(videoId);
                    return [4 /*yield*/, getVideoDuration(videoUrl)];
                case 1:
                    durationSec = _a.sent();
                    totalMinutes = Math.floor(durationSec / 60);
                    m = 0;
                    _a.label = 2;
                case 2:
                    if (!(m <= totalMinutes)) return [3 /*break*/, 8];
                    t = m * 60;
                    process.stdout.write("Extracting minute ".concat(m, "\u2026 "));
                    _a.label = 3;
                case 3:
                    _a.trys.push([3, 6, , 7]);
                    return [4 /*yield*/, extractFrame(videoUrl, t)];
                case 4:
                    pngBuf = _a.sent();
                    filename = path_1.default.resolve(__dirname, "frame_".concat(String(m).padStart(2, '0'), "m.png"));
                    return [4 /*yield*/, (0, promises_1.writeFile)(filename, pngBuf)];
                case 5:
                    _a.sent();
                    console.log('saved to', filename);
                    return [3 /*break*/, 7];
                case 6:
                    err_1 = _a.sent();
                    console.error('failed at minute', m, err_1);
                    return [3 /*break*/, 7];
                case 7:
                    ++m;
                    return [3 /*break*/, 2];
                case 8: return [2 /*return*/];
            }
        });
    });
}
main().catch(function (err) {
    console.error('fatal error:', err);
    process.exit(1);
});
