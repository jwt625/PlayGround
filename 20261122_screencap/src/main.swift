// ScreenCap: menu bar region screen recorder with system audio (ScreenCaptureKit).
// Build with ../build.sh. Usage: Cmd+Shift+7 or menu bar icon -> drag region -> Enter.

import AppKit
import AVFoundation
import AVKit
import Carbon.HIToolbox
import CoreMedia
import ScreenCaptureKit

// MARK: - Geometry helpers

/// Output pixel size for a region of `size` points at `scale`. Even dimensions (H.264 4:2:0),
/// capped to the H.264 encoder limit (max side 4096, area 4096x2304) with aspect preserved.
func outputSize(_ size: CGSize, scale: CGFloat) -> (Int, Int) {
    var w = size.width * scale
    var h = size.height * scale
    let f = min(1, 4096 / max(w, h), (4096 * 2304 / (w * h)).squareRoot())
    w *= f
    h *= f
    return (max(2, Int(w.rounded()) & ~1), max(2, Int(h.rounded()) & ~1))
}

/// Snap a rect (points) to the device pixel grid.
func snap(_ r: NSRect, scale: CGFloat) -> NSRect {
    let x0 = (r.minX * scale).rounded() / scale
    let y0 = (r.minY * scale).rounded() / scale
    let x1 = (r.maxX * scale).rounded() / scale
    let y1 = (r.maxY * scale).rounded() / scale
    return NSRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
}

func displayID(of screen: NSScreen) -> CGDirectDisplayID {
    (screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? NSNumber)?.uint32Value ?? CGMainDisplayID()
}

func screenUnderMouse() -> NSScreen? {
    let p = NSEvent.mouseLocation
    return NSScreen.screens.first { NSMouseInRect(p, $0.frame, false) } ?? NSScreen.main
}

// MARK: - Recorder

final class Recorder: NSObject, SCStreamOutput, SCStreamDelegate {
    struct Target {
        let displayID: CGDirectDisplayID
        let rect: CGRect  // display-local points, top-left origin
        let scale: CGFloat
    }

    private let queue = DispatchQueue(label: "screencap.recorder")
    private var stream: SCStream?
    private var writer: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var audioInput: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var sessionStart: CMTime?
    private var lastPixelBuffer: CVPixelBuffer?
    private var lastVideoTime = CMTime.invalid
    private var url: URL?
    var onError: ((Error) -> Void)?

    func start(_ target: Target, url: URL) async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        guard let display = content.displays.first(where: { $0.displayID == target.displayID }) else {
            throw NSError(domain: "ScreenCap", code: 1, userInfo: [NSLocalizedDescriptionKey: "Display not found"])
        }
        let pid = ProcessInfo.processInfo.processIdentifier
        let own = content.applications.filter { $0.processID == pid }
        let filter = SCContentFilter(display: display, excludingApplications: own, exceptingWindows: [])

        let (w, h) = outputSize(target.rect.size, scale: target.scale)
        let cfg = SCStreamConfiguration()
        cfg.sourceRect = target.rect
        cfg.width = w
        cfg.height = h
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: 60)
        cfg.queueDepth = 6
        cfg.showsCursor = true
        cfg.pixelFormat = kCVPixelFormatType_32BGRA
        cfg.colorSpaceName = CGColorSpace.sRGB
        cfg.capturesAudio = true
        cfg.excludesCurrentProcessAudio = true
        cfg.sampleRate = 48_000
        cfg.channelCount = 2

        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        writer.shouldOptimizeForNetworkUse = true
        let bitrate = min(80_000_000, max(4_000_000, w * h * 6))
        let vIn = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoExpectedSourceFrameRateKey: 60,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoMaxKeyFrameIntervalDurationKey: 2,
            ],
            AVVideoColorPropertiesKey: [
                AVVideoColorPrimariesKey: AVVideoColorPrimaries_ITU_R_709_2,
                AVVideoTransferFunctionKey: AVVideoTransferFunction_ITU_R_709_2,
                AVVideoYCbCrMatrixKey: AVVideoYCbCrMatrix_ITU_R_709_2,
            ],
        ])
        vIn.expectsMediaDataInRealTime = true
        let aIn = AVAssetWriterInput(mediaType: .audio, outputSettings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 48_000,
            AVNumberOfChannelsKey: 2,
            AVEncoderBitRateKey: 192_000,
        ])
        aIn.expectsMediaDataInRealTime = true
        writer.add(vIn)
        writer.add(aIn)
        // The adaptor must exist before startWriting().
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: vIn, sourcePixelBufferAttributes: nil)
        guard writer.startWriting() else {
            throw writer.error ?? NSError(domain: "ScreenCap", code: 2, userInfo: [NSLocalizedDescriptionKey: "Writer failed to start"])
        }
        self.writer = writer
        self.videoInput = vIn
        self.audioInput = aIn
        self.adaptor = adaptor
        self.url = url

        let stream = SCStream(filter: filter, configuration: cfg, delegate: self)
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: queue)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: queue)
        do {
            try await stream.startCapture()
        } catch {
            writer.cancelWriting()
            try? FileManager.default.removeItem(at: url)
            throw error
        }
        self.stream = stream
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard sb.isValid, let writer, writer.status == .writing else { return }
        switch type {
        case .screen:
            guard
                let atts = CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]],
                let raw = atts.first?[.status] as? Int,
                SCFrameStatus(rawValue: raw) == .complete,
                let pb = CMSampleBufferGetImageBuffer(sb)
            else { return }
            let t = sb.presentationTimeStamp
            if sessionStart == nil {
                writer.startSession(atSourceTime: t)
                sessionStart = t
            }
            if let videoInput, videoInput.isReadyForMoreMediaData {
                adaptor?.append(pb, withPresentationTime: t)
            }
            lastPixelBuffer = pb
            lastVideoTime = t
        case .audio:
            guard let start = sessionStart, sb.presentationTimeStamp >= start,
                  let audioInput, audioInput.isReadyForMoreMediaData else { return }
            audioInput.append(sb)
        default:
            break
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        DispatchQueue.main.async { self.onError?(error) }
    }

    /// Stops capture and finalizes the file. Returns the output URL.
    func stop() async throws -> URL {
        if let stream { try? await stream.stopCapture() }
        stream = nil
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                guard let writer = self.writer, let url = self.url,
                      let vIn = self.videoInput, let aIn = self.audioInput else {
                    cont.resume(throwing: NSError(domain: "ScreenCap", code: 3, userInfo: [NSLocalizedDescriptionKey: "Not recording"]))
                    return
                }
                guard self.sessionStart != nil else {
                    writer.cancelWriting()
                    try? FileManager.default.removeItem(at: url)
                    cont.resume(throwing: NSError(domain: "ScreenCap", code: 4, userInfo: [NSLocalizedDescriptionKey: "No frames were captured"]))
                    return
                }
                // SCK only sends frames on change: repeat the last frame at stop so video spans the full duration.
                let end = CMClockGetTime(CMClockGetHostTimeClock())
                if let pb = self.lastPixelBuffer, end > self.lastVideoTime, vIn.isReadyForMoreMediaData {
                    self.adaptor?.append(pb, withPresentationTime: end)
                }
                vIn.markAsFinished()
                aIn.markAsFinished()
                writer.endSession(atSourceTime: end)
                writer.finishWriting {
                    if writer.status == .completed {
                        cont.resume(returning: url)
                    } else {
                        cont.resume(throwing: writer.error ?? NSError(domain: "ScreenCap", code: 5, userInfo: [NSLocalizedDescriptionKey: "Writer failed"]))
                    }
                }
            }
        }
    }
}

// MARK: - Region selector

final class OverlayWindow: NSWindow {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { true }
}

final class SelectionController {
    var windows: [OverlayWindow] = []
    var selection: NSRect?  // global Cocoa coordinates (points)
    var screen: NSScreen?
    var onDone: ((NSScreen, NSRect) -> Void)?
    var onCancel: (() -> Void)?

    func show(initial: NSRect?) {
        if let r = initial, let s = NSScreen.screens.first(where: { $0.frame.contains(r) }) {
            selection = r
            screen = s
        }
        let mouseScreen = screenUnderMouse()
        var keyWindow: OverlayWindow?
        for s in NSScreen.screens {
            let w = OverlayWindow(contentRect: s.frame, styleMask: .borderless, backing: .buffered, defer: false)
            w.setFrame(s.frame, display: false)
            w.level = .screenSaver
            w.isOpaque = false
            w.backgroundColor = .clear
            w.hasShadow = false
            w.acceptsMouseMovedEvents = true
            w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
            let v = OverlayView(frame: NSRect(origin: .zero, size: s.frame.size))
            v.controller = self
            v.screen = s
            w.contentView = v
            w.orderFrontRegardless()
            windows.append(w)
            if s == mouseScreen { keyWindow = w }
        }
        NSApp.activate(ignoringOtherApps: true)
        if let k = keyWindow ?? windows.first {
            k.makeKeyAndOrderFront(nil)
            k.makeFirstResponder(k.contentView)
        }
    }

    func redraw() { windows.forEach { $0.contentView?.needsDisplay = true } }

    func close() {
        windows.forEach { $0.orderOut(nil) }
        windows = []
    }

    func confirm() {
        guard let sel = selection, let screen, sel.width >= 4, sel.height >= 4 else { return }
        close()
        onDone?(screen, sel)
    }

    func cancel() {
        close()
        onCancel?()
    }
}

struct Edges: OptionSet {
    let rawValue: Int
    static let left = Edges(rawValue: 1)
    static let right = Edges(rawValue: 2)
    static let bottom = Edges(rawValue: 4)
    static let top = Edges(rawValue: 8)
}

final class OverlayView: NSView {
    weak var controller: SelectionController?
    var screen: NSScreen!

    private enum Drag {
        case none
        case create(NSPoint)
        case move(NSPoint, NSRect)
        case resize(Edges, NSRect)
    }

    private var drag = Drag.none
    private let tol: CGFloat = 6

    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        trackingAreas.forEach { removeTrackingArea($0) }
        addTrackingArea(NSTrackingArea(rect: bounds, options: [.activeAlways, .mouseMoved, .inVisibleRect], owner: self))
    }

    private var localSelection: NSRect? {
        guard let c = controller, c.screen == screen, let s = c.selection else { return nil }
        return s.offsetBy(dx: -screen.frame.minX, dy: -screen.frame.minY)
    }

    private func setLocal(_ r: NSRect) {
        controller?.screen = screen
        controller?.selection = r.offsetBy(dx: screen.frame.minX, dy: screen.frame.minY)
        controller?.redraw()
    }

    private func hit(_ p: NSPoint) -> (Edges, Bool) {
        guard let r = localSelection else { return ([], false) }
        var e: Edges = []
        let inY = p.y > r.minY - tol && p.y < r.maxY + tol
        let inX = p.x > r.minX - tol && p.x < r.maxX + tol
        if inY && abs(p.x - r.minX) < tol { e.insert(.left) }
        if inY && abs(p.x - r.maxX) < tol { e.insert(.right) }
        if inX && abs(p.y - r.minY) < tol { e.insert(.bottom) }
        if inX && abs(p.y - r.maxY) < tol { e.insert(.top) }
        if e.contains([.left, .right]) { e.remove(abs(p.x - r.minX) < abs(p.x - r.maxX) ? .right : .left) }
        if e.contains([.bottom, .top]) { e.remove(abs(p.y - r.minY) < abs(p.y - r.maxY) ? .top : .bottom) }
        return (e, r.contains(p))
    }

    private func recordButtonRect() -> NSRect? {
        guard let r = localSelection else { return nil }
        let size = NSSize(width: 76, height: 24)
        var y = r.minY - size.height - 10
        if y < 4 { y = r.minY + 10 }
        let x = min(max(4, r.midX - size.width / 2), bounds.width - size.width - 4)
        return NSRect(x: x, y: y, width: size.width, height: size.height)
    }

    private func clampPoint(_ p: NSPoint) -> NSPoint {
        NSPoint(x: min(max(0, p.x), bounds.width), y: min(max(0, p.y), bounds.height))
    }

    override func mouseMoved(with event: NSEvent) {
        let p = convert(event.locationInWindow, from: nil)
        if let b = recordButtonRect(), b.contains(p) { NSCursor.pointingHand.set(); return }
        let (e, inside) = hit(p)
        if e == [.left] || e == [.right] {
            NSCursor.resizeLeftRight.set()
        } else if e == [.top] || e == [.bottom] {
            NSCursor.resizeUpDown.set()
        } else if !e.isEmpty {
            NSCursor.crosshair.set()
        } else if inside {
            NSCursor.openHand.set()
        } else {
            NSCursor.crosshair.set()
        }
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeKey()
        window?.makeFirstResponder(self)
        let p = convert(event.locationInWindow, from: nil)
        if let b = recordButtonRect(), b.contains(p) { controller?.confirm(); return }
        let (e, inside) = hit(p)
        if let r = localSelection, !e.isEmpty {
            drag = .resize(e, r)
        } else if let r = localSelection, inside {
            if event.clickCount == 2 { controller?.confirm(); return }
            drag = .move(p, r)
            NSCursor.closedHand.set()
        } else {
            drag = .create(p)
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let p = clampPoint(convert(event.locationInWindow, from: nil))
        switch drag {
        case .none:
            break
        case .create(let a):
            setLocal(NSRect(x: min(a.x, p.x), y: min(a.y, p.y), width: abs(p.x - a.x), height: abs(p.y - a.y)))
        case .move(let a, let r0):
            var r = r0.offsetBy(dx: p.x - a.x, dy: p.y - a.y)
            r.origin.x = min(max(0, r.minX), bounds.width - r.width)
            r.origin.y = min(max(0, r.minY), bounds.height - r.height)
            setLocal(r)
        case .resize(let e, let r0):
            var x0 = r0.minX, x1 = r0.maxX, y0 = r0.minY, y1 = r0.maxY
            if e.contains(.left) { x0 = p.x }
            if e.contains(.right) { x1 = p.x }
            if e.contains(.bottom) { y0 = p.y }
            if e.contains(.top) { y1 = p.y }
            setLocal(NSRect(x: min(x0, x1), y: min(y0, y1), width: abs(x1 - x0), height: abs(y1 - y0)))
        }
    }

    override func mouseUp(with event: NSEvent) {
        drag = .none
        if let r = localSelection, r.width < 4 || r.height < 4 {
            controller?.selection = nil
            controller?.redraw()
        }
    }

    override func keyDown(with event: NSEvent) {
        switch Int(event.keyCode) {
        case kVK_Escape: controller?.cancel()
        case kVK_Return, kVK_ANSI_KeypadEnter: controller?.confirm()
        default: super.keyDown(with: event)
        }
    }

    private func drawLabel(_ text: String, centeredAt c: NSPoint) {
        let attrs: [NSAttributedString.Key: Any] = [
            .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .medium),
            .foregroundColor: NSColor.white,
        ]
        let s = NSAttributedString(string: text, attributes: attrs)
        let sz = s.size()
        let box = NSRect(x: c.x - sz.width / 2 - 6, y: c.y - sz.height / 2 - 3, width: sz.width + 12, height: sz.height + 6)
        NSColor(white: 0, alpha: 0.75).setFill()
        box.fill()
        s.draw(at: NSPoint(x: box.minX + 6, y: box.minY + 3))
    }

    override func draw(_ dirtyRect: NSRect) {
        NSColor(white: 0, alpha: 0.4).setFill()
        bounds.fill()
        guard let r = localSelection else {
            drawLabel("Drag to select    Enter to record    Esc to cancel", centeredAt: NSPoint(x: bounds.midX, y: bounds.midY))
            return
        }
        NSColor.clear.setFill()
        r.fill(using: .copy)

        let path = NSBezierPath(rect: r.insetBy(dx: -0.5, dy: -0.5))
        path.lineWidth = 1
        path.setLineDash([4, 3], count: 2, phase: 0)
        NSColor.white.setStroke()
        path.stroke()

        NSColor.white.setFill()
        for x in [r.minX, r.midX, r.maxX] {
            for y in [r.minY, r.midY, r.maxY] where !(x == r.midX && y == r.midY) {
                NSRect(x: x - 3, y: y - 3, width: 6, height: 6).fill()
            }
        }

        let (pw, ph) = outputSize(r.size, scale: screen.backingScaleFactor)
        var labelY = r.maxY + 16
        if labelY > bounds.height - 12 { labelY = r.maxY - 16 }
        drawLabel("\(pw) x \(ph)", centeredAt: NSPoint(x: r.midX, y: labelY))

        if let b = recordButtonRect() {
            NSColor(srgbRed: 0.82, green: 0.16, blue: 0.16, alpha: 1).setFill()
            b.fill()
            let attrs: [NSAttributedString.Key: Any] = [
                .font: NSFont.systemFont(ofSize: 12, weight: .semibold),
                .foregroundColor: NSColor.white,
            ]
            let s = NSAttributedString(string: "Record", attributes: attrs)
            let sz = s.size()
            s.draw(at: NSPoint(x: b.midX - sz.width / 2, y: b.midY - sz.height / 2))
        }
    }
}

// MARK: - Recording border

final class BorderView: NSView {
    override func draw(_ dirtyRect: NSRect) {
        let p = NSBezierPath(rect: bounds.insetBy(dx: 1, dy: 1))
        p.lineWidth = 2
        NSColor.systemRed.setStroke()
        p.stroke()
    }
}

func makeBorderWindow(around r: NSRect) -> NSWindow {
    let f = r.insetBy(dx: -3, dy: -3)
    let w = NSWindow(contentRect: f, styleMask: .borderless, backing: .buffered, defer: false)
    w.setFrame(f, display: false)
    w.isOpaque = false
    w.backgroundColor = .clear
    w.hasShadow = false
    w.ignoresMouseEvents = true
    w.level = .statusBar
    w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .stationary]
    w.contentView = BorderView(frame: NSRect(origin: .zero, size: f.size))
    w.orderFrontRegardless()
    return w
}

// MARK: - Trim

/// Passthrough (no re-encode) export of `range` from `src`. On success the original goes to the Trash
/// and the trimmed file takes its name.
func exportTrim(src: URL, range: CMTimeRange, completion: @escaping (Result<URL, Error>) -> Void) {
    let asset = AVURLAsset(url: src)
    guard let ex = AVAssetExportSession(asset: asset, presetName: AVAssetExportPresetPassthrough) else {
        completion(.failure(NSError(domain: "ScreenCap", code: 6, userInfo: [NSLocalizedDescriptionKey: "Cannot create export session"])))
        return
    }
    let tmp = src.deletingPathExtension().appendingPathExtension("trimming.mp4")
    try? FileManager.default.removeItem(at: tmp)
    ex.outputURL = tmp
    ex.outputFileType = .mp4
    ex.timeRange = range
    ex.shouldOptimizeForNetworkUse = true
    ex.exportAsynchronously {
        guard ex.status == .completed else {
            try? FileManager.default.removeItem(at: tmp)
            completion(.failure(ex.error ?? NSError(domain: "ScreenCap", code: 7, userInfo: [NSLocalizedDescriptionKey: "Export failed"])))
            return
        }
        do {
            try FileManager.default.trashItem(at: src, resultingItemURL: nil)
            try FileManager.default.moveItem(at: tmp, to: src)
            completion(.success(src))
        } catch {
            completion(.failure(error))
        }
    }
}

/// Window with the QuickTime-style trim bar (AVPlayerView trimming) shown right after a recording.
final class TrimController: NSObject, NSWindowDelegate {
    let url: URL
    private let window: NSWindow
    private let playerView = AVPlayerView()
    private var statusObs: NSKeyValueObservation?
    private var finished = false
    /// Called once with the final file URL (trimmed or untouched), or an error.
    var onFinish: ((TrimController, Result<URL, Error>) -> Void)?

    init(url: URL) {
        self.url = url
        let item = AVPlayerItem(asset: AVURLAsset(url: url))
        playerView.player = AVPlayer(playerItem: item)
        playerView.controlsStyle = .inline
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 600),
                          styleMask: [.titled, .closable, .resizable, .miniaturizable], backing: .buffered, defer: false)
        super.init()
        window.title = url.lastPathComponent
        window.contentView = playerView
        window.delegate = self
        window.isReleasedWhenClosed = false
        statusObs = item.observe(\.status, options: [.initial, .new]) { [weak self] item, _ in
            guard item.status == .readyToPlay else { return }
            DispatchQueue.main.async { self?.ready() }
        }
    }

    func show() {
        window.center()
        NSApp.activate(ignoringOtherApps: true)
        window.makeKeyAndOrderFront(nil)
    }

    private func ready() {
        guard statusObs != nil else { return }
        statusObs?.invalidate()
        statusObs = nil
        // Fit the window to the video aspect within 70% of the screen.
        if let size = playerView.player?.currentItem?.presentationSize, size.width > 0, size.height > 0,
           let vis = (window.screen ?? NSScreen.main)?.visibleFrame {
            let f = min(1, vis.width * 0.7 / size.width, vis.height * 0.7 / size.height)
            let w = max(480, size.width * f)
            window.setContentSize(NSSize(width: w, height: w * size.height / size.width + 60))
            window.center()
        }
        guard playerView.canBeginTrimming else { return }
        playerView.beginTrimming { [weak self] result in self?.trimDone(result) }
    }

    private func trimDone(_ result: AVPlayerViewTrimResult) {
        guard !finished else { return }
        finished = true
        playerView.player?.pause()
        guard result == .okButton, let item = playerView.player?.currentItem else {
            window.close()
            onFinish?(self, .success(url))
            return
        }
        let dur = item.duration
        let start = item.reversePlaybackEndTime.isValid ? item.reversePlaybackEndTime : .zero
        let end = item.forwardPlaybackEndTime.isValid ? item.forwardPlaybackEndTime : dur
        window.close()
        if start == .zero && (end == dur || !dur.isNumeric) {
            onFinish?(self, .success(url))
            return
        }
        exportTrim(src: url, range: CMTimeRange(start: start, end: end)) { res in
            DispatchQueue.main.async { self.onFinish?(self, res) }
        }
    }

    func windowWillClose(_ notification: Notification) {
        playerView.player?.pause()
        guard !finished else { return }
        finished = true
        onFinish?(self, .success(url))
    }
}

// MARK: - App

final class AppDelegate: NSObject, NSApplicationDelegate {
    static var shared: AppDelegate!

    private var statusItem: NSStatusItem!
    private let menu = NSMenu()
    private var selector: SelectionController?
    private var recorder: Recorder?
    private var stopping = false
    private var border: NSWindow?
    private var timer: Timer?
    private var startDate: Date?
    private var hotKeyRef: EventHotKeyRef?
    private var testMode = false
    private var trimmers: [TrimController] = []

    private var lastRegion: NSRect? {
        get {
            guard let s = UserDefaults.standard.string(forKey: "lastRegion") else { return nil }
            let r = NSRectFromString(s)
            return r.width >= 4 && r.height >= 4 ? r : nil
        }
        set { UserDefaults.standard.set(newValue.map(NSStringFromRect), forKey: "lastRegion") }
    }

    static var outputDir: URL {
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Movies/ScreenCap", isDirectory: true)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let b = statusItem.button {
            b.target = self
            b.action = #selector(statusClicked)
            b.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
        setIdleUI()

        let region = NSMenuItem(title: "Record Region", action: #selector(selectRegion), keyEquivalent: "7")
        region.keyEquivalentModifierMask = [.command, .shift]
        menu.addItem(region)
        menu.addItem(NSMenuItem(title: "Record Full Screen", action: #selector(recordFullScreen), keyEquivalent: ""))
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Open Recordings Folder", action: #selector(openFolder), keyEquivalent: ""))
        menu.addItem(.separator())
        menu.addItem(NSMenuItem(title: "Quit ScreenCap", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q"))
        for item in menu.items where item.action != #selector(NSApplication.terminate(_:)) { item.target = self }

        registerHotKey()

        let args = CommandLine.arguments
        if let i = args.firstIndex(of: "--test"), i + 1 < args.count, let secs = Double(args[i + 1]) {
            runTest(seconds: secs, args: args)
        } else if let i = args.firstIndex(of: "--trim"), i + 3 < args.count,
                  let a = Double(args[i + 2]), let b = Double(args[i + 3]) {
            // Headless trim test: --trim FILE START_S END_S
            testMode = true
            let range = CMTimeRange(start: CMTime(seconds: a, preferredTimescale: 600), end: CMTime(seconds: b, preferredTimescale: 600))
            exportTrim(src: URL(fileURLWithPath: args[i + 1]), range: range) { res in
                switch res {
                case .success(let u): print(u.path); exit(0)
                case .failure(let e): FileHandle.standardError.write("\(e)\n".data(using: .utf8)!); exit(1)
                }
            }
        } else if let i = args.firstIndex(of: "--edit"), i + 1 < args.count {
            // Open the trim window on an existing file: --edit FILE
            openTrimmer(URL(fileURLWithPath: args[i + 1]))
        }
    }

    private func openTrimmer(_ url: URL) {
        let t = TrimController(url: url)
        t.onFinish = { [weak self] t, res in
            guard let self else { return }
            self.trimmers.removeAll { $0 === t }
            switch res {
            case .success(let u): NSWorkspace.shared.activateFileViewerSelecting([u])
            case .failure(let e): self.fail("Trim failed: \(e.localizedDescription). Original kept at \(t.url.path)")
            }
        }
        trimmers.append(t)
        t.show()
    }

    // MARK: UI state

    private func setIdleUI() {
        guard let b = statusItem.button else { return }
        b.attributedTitle = NSAttributedString(string: "")
        let img = NSImage(systemSymbolName: "record.circle", accessibilityDescription: "ScreenCap")
        img?.isTemplate = true
        b.image = img
    }

    private func updateRecordingTitle() {
        guard let b = statusItem.button, let start = startDate else { return }
        let s = Int(Date().timeIntervalSince(start))
        let text = String(format: "REC %d:%02d", s / 60, s % 60)
        b.image = nil
        b.attributedTitle = NSAttributedString(string: text, attributes: [
            .foregroundColor: NSColor.systemRed,
            .font: NSFont.monospacedDigitSystemFont(ofSize: 12, weight: .semibold),
        ])
    }

    @objc private func statusClicked() {
        if recorder != nil {
            stopRecording()
            return
        }
        statusItem.menu = menu
        statusItem.button?.performClick(nil)
        statusItem.menu = nil
    }

    @objc private func openFolder() {
        try? FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(Self.outputDir)
    }

    // MARK: Hotkey (Cmd+Shift+7): select region, or stop if recording

    private func registerHotKey() {
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetApplicationEventTarget(), { _, _, _ -> OSStatus in
            DispatchQueue.main.async { AppDelegate.shared.toggle() }
            return noErr
        }, 1, &spec, nil, nil)
        let id = EventHotKeyID(signature: OSType(0x5343_6170), id: 1)  // 'SCap'
        RegisterEventHotKey(UInt32(kVK_ANSI_7), UInt32(cmdKey | shiftKey), id, GetApplicationEventTarget(), 0, &hotKeyRef)
    }

    func toggle() {
        if recorder != nil {
            stopRecording()
        } else if let sel = selector {
            sel.cancel()
        } else {
            selectRegion()
        }
    }

    // MARK: Actions

    @objc private func selectRegion() {
        guard recorder == nil, selector == nil else { return }
        let sc = SelectionController()
        sc.onDone = { [weak self] screen, rect in
            guard let self else { return }
            self.selector = nil
            self.lastRegion = rect
            self.startRecording(screen: screen, rect: rect, showBorder: true)
        }
        sc.onCancel = { [weak self] in self?.selector = nil }
        selector = sc
        sc.show(initial: lastRegion)
    }

    @objc private func recordFullScreen() {
        guard recorder == nil, selector == nil, let s = screenUnderMouse() else { return }
        startRecording(screen: s, rect: s.frame, showBorder: false)
    }

    private func newOutputURL() -> URL {
        try? FileManager.default.createDirectory(at: Self.outputDir, withIntermediateDirectories: true)
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH.mm.ss"
        return Self.outputDir.appendingPathComponent("ScreenCap \(f.string(from: Date())).mp4")
    }

    private func startRecording(screen: NSScreen, rect: NSRect, showBorder: Bool) {
        let scale = screen.backingScaleFactor
        let r = snap(rect.intersection(screen.frame), scale: scale)
        let local = CGRect(x: r.minX - screen.frame.minX, y: screen.frame.maxY - r.maxY, width: r.width, height: r.height)
        let target = Recorder.Target(displayID: displayID(of: screen), rect: local, scale: scale)
        let url = newOutputURL()
        let rec = Recorder()
        rec.onError = { [weak self] err in self?.fail("Recording stopped: \(err.localizedDescription)") }
        recorder = rec
        startDate = Date()
        updateRecordingTitle()
        Task { @MainActor in
            do {
                try await rec.start(target, url: url)
                if showBorder { self.border = makeBorderWindow(around: r) }
                self.startDate = Date()
                self.timer = Timer.scheduledTimer(withTimeInterval: 0.5, repeats: true) { _ in self.updateRecordingTitle() }
                if self.testMode {
                    let (w, h) = outputSize(r.size, scale: scale)
                    print("recording \(w)x\(h) from display \(target.displayID), rect \(local)")
                }
            } catch {
                self.recorder = nil
                self.setIdleUI()
                self.fail("Could not start recording: \(error.localizedDescription)")
            }
        }
    }

    private func stopRecording() {
        guard let rec = recorder, !stopping else { return }
        stopping = true
        timer?.invalidate()
        timer = nil
        border?.orderOut(nil)
        border = nil
        Task { @MainActor in
            do {
                let url = try await rec.stop()
                if self.testMode {
                    print(url.path)
                    exit(0)
                }
                self.openTrimmer(url)
            } catch {
                self.fail("Could not save recording: \(error.localizedDescription)")
            }
            self.recorder = nil
            self.stopping = false
            self.setIdleUI()
        }
    }

    private func fail(_ message: String) {
        if testMode {
            FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
            exit(1)
        }
        timer?.invalidate()
        timer = nil
        border?.orderOut(nil)
        border = nil
        let a = NSAlert()
        a.messageText = "ScreenCap"
        a.informativeText = message
        NSApp.activate(ignoringOtherApps: true)
        a.runModal()
    }

    // MARK: Self-test: --test SECONDS [--region x,y,w,h]  (global Cocoa points)

    private func runTest(seconds: Double, args: [String]) {
        testMode = true
        guard let main = NSScreen.main else { exit(2) }
        var rect = main.frame
        if let j = args.firstIndex(of: "--region"), j + 1 < args.count {
            let v = args[j + 1].split(separator: ",").compactMap { Double($0) }
            if v.count == 4 { rect = NSRect(x: v[0], y: v[1], width: v[2], height: v[3]) }
        }
        let screen = NSScreen.screens.first { $0.frame.intersects(rect) } ?? main
        print("screen \(screen.frame) scale \(screen.backingScaleFactor)")
        startRecording(screen: screen, rect: rect, showBorder: true)
        DispatchQueue.main.asyncAfter(deadline: .now() + seconds) { self.stopRecording() }
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
AppDelegate.shared = delegate
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
