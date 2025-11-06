# DevLog 001: Initial Implementation and Critical Bugfixes

**Date:** 2025-11-06 13:57 MST  
**Author:** Wentao Jiang  
**Status:** Completed

---

## Overview

This log documents the initial implementation of the 3D Gaussian Splatting viewer prototype and resolution of two critical bugs that prevented the application from functioning correctly.

---

## Current Repository Status

### Frontend Stack
- **Framework:** React 19.2.0 + Vite 7.2.1
- **3D Rendering:** Three.js 0.181.0 + Spark 0.1.10
- **Package Manager:** pnpm
- **Location:** `app/frontend/`

### Backend Stack
- **Runtime:** Node.js with TypeScript
- **Framework:** Express 5.1.0
- **File Upload:** Multer 2.0.2
- **Package Manager:** pnpm
- **Location:** `app/backend/`

### Key Features Implemented
1. 3D Gaussian Splatting viewer with SplatViewer component
2. File upload interface with drag-and-drop support
3. PLY file upload endpoint with optional SOG conversion
4. Static file serving for splat assets
5. WASDQE keyboard controls for camera positioning
6. OrbitControls for mouse-based camera manipulation

---

## Bug 1: Canvas Rendering Issue

### Problem Description
The 3D Gaussian Splatting scene rendered correctly only in the bottom half of the canvas, with the top half appearing completely black. Additionally, OrbitControls (mouse camera controls) were non-functional while keyboard controls (WASD) continued to work.

### Root Cause
React StrictMode in development mode intentionally mounts components twice to detect side effects. The `useEffect` hook in `SplatViewer.tsx` was creating duplicate canvas elements without properly cleaning up the first instance, resulting in:
- Two overlapping canvas elements with different dimensions
- First canvas: 2628x1058 internal resolution, 1314x529 CSS (landscape)
- Second canvas: 1572x2116 internal resolution, 786x1058 CSS (portrait, incorrect aspect ratio)
- OrbitControls attached to the wrong canvas element
- WebGL viewport confusion between the two canvases

### Solution
Modified `SplatViewer.tsx` to clear any existing canvas elements before initialization:

```typescript
const container = containerRef.current;

// Clear any existing canvas elements (in case of StrictMode double-mount)
while (container.firstChild) {
  container.removeChild(container.firstChild);
}
```

### Files Modified
- `app/frontend/src/components/SplatViewer.tsx` (lines 21-57)

### Outcome
- Single canvas element correctly rendered
- Full viewport rendering (no split screen)
- OrbitControls functioning correctly
- WASDQE keyboard controls continue to work

---

## Bug 2: File Upload and Processing Failure

### Problem Description
PLY file uploads failed with HTTP 500 Internal Server Error. Backend logs showed:
```
/bin/sh: splat-transform: command not found
```

### Root Cause
The backend server attempted to execute `splat-transform` (PlayCanvas CLI tool for PLY to SOG conversion) which was not installed on the system. The conversion process had no fallback mechanism, causing the entire upload operation to fail.

### Solution
Implemented graceful fallback with try-catch error handling:

1. **Primary path:** Attempt PLY to SOG conversion using `splat-transform` if available
2. **Fallback path:** Serve original PLY file directly (Spark supports PLY natively)
3. **Enhanced response:** Return metadata including `converted`, `format`, and `compressionRatio` fields

### Files Modified
- `app/backend/src/server.ts` (lines 53-132)
- `app/frontend/src/App.tsx` (lines 37-46)

### Technical Details

**Backend changes:**
- Wrapped `execAsync` call in try-catch block
- On conversion failure, retain original PLY file instead of deleting
- Log warnings for conversion failures without breaking the upload flow
- Return format-specific metadata to frontend

**Frontend changes:**
- Display different status messages based on conversion success
- Show file format being served (PLY vs SOG)

### Storage Architecture
Files are persisted to disk using `multer.diskStorage`:
- **Location:** `app/backend/uploads/`
- **Naming:** `upload-{timestamp}-{random}.{ext}`
- **Size limit:** 500MB
- **Serving:** Static file endpoint at `/files`

### Outcome
- PLY uploads succeed regardless of `splat-transform` availability
- Automatic conversion when tool is installed
- User feedback indicates whether conversion occurred
- Large files (327MB tested) handled correctly

---

## Installation Notes

### Optional: Enable PLY to SOG Conversion
To enable automatic conversion for better compression (2-5x file size reduction):

```bash
pnpm add -g @playcanvas/splat-transform
```

The server automatically detects and uses the tool if available.

---

## Testing Performed

1. **Canvas rendering:** Verified single canvas creation, full viewport rendering, and control functionality
2. **File upload without conversion:** Successfully uploaded 327MB PLY file, served directly
3. **Frontend integration:** Confirmed SplatViewer loads and renders uploaded PLY files
4. **Error handling:** Verified graceful degradation when conversion tool unavailable

---

## Next Steps

1. Test with `splat-transform` installed to verify SOG conversion path
2. Implement file cleanup/management for uploads directory
3. Add progress indicators for large file uploads
4. Test progressive loading with SOG format
5. Prepare for EKS deployment (containerization, VPN configuration)

---

## References

- React StrictMode: https://react.dev/reference/react/StrictMode
- PlayCanvas splat-transform: https://github.com/playcanvas/splat-transform
- Spark documentation: https://sparkjs.dev

