# 3D Gaussian Splatting Viewer Prototype — Architecture Summary

## Objective
Serve an interactive 3D viewer for large (~120 MB) `.ply` Gaussian Splatting files within an **existing EKS cluster** (company playground environment), using open-source WebGL/WebGPU-based technologies.

## Context
- Deploying as a **subroute** on existing EKS playground for internal company demos
- All assets behind **company VPN**
- 3DGS models trained with **confirmed 3rd-order SH (SH3)** data
- Existing infrastructure: Thanos metrics scraping, monitoring endpoints

---

## Decision Summary

### ✅ Renderer Choice: **Spark (Three.js)**
- Fully supports **up to 3rd-order spherical harmonics (SH3)**.
- Actively maintained, open-source, and built on **Three.js** for high browser compatibility.
- Provides:
  - Progressive and LOD streaming formats (`.spz`, `.sogz`)
  - Environment lighting and tone-mapping
  - Configurable shader parameter `maxSh = 3`

**Alternative (fallback):**
- **GaussianSplats3D** — lighter but supports only **SH2**.

---

## Frontend Stack

| Component | Choice | Notes |
|------------|---------|-------|
| Framework | **React + Vite + pnpm** | Fast builds and efficient monorepo dependency management |
| Renderer | **Spark** | Handles SH3 and 3DGS-specific splat shaders |
| Viewer Integration | Dynamic import + URL/file loader | Example: `?src=https://cdn.example.com/scene.spz` |
| State/UI | React hooks or Zustand | Camera controls, exposure, fps overlay, SH toggles |
| Asset Format | **`.spz`** (Spark native) | Compact, progressive, streamable, preserves SH3 |

---

## Server-Side Stack (EKS)

| Layer | Component | Responsibility |
|-------|------------|----------------|
| Storage | **S3 bucket or RDS** | Stores `.spz` files (behind VPN) |
| Ingress | NGINX Ingress or AWS ALB | Routes subroute to viewer service |
| Backend Service | Node.js or FastAPI microservice | Future: `.ply` upload + conversion to `.spz` |
| Container | Docker image (NGINX or Node.js) | Serves frontend bundle + API endpoints |
| Monitoring | Thanos | Existing metrics scraping infrastructure |

**Note:** Deployment details deferred. Focus on local testing first.

---

## Data Conversion Pipeline
Convert `.ply` → **`.spz`** (Spark native format) to reduce file size (2–5× smaller) and enable progressive streaming while preserving SH3.

**Current Phase:** Pre-processed conversion (manual/offline)
**Future Phase:** Runtime conversion service for user uploads

Recommended tools:
- Spark’s CLI converters
- SuperSplat or PlayCanvas `SplatTransform` for preprocessing

**TODO:** Investigate Spark's `.spz` progressive streaming capabilities via HTTP range requests

---

## Development Phases

### Phase 1: Local Testing (Current Focus)
1. **Frontend Stack:**
   - Set up React + Vite + pnpm project
   - Integrate Spark renderer with SH3 support
   - Test with local `.ply` and `.spz` files
   - Verify SH3 rendering quality

2. **Backend Stack (Local):**
   - Simple file server for `.spz` assets
   - Test CORS and range request handling
   - Validate progressive loading behavior

3. **Conversion Testing:**
   - Convert sample `.ply` → `.spz` using Spark CLI
   - Measure file size reduction
   - Verify SH3 data preservation

### Phase 2: EKS Deployment (Deferred)
1. Containerize frontend + backend
2. Configure as subroute on existing EKS playground
3. Set up S3/RDS storage behind VPN
4. Integrate with existing Thanos monitoring

---

## Future Extensions
- Add **WebGPU renderer** (e.g., aczw/webgpu-gaussian-splat-viewer) behind feature detection
- Integrate web upload + conversion UI for `.ply` ingestion (runtime conversion service)
- **TODO:** Research and implement error handling:
  - Browser compatibility detection (WebGL 2 fallback)
  - Network error handling and retry logic
  - Loading states and progress indicators
  - Malformed file validation
  - CORS configuration for VPN environment

## Open Questions / TODOs
1. **Range Request Support:** Verify Spark's `.spz` format supports HTTP range requests for progressive streaming
2. **Error Handling:** Design fallback UI for unsupported browsers or failed loads
3. **Access Control:** Determine if VPN-only access is sufficient or if additional auth is needed
4. **Storage Decision:** S3 vs RDS for `.spz` file storage (evaluate based on access patterns)

---

## TL;DR
Use **Spark** (Three.js-based, SH3-capable) with a **React + Vite + pnpm** frontend.
Deploy as subroute on existing EKS playground (behind company VPN).
Convert `.ply` assets to **`.spz`** format for efficient delivery.
**Current focus:** Local testing of frontend and server stacks before EKS deployment.
