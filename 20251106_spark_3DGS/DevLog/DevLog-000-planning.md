# 3D Gaussian Splatting Viewer Prototype — Architecture Summary

## Objective
Serve an interactive 3D viewer for large (~120 MB) `.ply` Gaussian Splatting files within an **EKS cluster**, using open-source WebGL/WebGPU-based technologies.

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
| Asset Format | `.spz` or `.ksplat` | Compact, progressive, streamable |

---

## Server-Side Stack (EKS)

| Layer | Component | Responsibility |
|-------|------------|----------------|
| Static Hosting | S3 bucket | Stores `.spz`/`.ply` and frontend bundle |
| CDN | CloudFront | Caching, Range requests, gzip/brotli |
| Ingress | NGINX Ingress or AWS ALB | Routes `/` to viewer and `/assets/*` to CDN |
| Optional Service | Node.js or FastAPI microservice | Uploads, `.ply → .spz` conversion, thumbnails |
| Container | Docker image running NGINX or Vite preview | Minimal viewer runtime |
| Scaling | HPA-enabled pods | Low compute demand since rendering is client-side |

---

## Data Conversion Pipeline
Convert `.ply` → `.spz` (Spark) or `.ksplat` (GaussianSplats3D) to reduce file size (2–5× smaller) and enable progressive streaming while preserving SH3.

Recommended tools:
- Spark’s CLI converters  
- SuperSplat or PlayCanvas `SplatTransform` for preprocessing

---

## Deployment Overview

1. Build frontend: `pnpm run build`
2. Sync build output to S3: `aws s3 sync dist/ s3://your-viewer-bucket/`
3. Deploy on EKS:
   - **Deployment:** containerized Spark viewer
   - **Service:** ClusterIP or LoadBalancer
   - **Ingress:** routes `/` to viewer; heavy assets via CloudFront
4. Access via: `https://viewer.example.com/?src=https://cdn.example.com/scene.spz`

---

## Future Extensions
- Add **WebGPU renderer** (e.g., aczw/webgpu-gaussian-splat-viewer) behind feature detection.
- Integrate web upload + conversion UI for `.ply` ingestion.
- Optional auth with AWS Cognito or OIDC.

---

## TL;DR
Use **Spark** (Three.js-based, SH3-capable) with a **React + Vite + pnpm** frontend, hosted via **S3 + CloudFront** and proxied through **EKS Ingress**.  
Convert `.ply` assets to `.spz` for efficient delivery and rely on Spark’s native SH3 rendering.
