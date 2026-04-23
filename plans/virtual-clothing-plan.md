# Virtual Clothing / Formal Wear Replacement — Implementation Plan

> **Goal:** Replace the user's clothing with formal attire (suits, blazers, dresses, etc.) in a real-time or near-real-time webcam/video pipeline.
>
> **Hardware context:** RTX 4090 (24 GB VRAM), Ryzen 7950X3D, 64 GB DDR5, 4 TB NVMe, 5 Gbps internet.

---

## 1. Current State — No Built-In SDK Feature

The NVIDIA Maxine suite (AR SDK, Video Effects SDK, Audio Effects SDK) **does not include any clothing replacement or virtual try-on feature**.

---

## 2. Viable Local Options (RTX 4090)

### 🏆 Option A — RTV: Real-Time Per-Garment Virtual Try-On (Recommended for Live Webcam)

**What it is:** Open-source project (`ZaiqiangWu/RTV`) specifically designed for **real-time virtual try-on from a webcam**. Trains a dedicated lightweight network per garment.

**Performance:**
- **Real-time on RTX 3060+** — your RTX 4090 will have headroom to spare.
- Authors report **10–15 FPS** on older hardware; expect **20–30+ FPS** on your 4090.
- Uses per-garment training (~12 hours per garment on RTX 4090, ~3,000 images needed).

**How it works:**
- Does **NOT** remove the original garment. Instead, it synthesizes the new garment and overlays it on top of the body.
- Uses BodyMap + ReGarSyn networks with ConvLSTM for temporal consistency.
- Runs DensePose + SMPL-like body estimation for anchoring.

**Pros:**
- ✅ **Actually real-time** — designed for live webcam.
- ✅ Open-source, runs entirely local.
- ✅ Temporal consistency built-in (no flicker between frames).
- ✅ Perfect for "always wear a blazer" mode during video calls.

**Cons:**
- ⚠️ Requires **per-garment training** (~12 hours per item on your RTX 4090).
- ⚠️ Non-commercial license (Apache 2.0 with NC clause); commercial use requires author permission.
- ⚠️ Loose-fitting garments work better than tight ones.
- ⚠️ Body pose changes dramatically break the overlay.

**Feasibility:** ✅ **High for personal/non-commercial use.**

**Repo:** `https://github.com/ZaiqiangWu/RTV`

---

### 🏆 Option B — MagicTryOn (Recommended for Video/Pre-Recorded)

**What it is:** 2025 diffusion-transformer (DiT) based video virtual try-on. Uses **distribution-matching distillation** to compress sampling to **4 steps**.

**Performance:**
- Claims **real-time inference** after 4-step distillation.
- On RTX 4090, expect **5–15 FPS** for video try-on depending on resolution.
- Supports both upper and lower body garments.
- Preserves fine-grained details (lace, prints) and temporal stability.

**Pros:**
- ✅ **Near real-time** with distilled 4-step sampling.
- ✅ No per-garment training needed — zero-shot garment transfer.
- ✅ Handles video natively with garment-aware spatiotemporal RoPE.
- ✅ High fidelity on complex garments.

**Cons:**
- ⚠️ Code not yet released as of early 2025 (track releases).
- ⚠️ 4-step diffusion still has some quality trade-off vs full sampling.
- ⚠️ Requires significant VRAM (likely 16–24 GB for video).

**Feasibility:** ✅ **High — monitor for code release.**

**Paper:** `https://magic-tryon.com/`

---

### Option C — CatVTON / CatV2TON (Best Speed/Flexibility for Images)

**What it is:** Lightweight diffusion-based try-on. Only **899M total params**, **< 8 GB VRAM** for 1024×768.

**Performance on RTX 4090:**
- Image try-on: **~2–5 seconds per image** (depending on steps).
- **CatV2TON** (CVPR 2025 Workshop) adds **video try-on** support via DiT.
- No preprocessing needed — just concatenate person + garment images.

**Pros:**
- ✅ Extremely lightweight; runs easily on your hardware.
- ✅ No pose estimation or human parsing required.
- ✅ **CatVTON-FLUX** variant uses only 37.4M LoRA weights.
- ✅ Good for "Formal Snapshot" feature (still image).

**Cons:**
- ⚠️ Not real-time for video — 2–5 sec per frame.
- ⚠️ Video variant (CatV2TON) is new; performance unverified.

**Feasibility:** ✅ **High for still-image / snapshot use.** ⚠️ Medium for video.

**Repos:**
- `https://github.com/Zheng-Chong/CatVTON`
- `https://github.com/Zheng-Chong/CatV2TON`

---

### Option D — OOTDiffusion + TensorRT (Best Quality, Slower)

**What it is:** State-of-the-art diffusion try-on. Can be optimized heavily.

**Performance on RTX 4090:**
| Configuration | Time | Throughput |
|---|---|---|
| PyTorch baseline (20 steps) | ~7.5 s | 0.13 img/s |
| FP16 + attention slicing + UniPC | ~2.3 s | 0.43 img/s |
| **TensorRT FP16** | **~186 ms** | **~5.4 img/s** |
| TensorRT INT8 | ~112 ms | ~8.9 img/s |

**Pros:**
- ✅ Highest quality among open-source methods.
- ✅ TensorRT brings it to near-interactive speeds.
- ✅ 8K resolution support (~20 sec on RTX 4090).

**Cons:**
- ⚠️ Even with TensorRT, ~5 img/s is not true real-time video.
- ⚠️ TensorRT engine build takes 20–30 minutes per model.
- ⚠️ Temporal flicker between frames unless post-processed.

**Feasibility:** ⚠️ **Medium — best for batch processing, not live webcam.**

**Repo:** `https://github.com/levihsu/OOTDiffusion`

---

### Option E — VidClothEditor / ChronoTailor (Video Try-On Research)

| Model | Speed | Notes |
|---|---|---|
| **VidClothEditor** | LCM scheduler, 10 steps | Full-body inpainting; relaxed mask requirements. |
| **ChronoTailor** | Diffusion-based | Spatio-temporal attention for consistency. |

**Feasibility:** ⚠️ Medium — good for research/batch video editing. Not yet real-time.

---

### Option F — ComfyUI FLUX Fill Inpainting Workflow (Flexible but Slow)

**What it is:** General-purpose inpainting using FLUX Fill + BiRefNet segmentation.

**Performance on RTX 4090:**
- 512×512 inpaint crop: ~3.2 sec
- 1024×1024 inpaint crop: ~12.8 sec
- With FLUX Fill FP8: lower VRAM, similar speed.

**Use case:** Best for one-off creative edits, not live video.

---

### Option G — AR Overlay (Ultra-Fast, Low Quality)

**How it works:** Segment upper body with SAM2/MediaPipe, overlay pre-cut formal garment anchored to shoulder keypoints.

**Performance:** 30+ FPS easily.
**Quality:** Obviously fake — flat sticker effect.

**Feasibility:** ✅ High for novelty filters, ❌ Low for photorealistic formal wear.

---

## 3. Recommended Implementation Roadmap

### Phase 1 — "Formal Snapshot" (Immediate, 1 week)
- [ ] Integrate **CatVTON** for still-image try-on.
- [ ] User uploads: (1) webcam snapshot, (2) formal garment image.
- [ ] Pipeline returns result in ~2–5 seconds.
- [ ] Use this for profile pictures, LinkedIn photos, meeting prep.

### Phase 2 — Live Webcam Formal Overlay (2–3 weeks)
- [ ] Set up **RTV** (`ZaiqiangWu/RTV`).
- [ ] Train 3–5 formal garment models (blazer, suit jacket, dress shirt) — ~12 hours each on your RTX 4090.
- [ ] Integrate into webcam pipeline:
  - Capture frame → run RTV inference → output to v4l2loopback.
- [ ] Add toggle: enable/disable formal wear mode.
- [ ] Test at 720p / 1080p with your 7950X3D handling preprocessing.

### Phase 3 — Near-Real-Time Video Try-On (Monitor & Adopt)
- [ ] Monitor **MagicTryOn** for code release.
- [ ] If released, benchmark 4-step distilled inference on your RTX 4090.
- [ ] Target: 10+ FPS for pre-recorded video formalization.

### Phase 4 — Avatar Fallback (1 week)
- [ ] Pre-capture a frontal photo of user in formal wear (or generate via CatVTON).
- [ ] Use **LivePortrait** to drive this photo with user's live expressions.
- [ ] Output the animated formal-wear portrait as webcam feed.
- [ ] This bypasses live clothing synthesis entirely — always dressed up, always frontal.

---

## 4. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| RTV per-garment training is time-consuming | Medium | Train 3–5 core garments once; cache weights. |
| RTV non-commercial license | **BLOCKER** for commercial use | Contact authors for commercial license; or use CatVTON/OOTD instead. |
| MagicTryOn code not yet released | Medium | Monitor repo; fall back to RTV/CatVTON meanwhile. |
| OOTDiffusion TensorRT temporal flicker | High | Only use for still images; don't pipe to video directly. |
| Body pose changes break garment fit | High | RTV handles this better than diffusion per-frame; use torso tracking. |

---

## 5. Decision Matrix (Given Your Hardware)

| Approach | Real-Time | Quality | Per-Garment Training | License | Recommended? |
|---|---|---|---|---|---|
| AR Overlay (flat sticker) | ✅ 30+ FPS | ❌ Fake | ❌ No | OSS | Novelty only |
| **RTV** | ✅ **~20–30 FPS** | **Good** | **✅ Yes (~12h)** | **NC Apache** | **🏆 Live webcam** |
| **MagicTryOn** | ⚠️ **~5–15 FPS** | **Very High** | ❌ No | TBD | **🏆 Video (when released)** |
| CatVTON | ❌ ~2–5 s/frame | High | ❌ No | OSS | Still images |
| OOTDiffusion + TensorRT | ❌ ~5 img/s | Very High | ❌ No | OSS | Batch processing |
| LivePortrait Avatar | ✅ ~78 FPS | Stylized | ❌ No | OSS | Always-on avatar mode |

---

## 6. Summary

> **Your RTX 4090 + 7950X3D combo can absolutely run real-time or near-real-time virtual clothing replacement locally.**
>
> **For live webcam:** **RTV** is the only open-source project explicitly designed for real-time webcam try-on. Train a few formal garments and you have an "always wear a blazer" mode. The NC license is the main caveat.
>
> **For highest quality video:** Monitor **MagicTryOn** — its 4-step distilled DiT pipeline promises real-time video try-on with garment preservation. When code drops, it will likely run well on your 24 GB VRAM.
>
> **For still images:** **CatVTON** is lightning-fast to set up and gives excellent results in seconds.
>
> **For a guaranteed-always-formal fallback:** Combine CatVTON (generate one formal portrait) + LivePortrait (animate it with expressions) = real-time formal avatar.
