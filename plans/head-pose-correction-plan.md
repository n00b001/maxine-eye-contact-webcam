# Head Pose Correction / Face Frontalization — Implementation Plan

> **Goal:** Rotate the user's entire head to face the camera in a real-time webcam pipeline, going beyond eye-gaze redirection.
>
> **Hardware context:** RTX 4090 (24 GB VRAM), Ryzen 7950X3D, 64 GB DDR5, 4 TB NVMe, 5 Gbps internet.

---

## 1. Current State — AR SDK Limitations

The existing pipeline uses **NVIDIA Maxine AR SDK `NvAR_Feature_GazeRedirection`**.

| Capability | Status |
|---|---|
| Eye gaze redirected toward camera | ✅ Supported |
| Head pose used as input guard | ✅ Output: `HeadPose` quaternion, `OutputHeadTranslation` |
| Head pixels rotated to face camera | ❌ **Not supported** |
| Face re-rendered from novel viewpoint | ❌ **Not supported** |

The AR SDK's `GazeRedirection` uses head-pitch/yaw **thresholds only** to decide *when* to disengage eye redirection. It never modifies the head itself.

---

## 2. Viable Local Options (RTX 4090)

### 🏆 Option A — LivePortrait (Recommended)

**What it is:** Open-source portrait animation framework by Kuaishou (KwaiVGI). Uses implicit keypoints + stitching/retargeting modules.

**Why it solves head pose correction:**
- Supports **absolute head pose specification** via Euler angles (pitch, yaw, roll).
- Setting `flag_relative_motion = False` + `animation_region = "pose"` forces the source face into an arbitrary target pose while preserving identity.
- **Inference speed: 12.8 ms/frame = ~78 FPS on RTX 4090** (PyTorch, no TensorRT yet).
- Community **FasterLivePortrait** project adds TensorRT acceleration.
- `--flag_do_torch_compile` gives an extra 20–30% speedup.

**Head-pose-correction pipeline:**
1. Capture webcam frame → crop face region.
2. Feed into LivePortrait as **source** image.
3. Set target pose: `pitch=0, yaw=0, roll=0` (frontal).
4. Set `animation_region="pose"` so only head rotation changes; expressions stay natural.
5. Output is the same person, now facing the camera.

**Pros:**
- ✅ **Real-time** on your hardware (78+ FPS).
- ✅ Absolute pose control — perfect for frontalization.
- ✅ Open-source, runs entirely local.
- ✅ Stitching module gives pixel-perfect face integration.

**Cons:**
- ⚠️ When the source head is turned > 45°, the model must hallucinate the occluded side of the face — artifacts possible.
- ⚠️ Hair and neck boundaries can show seams; works best when combined with a small crop around the face.
- ⚠️ Shoulders/torso are not rotated — only the face region.

**Feasibility:** ✅ **High — this is the best near-term solution.**

**Repos:**
- `https://github.com/KwaiVGI/LivePortrait`
- `https://github.com/FasterLivePortrait` (community TensorRT fork)

---

### Option B — SoulX-FlashHead / X-NeMo / Takin-ADA (Alternatives)

| Model | Speed (RTX 4090) | Notes |
|---|---|---|
| **SoulX-FlashHead-Lite** | 96 FPS | Audio-driven talking head. Real-time but focused on lip-sync, not pose control. |
| **Takin-ADA** | 42 FPS @ 512×512 | High-res facial animation. Research code availability unclear. |
| **X-NeMo** | Very fast | Instant feedforward Gaussian head avatar from NVIDIA research. 3D-aware, expressive. Not yet open-sourced. |

**Verdict:** LivePortrait is more mature and directly controllable for pose. Monitor X-NeMo for future integration.

---

### Option C — 3D Face Mesh Reconstruction (DECA / EMOCA)

**How it works:** Fit a 3DMM to the face, rotate mesh to frontal, re-render.

**Pros:** Deterministic, no hallucination of identity.
**Cons:** Looks synthetic / uncanny. Hair, glasses, neck boundaries are hard. Not real-time at 30 fps without heavy optimization.

**Feasibility:** ⚠️ Medium — doable as R&D prototype, not production-ready.

---

### Option D — Neural Face Frontalization (Offline Only)

- **eMotion-GAN** (2024) — motion-based GAN; seconds per frame.
- **LLM-Based Pose Normalization** (Qwen-image-edit) — ~seconds per frame on A100.

**Feasibility:** ❌ Not real-time — skip for live video.

---

## 3. Recommended Implementation Roadmap

### Phase 1 — LivePortrait Head Pose Correction (1–2 weeks)
- [ ] Clone `KwaiVGI/LivePortrait` and verify 78 FPS on RTX 4090.
- [ ] Implement a wrapper that:
  - Reads webcam frame.
  - Detects face crop (use InsightFace or MediaPipe).
  - Feeds crop into LivePortrait with `animation_region="pose"`, `flag_relative_motion=False`, target Euler = `(0, 0, 0)`.
  - Pastes result back into original frame.
- [ ] Add a `--head-pose-frontal` toggle to the existing gaze-redirection pipeline.
- [ ] Evaluate artifacts at yaw angles of 15°, 30°, 45°.

### Phase 2 — Optimization & Integration (1 week)
- [ ] Try `torch.compile` on the LivePortrait pipeline for +20–30% speed.
- [ ] Evaluate community **FasterLivePortrait** TensorRT fork.
- [ ] Combine with existing AR SDK gaze redirection:
  - Run Maxine GazeRedirection on the frontalized face for eye contact.
  - OR disable Maxine eye redirection and let LivePortrait handle both head + expression (it has eye/lip retargeting too).

### Phase 3 — Hybrid Fallback (1 week)
- [ ] If LivePortrait artifacts are too strong at extreme angles (> 40° yaw), fall back to:
  - A pre-captured frontal photo of the user animated by their live expressions (avatar mode).
  - This is essentially "always frontal, always dressed up" — same concept as clothing plan Phase 4.

### Phase 4 — Monitor Future Releases
- [ ] Track **X-NeMo** (NVIDIA instant Gaussian head avatar) for open-source release.
- [ ] Track LivePortrait v3+ for quality improvements.

---

## 4. Key Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| LivePortrait hallucinates occluded face side at > 45° | High | Limit activation to < 40° yaw; use avatar fallback beyond that. |
| Neck/hair seams visible | Medium | Tight face crop + feathered alpha blending around edges. |
| Latency spikes from face detection | Medium | Reuse previous crop region; only run detector every N frames. |
| Shoulders remain turned while face is frontal | Low | Acceptable for video calls; focus is on face anyway. |

---

## 5. Decision Matrix (Given Your Hardware)

| Approach | Real-Time | Quality | Effort | Controllable Pose | Recommended? |
|---|---|---|---|---|---|
| AR SDK GazeRedirection only | ✅ Yes | Eye-only | Done | ❌ No | Current baseline |
| **LivePortrait** | ✅ **~78 FPS** | **High** | **1–2 wks** | **✅ Absolute Euler** | **🏆 Primary choice** |
| DECA/EMOCA 3DMM | ⚠️ ~10 fps | Synthetic | 2–3 wks | ✅ Yes | Experimental only |
| Diffusion per-frame | ❌ No | Very High | 1–2 wks | ✅ Yes | Offline use only |
| X-NeMo (future) | ✅ Expected | Very High | Unknown | ✅ Yes | Monitor |

---

## 6. Summary

> **LivePortrait is a mature, open-source, real-time solution that can absolutely run on your RTX 4090 to correct head pose.**
>
> By forcing `animation_region="pose"` with `flag_relative_motion=False` and setting target Euler angles to `(0,0,0)`, you get genuine face frontalization at ~78 FPS. The main limitation is artifacting when the source head is turned very far (> 45°), but for typical video conferencing ranges (±30°) it should work well.
>
> **Next step:** Integrate LivePortrait as a post-processing stage after (or in place of) the AR SDK gaze redirection.
