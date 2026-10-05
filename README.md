# TinySwap (MiniSwap2)

TinySwap is an Android app that does **on-device face swapping on video**. It takes one source face image and a target video, swaps the source identity onto every detected face in the video, and writes a new MP4. Inference runs locally through ONNX Runtime. Nothing is sent to a server.

The code was written by Zhoulun as "MiniSwap2" and is published here by the Resource-Efficient AI Lab.

> **Responsible use:** this is research software for studying efficient on-device generative models. Only use faces and footage you have the rights and consent to use.

## Pipeline

```
source image ──► SCRFD (detect + 5 keypoints) ──► align 224×224 ──► ArcFace ──► id embedding + feature map
                                                                                      │
                                                                                      ▼
                                                                  IIM (identity injection module)
                                                                  produces 18 depthwise/pointwise
                                                                  conv weights for the IDN
                                                                                      │
                                       injected as ONNX Runtime initializers ◄────────┘
                                                      │
target video frame ──► SCRFD ──► align ──► IDN (id network, 224×224) ──► image + mask ──► de-align & blend ──► output frame
```

- **Once per source face** (`FaceSwapEngine.injectSource`): ArcFace and IIM run a single time. The 18 generated weight tensors replace the matching initializers of `idn_template.onnx` through `SessionOptions.addInitializer`, which yields an identity-specific IDN session.
- **Every frame** (`FaceSwapEngine.swapFrame`): the frame goes through detection, alignment, one IDN forward pass, then blending back with the predicted mask. Frames with no detected face pass through unchanged.
- **Video I/O** (`VideoSwapper`): MediaCodec decodes YUV, converts to BGR `Mat`, swaps the face, converts back to YUV and re-encodes. The audio track is copied over without re-encoding.

### Source files (`app/src/main/java/com/example/miniswap/`)

| File | Role |
|---|---|
| `MainActivity.kt` | Entry point. Starts `VerifyStage2.runVideoSwap` on a background thread. |
| `FaceSwapEngine.kt` | Swap engine: weight injection plus per-frame swapping, with timing stats. |
| `ScrfdDetector.kt` | SCRFD face and 5-keypoint detector (letterbox, decode, NMS). |
| `FaceUtils.kt` | Similarity-transform alignment, de-alignment and mask blending (OpenCV). |
| `VideoSwapper.kt` | MP4 → per-frame swap → MP4 with MediaCodec/MediaMuxer. |
| `VerifyStage2.kt` | Demo driver (`runVideoSwap`) plus development checks that compare on-device tensors with reference dumps. |

Code comments and log messages are mostly in Chinese.

## Requirements

- Android Studio with Android Gradle Plugin 9.1.x support (Gradle 9.3.1 is pinned by the wrapper)
- JDK 21 for the Gradle daemon (`gradle/gradle-daemon-jvm.properties`; it is auto-provisioned through the foojay resolver)
- Android SDK platform 36 (compileSdk 36.1, targetSdk 36)
- An arm64 Android device running **Android 8.0 / API 26 or newer**. A physical device is recommended because the models are large and video encoding is slow on emulators.
- Dependencies come from Maven: `com.microsoft.onnxruntime:onnxruntime-android:1.20.0` and `org.opencv:opencv:4.10.0`. You do not need a local OpenCV SDK checkout.

## Models

The model weights are **not stored in this repository** (see [Provenance](#provenance-and-licensing)). Put them in `app/src/main/assets/`:

| File | Size | Used by | SHA-256 |
|---|---|---|---|
| `scrfd_500m.onnx` | 2.5 MB | Face detector used by the app | `5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a` |
| `arcface.onnx` | 209 MB | Identity encoder (runs once per source) | `8638062c828d8da953d6e3a9d1129beb5216eba5ca37d804fd8cd944352e199b` |
| `iim.onnx` | 82.5 MB | Identity injection module (runs once per source) | `27ceb17b5a5007b671b20e38104a52b3e257880d52ad49d1ad7aa1145b530e57` |
| `idn_template.onnx` | 2.0 MB | Per-frame ID network; its weights are overwritten at runtime | `51b17498e5beadf4fa211235b0e9879d86b9270fe7574295d4fbb2e360b5e1e7` |
| `scrfd.onnx` | 16.9 MB | Larger detector. Only the legacy checks in `VerifyStage2` use it (`runFullPipeline`). | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |

`ScrfdDetector.outputNames` is hard-coded to the output tensor names of the `scrfd_500m.onnx` export above. If you use a different SCRFD export, update those names.

Two small JSON files that are committed to the repo are also read at runtime:

- `deploy_contract.json`: IIM output order, IDN initializer names and preprocessing recipe
- `verify/manifest.json`: tensor shapes used during weight injection

Check the weights after copying them:

```sh
cd app/src/main/assets && shasum -a 256 *.onnx
```

### Demo inputs

The demo flow in `VerifyStage2.runVideoSwap` expects two more assets that are **not** included because they are third-party media:

- `app/src/main/assets/source2.jpg`: a photo containing the source face
- `app/src/main/assets/test.mp4`: the target video

Supply your own. The swapped video is written to the app's external files directory as `swapped.mp4` and plays back in the app when processing finishes.

## Build and run

```sh
git clone https://github.com/Resource-Efficient-AI-Lab/tinyswap.git
cd tinyswap
# 1. copy the .onnx files plus source2.jpg and test.mp4 into app/src/main/assets/
# 2. point Gradle at your Android SDK (Android Studio does this automatically):
echo "sdk.dir=$HOME/Library/Android/sdk" > local.properties   # macOS default path
# 3. build and install
./gradlew :app:assembleDebug
./gradlew :app:installDebug
```

You can also open the project in Android Studio and run the `app` configuration. Progress and per-stage timings are logged under the tags `VerifyStage2` and `VideoSwapper`:

```sh
adb logcat -s VerifyStage2 VideoSwapper
```

## Provenance and licensing

**Not yet confirmed. To be verified with Zhoulun before any weights are published.**

- `arcface.onnx` and `iim.onnx` were exported with paddle2onnx from PaddlePaddle. The ArcFace + IIM + IDN design matches Baidu's MobileFaceSwap (AAAI 2022). We still need to confirm where the weights came from and under what license they were released.
- `scrfd*.onnx` were exported from PyTorch. The sizes match InsightFace's SCRFD-500M-KPS and SCRFD-10G-KPS. InsightFace's pretrained models are generally released for **non-commercial research use only**.
- `idn_template.onnx` was presumably produced by this project from the IDN architecture. To be confirmed.

## License

No license yet. We are still confirming the licensing and provenance of the original MiniSwap2 code and the model weights, and will choose a license after that. Until then, all rights are reserved.
