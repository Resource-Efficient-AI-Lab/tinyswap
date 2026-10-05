package com.example.miniswap

import android.media.MediaCodec
import android.media.MediaCodecInfo
import android.media.MediaExtractor
import android.media.MediaFormat
import android.media.MediaMuxer
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import java.nio.ByteBuffer

/**
 * 视频换脸:mp4 -> 逐帧换脸 -> mp4,音频原样复制。
 * 视频:解码YUV -> BGR Mat -> 换脸 -> YUV -> 编码。
 * 音频:压缩包直接从输入搬到输出,不解码。
 * 编解码两侧都按Image的实际stride/pixelStride布局读写,自适应planar/semi-planar设备差异。
 */
class VideoSwapper(private val engine: FaceSwapEngine) {
    private val TAG = "VideoSwapper"
    private val TIMEOUT_US = 10000L
    private var nsDecode = 0L
    private var nsYuvToBgr = 0L
    private var nsBgrToYuv = 0L

    fun process(inputPath: String, outputPath: String, onProgress: (Int) -> Unit) {
        // ---------- 找视频轨和音频轨 ----------
        val vExtractor = MediaExtractor()
        vExtractor.setDataSource(inputPath)
        var videoTrack = -1
        var audioTrack = -1
        var vInFormat: MediaFormat? = null
        var aInFormat: MediaFormat? = null
        for (i in 0 until vExtractor.trackCount) {
            val fmt = vExtractor.getTrackFormat(i)
            val mime = fmt.getString(MediaFormat.KEY_MIME)!!
            if (mime.startsWith("video/") && videoTrack < 0) {
                videoTrack = i; vInFormat = fmt
            } else if (mime.startsWith("audio/") && audioTrack < 0) {
                audioTrack = i; aInFormat = fmt
            }
        }
        require(videoTrack >= 0) { "No video track found" }
        val hasAudio = audioTrack >= 0
        Log.i(TAG, "videoTrack=$videoTrack audioTrack=$audioTrack (hasAudio=$hasAudio)")

        val width = vInFormat!!.getInteger(MediaFormat.KEY_WIDTH)
        val height = vInFormat.getInteger(MediaFormat.KEY_HEIGHT)
        val mime = vInFormat.getString(MediaFormat.KEY_MIME)!!
        val frameRate = if (vInFormat.containsKey(MediaFormat.KEY_FRAME_RATE))
            vInFormat.getInteger(MediaFormat.KEY_FRAME_RATE) else 30

        // ---------- 解码器 ----------
        vExtractor.selectTrack(videoTrack)
        val decoder = MediaCodec.createDecoderByType(mime)
        vInFormat.setInteger(
            MediaFormat.KEY_COLOR_FORMAT,
            MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
        )
        decoder.configure(vInFormat, null, null, 0)
        decoder.start()

        // ---------- 编码器 ----------
        val encW = width - width % 2
        val encH = height - height % 2
        val encFormat = MediaFormat.createVideoFormat("video/avc", encW, encH).apply {
            setInteger(
                MediaFormat.KEY_COLOR_FORMAT,
                MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible
            )
            setInteger(MediaFormat.KEY_BIT_RATE, encW * encH * 4)
            setInteger(MediaFormat.KEY_FRAME_RATE, frameRate)
            setInteger(MediaFormat.KEY_I_FRAME_INTERVAL, 1)
        }
        val encoder = MediaCodec.createEncoderByType("video/avc")
        encoder.configure(encFormat, null, null, MediaCodec.CONFIGURE_FLAG_ENCODE)
        encoder.start()

        // ---------- muxer ----------
        val muxer = MediaMuxer(outputPath, MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4)
        var muxVideoTrack = -1
        var muxAudioTrack = -1
        var muxerStarted = false

        // 音频用独立extractor读
        val aExtractor = MediaExtractor()
        if (hasAudio) {
            aExtractor.setDataSource(inputPath)
            aExtractor.selectTrack(audioTrack)
            muxAudioTrack = muxer.addTrack(aInFormat!!)
        }

        val bufInfo = MediaCodec.BufferInfo()
        var frameCount = 0
        var decodeEOS = false
        var encodeEOS = false
        val pendingFrames = ArrayDeque<Pair<ByteArray, Long>>()

        // ---------- 视频处理主循环 ----------
        while (!encodeEOS) {
            // 1. 喂解码器
            if (!decodeEOS) {
                val inIdx = decoder.dequeueInputBuffer(TIMEOUT_US)
                if (inIdx >= 0) {
                    val inBuf = decoder.getInputBuffer(inIdx)!!
                    val sampleSize = vExtractor.readSampleData(inBuf, 0)
                    if (sampleSize < 0) {
                        decoder.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        decodeEOS = true
                    } else {
                        decoder.queueInputBuffer(inIdx, 0, sampleSize, vExtractor.sampleTime, 0)
                        vExtractor.advance()
                    }
                }
            }

            // 2. 解码输出 -> 换脸 -> 队列
            val tDec = System.nanoTime()
            val outIdx = decoder.dequeueOutputBuffer(bufInfo, TIMEOUT_US)
            nsDecode += System.nanoTime() - tDec
            if (outIdx >= 0) {
                if (bufInfo.size > 0) {
                    val image = decoder.getOutputImage(outIdx)
                    if (image != null) {
                        var t = System.nanoTime()
                        val bgr = imageToBgr(image, width, height)
                        nsYuvToBgr += System.nanoTime() - t

                        val swapped = engine.swapFrame(bgr)

                        val cropped = if (swapped.cols() != encW || swapped.rows() != encH)
                            Mat(swapped, Rect(0, 0, encW, encH)) else swapped
                        t = System.nanoTime()
                        val yuv = bgrToYuvI420(cropped, encW, encH)
                        nsBgrToYuv += System.nanoTime() - t

                        pendingFrames.addLast(Pair(yuv, bufInfo.presentationTimeUs))
                        bgr.release()
                        if (swapped !== cropped) swapped.release()
                        cropped.release()
                        frameCount++
                        onProgress(frameCount)
                    }
                }
                decoder.releaseOutputBuffer(outIdx, false)
            }

            // 3. 喂编码器(按编码器Image的实际布局写入,自适应planar/semi-planar)
            while (pendingFrames.isNotEmpty()) {
                val encInIdx = encoder.dequeueInputBuffer(TIMEOUT_US)
                if (encInIdx < 0) break
                val (yuv, pts) = pendingFrames.removeFirst()
                val inImage = encoder.getInputImage(encInIdx)
                if (inImage != null) {
                    writeI420ToImage(yuv, inImage, encW, encH)
                    encoder.queueInputBuffer(encInIdx, 0, encW * encH * 3 / 2, pts, 0)
                } else {
                    // 极少数编码器不支持Image模式,退回ByteBuffer(planar假设)
                    val encInBuf = encoder.getInputBuffer(encInIdx)!!
                    encInBuf.clear()
                    encInBuf.put(yuv)
                    encoder.queueInputBuffer(encInIdx, 0, yuv.size, pts, 0)
                }
            }
            if (decodeEOS && pendingFrames.isEmpty()) {
                val encInIdx = encoder.dequeueInputBuffer(TIMEOUT_US)
                if (encInIdx >= 0) {
                    encoder.queueInputBuffer(encInIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                }
            }

            // 4. 编码输出 -> muxer
            val encOutIdx = encoder.dequeueOutputBuffer(bufInfo, TIMEOUT_US)
            if (encOutIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                muxVideoTrack = muxer.addTrack(encoder.outputFormat)
                muxer.start()
                muxerStarted = true
            } else if (encOutIdx >= 0) {
                val encOutBuf = encoder.getOutputBuffer(encOutIdx)!!
                if (bufInfo.size > 0 && muxerStarted &&
                    (bufInfo.flags and MediaCodec.BUFFER_FLAG_CODEC_CONFIG) == 0
                ) {
                    encOutBuf.position(bufInfo.offset)
                    encOutBuf.limit(bufInfo.offset + bufInfo.size)
                    muxer.writeSampleData(muxVideoTrack, encOutBuf, bufInfo)
                }
                encoder.releaseOutputBuffer(encOutIdx, false)
                if ((bufInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0) {
                    encodeEOS = true
                }
            }
        }

        // ---------- 复制音频轨 ----------
        if (hasAudio && muxerStarted) {
            val audioBuf = ByteBuffer.allocate(256 * 1024)
            val audioInfo = MediaCodec.BufferInfo()
            aExtractor.seekTo(0, MediaExtractor.SEEK_TO_CLOSEST_SYNC)
            while (true) {
                val size = aExtractor.readSampleData(audioBuf, 0)
                if (size < 0) break
                audioInfo.offset = 0
                audioInfo.size = size
                audioInfo.presentationTimeUs = aExtractor.sampleTime
                audioInfo.flags = if (aExtractor.sampleFlags and MediaExtractor.SAMPLE_FLAG_SYNC != 0)
                    MediaCodec.BUFFER_FLAG_KEY_FRAME else 0
                muxer.writeSampleData(muxAudioTrack, audioBuf, audioInfo)
                aExtractor.advance()
            }
            Log.i(TAG, "Audio track copied")
        }

        // 计时报告
        val f = frameCount.toDouble()
        if (frameCount > 0) {
            Log.i(TAG, "===== Pipeline avg per frame ($frameCount frames) =====")
            Log.i(TAG, "  decode wait  %.1f ms".format(nsDecode / f / 1e6))
            Log.i(TAG, "  YUV->BGR     %.1f ms".format(nsYuvToBgr / f / 1e6))
            Log.i(TAG, "  BGR->YUV     %.1f ms".format(nsBgrToYuv / f / 1e6))
            engine.reportTiming(TAG)
        }
        Log.i(TAG, "Done, $frameCount frames -> $outputPath")
        decoder.stop(); decoder.release()
        encoder.stop(); encoder.release()
        muxer.stop(); muxer.release()
        vExtractor.release()
        aExtractor.release()
    }

    /** 解码器Image(任意布局) -> 标准I420 -> BGR Mat */
    private fun imageToBgr(image: android.media.Image, w: Int, h: Int): Mat {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val ySize = w * h
        val uvSize = w * h / 4
        val out = ByteArray(ySize + uvSize * 2)

        // Y平面,逐行按rowStride
        var pos = 0
        val rowY = ByteArray(yRowStride)
        yBuf.position(0)
        for (row in 0 until h) {
            yBuf.position(row * yRowStride)
            val remaining = minOf(yRowStride, yBuf.remaining())
            yBuf.get(rowY, 0, remaining)
            System.arraycopy(rowY, 0, out, pos, w)
            pos += w
        }

        // U平面(按pixelStride取样)
        val uvH = h / 2
        val uvW = w / 2
        val rowUV = ByteArray(uvRowStride)
        uBuf.position(0)
        for (row in 0 until uvH) {
            uBuf.position(row * uvRowStride)
            val remaining = minOf(uvRowStride, uBuf.remaining())
            uBuf.get(rowUV, 0, remaining)
            for (col in 0 until uvW) {
                out[pos++] = rowUV[col * uvPixelStride]
            }
        }

        // V平面
        vBuf.position(0)
        for (row in 0 until uvH) {
            vBuf.position(row * uvRowStride)
            val remaining = minOf(uvRowStride, vBuf.remaining())
            vBuf.get(rowUV, 0, remaining)
            for (col in 0 until uvW) {
                out[pos++] = rowUV[col * uvPixelStride]
            }
        }

        val yuvMat = Mat(h + h / 2, w, CvType.CV_8UC1)
        yuvMat.put(0, 0, out)
        val bgr = Mat()
        Imgproc.cvtColor(yuvMat, bgr, Imgproc.COLOR_YUV2BGR_I420)
        yuvMat.release()
        image.close()
        return bgr
    }

    /** BGR Mat -> 标准I420字节 */
    private fun bgrToYuvI420(bgr: Mat, w: Int, h: Int): ByteArray {
        val yuv = Mat()
        Imgproc.cvtColor(bgr, yuv, Imgproc.COLOR_BGR2YUV_I420)
        val out = ByteArray(w * h * 3 / 2)
        yuv.get(0, 0, out)
        yuv.release()
        return out
    }

    /** 标准I420字节 -> 编码器输入Image(按其实际stride/pixelStride布局写入) */
    private fun writeI420ToImage(i420: ByteArray, image: android.media.Image, w: Int, h: Int) {
        val ySize = w * h
        val uvW = w / 2
        val uvH = h / 2
        val uvSize = uvW * uvH

        // Y平面
        val yPlane = image.planes[0]
        val yBuf = yPlane.buffer
        val yRowStride = yPlane.rowStride
        for (row in 0 until h) {
            yBuf.position(row * yRowStride)
            yBuf.put(i420, row * w, w)
        }

        // U平面
        val uPlane = image.planes[1]
        val uBuf = uPlane.buffer
        val uRowStride = uPlane.rowStride
        val uPixStride = uPlane.pixelStride
        for (row in 0 until uvH) {
            val base = row * uRowStride
            val src = ySize + row * uvW
            if (uPixStride == 1) {
                uBuf.position(base)
                uBuf.put(i420, src, uvW)
            } else {
                for (col in 0 until uvW) {
                    uBuf.put(base + col * uPixStride, i420[src + col])
                }
            }
        }

        // V平面
        val vPlane = image.planes[2]
        val vBuf = vPlane.buffer
        val vRowStride = vPlane.rowStride
        val vPixStride = vPlane.pixelStride
        for (row in 0 until uvH) {
            val base = row * vRowStride
            val src = ySize + uvSize + row * uvW
            if (vPixStride == 1) {
                vBuf.position(base)
                vBuf.put(i420, src, uvW)
            } else {
                for (col in 0 until uvW) {
                    vBuf.put(base + col * vPixStride, i420[src + col])
                }
            }
        }
    }
}