package com.example.miniswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.max
import kotlin.math.min

/** SCRFD人脸检测,输出最高分脸的5点关键点(原图坐标) */
class ScrfdDetector(
    private val env: OrtEnvironment,
    modelPath: String,
    private val inputSize: Int = 640      // 可调:640(原始) / 320(4倍快)
) {

    private val session = env.createSession(modelPath, OrtSession.SessionOptions())
    private val inputName = session.inputNames.first()

    private val inputStd = 128.0f
    private val inputMean = 127.5f
    private val strides = intArrayOf(8, 16, 32)
    private val numAnchors = 2
    private val fmc = 3
    private val detThresh = 0.6f
    private val nmsThresh = 0.4f
    private val outputNames = arrayOf("443", "468", "493", "446", "471", "496", "449", "474", "499")
    // ---- 内部计时(纳秒累加) ----
    var callCount = 0L
    var nsLetterbox = 0L
    var nsBlob = 0L
    var nsInfer = 0L
    var nsDecode = 0L

    fun reportTiming(tag: String) {
        if (callCount == 0L) return
        val f = callCount.toDouble()
        android.util.Log.i(tag, "  --- SCRFD内部分解(输入${inputSize}) ---")
        android.util.Log.i(tag, "    letterbox  %.1f ms".format(nsLetterbox / f / 1e6))
        android.util.Log.i(tag, "    blob前处理 %.1f ms".format(nsBlob / f / 1e6))
        android.util.Log.i(tag, "    ORT推理    %.1f ms".format(nsInfer / f / 1e6))
        android.util.Log.i(tag, "    解码+NMS   %.1f ms".format(nsDecode / f / 1e6))
    }

    class Face(val score: Float, val bbox: FloatArray, val kps: Array<FloatArray>)

    fun detectBestKps(imgBgr: Mat): Array<DoubleArray>? {
        callCount++
        var t = System.nanoTime()

        val imgH = imgBgr.rows()
        val imgW = imgBgr.cols()
        val imRatio = imgH.toDouble() / imgW
        val newW: Int
        val newH: Int
        if (imRatio > 1.0) {
            newH = inputSize
            newW = (newH / imRatio).toInt()
        } else {
            newW = inputSize
            newH = (newW * imRatio).toInt()
        }
        val detScale = newH.toDouble() / imgH

        val resized = Mat()
        Imgproc.resize(imgBgr, resized, Size(newW.toDouble(), newH.toDouble()))
        val detImg = Mat.zeros(Size(inputSize.toDouble(), inputSize.toDouble()), imgBgr.type())
        resized.copyTo(detImg.submat(Rect(0, 0, newW, newH)))
        resized.release()
        nsLetterbox += System.nanoTime() - t

        t = System.nanoTime()
        val input = blobFromImage(detImg)
        detImg.release()
        nsBlob += System.nanoTime() - t

        t = System.nanoTime()
        val inputTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(input),
            longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
        )
        val netOuts = HashMap<String, FloatArray>()
        val outNameSet = HashSet<String>()
        for (n in outputNames) outNameSet.add(n)
        session.run(mapOf(inputName to inputTensor), outNameSet).use { res ->
            for (name in outputNames) {
                val opt = res.get(name)
                if (opt.isPresent) {
                    netOuts[name] = toFloats(opt.get() as OnnxTensor)
                }
            }
        }
        inputTensor.close()
        nsInfer += System.nanoTime() - t

        t = System.nanoTime()
        val faces = ArrayList<Face>()
        for (sIdx in strides.indices) {
            val stride = strides[sIdx]
            val scores = netOuts[outputNames[sIdx]] ?: continue
            val bboxPreds = netOuts[outputNames[sIdx + fmc]] ?: continue
            val kpsPreds = netOuts[outputNames[sIdx + fmc * 2]] ?: continue

            val h = inputSize / stride
            val w = inputSize / stride
            val numPoints = h * w * numAnchors
            for (i in 0 until numPoints) {
                if (i >= scores.size) break
                val score = scores[i]
                if (score < detThresh) continue
                val gridIdx = i / numAnchors
                val gy = gridIdx / w
                val gx = gridIdx % w
                val cx = (gx * stride).toFloat()
                val cy = (gy * stride).toFloat()
                val x1 = cx - bboxPreds[i * 4 + 0] * stride
                val y1 = cy - bboxPreds[i * 4 + 1] * stride
                val x2 = cx + bboxPreds[i * 4 + 2] * stride
                val y2 = cy + bboxPreds[i * 4 + 3] * stride
                val kps = Array(5) { k ->
                    floatArrayOf(
                        cx + kpsPreds[i * 10 + k * 2] * stride,
                        cy + kpsPreds[i * 10 + k * 2 + 1] * stride
                    )
                }
                faces.add(Face(score, floatArrayOf(x1, y1, x2, y2), kps))
            }
        }

        if (faces.isEmpty()) { nsDecode += System.nanoTime() - t; return null }
        val kept = nms(faces)
        if (kept.isEmpty()) { nsDecode += System.nanoTime() - t; return null }

        var best = kept[0]
        for (f in kept) if (f.score > best.score) best = f
        nsDecode += System.nanoTime() - t

        return Array(5) { k ->
            doubleArrayOf(best.kps[k][0] / detScale, best.kps[k][1] / detScale)
        }
    }

    /** 优化版:一次性取Mat数据,纯Java循环转换(JNI调用从百万次降到1次) */
    private fun blobFromImage(bgr: Mat): FloatArray {
        val n = inputSize * inputSize
        val bytes = ByteArray(n * 3)
        bgr.get(0, 0, bytes)          // 1次JNI调用
        val out = FloatArray(3 * n)
        var i = 0
        for (p in 0 until n) {
            // BGR交错 -> RGB平面, (v-127.5)/128
            val b = (bytes[i].toInt() and 0xFF).toFloat()
            val g = (bytes[i + 1].toInt() and 0xFF).toFloat()
            val r = (bytes[i + 2].toInt() and 0xFF).toFloat()
            out[p] = (r - inputMean) / inputStd
            out[n + p] = (g - inputMean) / inputStd
            out[2 * n + p] = (b - inputMean) / inputStd
            i += 3
        }
        return out
    }

    private fun toFloats(t: OnnxTensor): FloatArray {
        val fb = t.floatBuffer
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    private fun nms(faces: List<Face>): List<Face> {
        val sorted = ArrayList(faces.sortedByDescending { it.score })
        val keep = ArrayList<Face>()
        while (sorted.isNotEmpty()) {
            val cur = sorted.removeAt(0)
            keep.add(cur)
            val it = sorted.iterator()
            while (it.hasNext()) {
                if (iou(cur.bbox, it.next().bbox) > nmsThresh) it.remove()
            }
        }
        return keep
    }

    private fun iou(a: FloatArray, b: FloatArray): Float {
        val x1 = max(a[0], b[0])
        val y1 = max(a[1], b[1])
        val x2 = min(a[2], b[2])
        val y2 = min(a[3], b[3])
        val inter = max(0f, x2 - x1) * max(0f, y2 - y1)
        val areaA = (a[2] - a[0]) * (a[3] - a[1])
        val areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter)
    }

    fun close() = session.close()
}