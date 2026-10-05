package com.example.miniswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

/**
 * 换脸引擎:封装"注入一次 + 每帧换脸"。
 * 用法:
 *   val engine = FaceSwapEngine(ctx)
 *   engine.injectSource(sourceBgrMat)   // 选定源脸,做一次
 *   val result = engine.swapFrame(targetBgrMat)  // 每帧调用,检测不到脸返回原帧
 *   engine.close()
 */
class FaceSwapEngine(private val ctx: android.content.Context) {

    private val env = OrtEnvironment.getEnvironment()
    private val contract = JSONObject(String(ctx.assets.open("deploy_contract.json").use { it.readBytes() }))
    private val order = contract.getJSONArray("output_order")
    private val initMap = contract.getJSONObject("idn_initializer_map")
    private val manifest = JSONObject(String(ctx.assets.open("verify/manifest.json").use { it.readBytes() }))

    private val detector = ScrfdDetector(env, assetToFile("scrfd_500m.onnx"))
    private var idnSession: OrtSession? = null
    private val keptTensors = ArrayList<OnnxTensor>()
    // ---- 计时统计(纳秒累加) ----
    var frameCount = 0L
    var nsDetect = 0L
    var nsAlign = 0L
    var nsPreproc = 0L
    var nsIdn = 0L
    var nsPostproc = 0L
    var nsDealign = 0L

    fun reportTiming(tag: String) {
        if (frameCount == 0L) return
        val f = frameCount.toDouble()
        android.util.Log.i(tag, "===== 换脸各环节平均耗时(共${frameCount}帧) =====")
        android.util.Log.i(tag, "  SCRFD检测   %.1f ms".format(nsDetect / f / 1e6))
        android.util.Log.i(tag, "  对齐        %.1f ms".format(nsAlign / f / 1e6))
        android.util.Log.i(tag, "  IDN前处理   %.1f ms".format(nsPreproc / f / 1e6))
        android.util.Log.i(tag, "  IDN推理     %.1f ms".format(nsIdn / f / 1e6))
        android.util.Log.i(tag, "  IDN后处理   %.1f ms".format(nsPostproc / f / 1e6))
        android.util.Log.i(tag, "  贴回融合    %.1f ms".format(nsDealign / f / 1e6))
        val total = (nsDetect + nsAlign + nsPreproc + nsIdn + nsPostproc + nsDealign) / f / 1e6
        android.util.Log.i(tag, "  换脸小计    %.1f ms".format(total))
        detector.reportTiming(tag)
    }
    private fun assetToFile(name: String): String {
        val f = File(ctx.filesDir, name)
        if (!f.exists() || f.length() == 0L) {
            ctx.assets.open(name).use { input -> f.outputStream().use { input.copyTo(it) } }
        }
        return f.absolutePath
    }

    private fun toFloats(t: OnnxTensor): FloatArray {
        val fb = t.floatBuffer; val out = FloatArray(fb.remaining()); fb.get(out); return out
    }

    private fun shapeOf(key: String): LongArray {
        val arr = manifest.getJSONArray(key)
        return LongArray(arr.length()) { arr.getLong(it) }
    }

    /** 选定源脸,生成并注入18组权重,建IDN session(每个源人脸做一次) */
    fun injectSource(srcBgr: Mat): Boolean {
        val kps = detector.detectBestKps(srcBgr) ?: return false
        val (aligned, _) = FaceUtils.alignImg(srcBgr, kps, 224)
        val src112 = Mat()
        Imgproc.resize(aligned, src112, Size(112.0, 112.0))

        // ArcFace
        val arcInput = matToArcInput(src112)
        val arcSess = env.createSession(assetToFile("arcface.onnx"), OrtSession.SessionOptions())
        val arcT = OnnxTensor.createTensor(env, FloatBuffer.wrap(arcInput), longArrayOf(1, 3, 112, 112))
        val (emb, fmap) = arcSess.run(mapOf(arcSess.inputNames.first() to arcT)).use {
            Pair(toFloats(it[0] as OnnxTensor), toFloats(it[1] as OnnxTensor))
        }
        arcT.close(); arcSess.close()

        // IIM
        val iimSess = env.createSession(assetToFile("iim.onnx"), OrtSession.SessionOptions())
        val embT = OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), longArrayOf(1, 512))
        val fmapT = OnnxTensor.createTensor(env, FloatBuffer.wrap(fmap), longArrayOf(1, 512, 7, 7))
        val w18 = iimSess.run(mapOf("id_emb" to embT, "id_feature_map" to fmapT)).use { res ->
            Array(order.length()) { i -> toFloats(res[i] as OnnxTensor) }
        }
        embT.close(); fmapT.close(); iimSess.close()

        // 注入 + 建IDN session
        val so = OrtSession.SessionOptions()
        keptTensors.forEach { it.close() }
        keptTensors.clear()
        for (i in 0 until order.length()) {
            val canon = order.getString(i)
            val shp = shapeOf("iim_raw_$canon")
            val injShape = if (canon.endsWith("_dw")) longArrayOf(shp[1], 1, shp[2], shp[3]) else shp
            val t = OnnxTensor.createTensor(env, FloatBuffer.wrap(w18[i]), injShape)
            keptTensors.add(t)
            so.addInitializer(initMap.getString(canon), t)
        }
        idnSession?.close()
        idnSession = env.createSession(assetToFile("idn_template.onnx"), so)
        return true
    }

    /** 换一帧:检测目标脸->对齐->IDN->贴回。检测不到脸返回原帧的拷贝 */
    /** 换一帧:检测目标脸->对齐->IDN->贴回。检测不到脸返回原帧的拷贝 */
    fun swapFrame(tgtBgr: Mat): Mat {
        val idn = idnSession ?: return tgtBgr.clone()

        var t = System.nanoTime()
        val kps = detector.detectBestKps(tgtBgr)
        nsDetect += System.nanoTime() - t
        if (kps == null) { frameCount++; return tgtBgr.clone() }

        t = System.nanoTime()
        val (alignedTgt, tgtBack) = FaceUtils.alignImg(tgtBgr, kps, 224)
        nsAlign += System.nanoTime() - t

        t = System.nanoTime()
        val idnInput = matToIdnInput(alignedTgt)
        nsPreproc += System.nanoTime() - t

        t = System.nanoTime()
        val idnT = OnnxTensor.createTensor(env, FloatBuffer.wrap(idnInput), longArrayOf(1, 3, 224, 224))
        val (outImg, outMask) = idn.run(mapOf("target_img" to idnT)).use {
            Pair(toFloats(it[0] as OnnxTensor), toFloats(it[1] as OnnxTensor))
        }
        idnT.close()
        nsIdn += System.nanoTime() - t

        t = System.nanoTime()
        val genMat = idnOutputToBgrMat(outImg)
        val maskMat = Mat(224, 224, CvType.CV_32F)
        maskMat.put(0, 0, outMask)
        nsPostproc += System.nanoTime() - t

        t = System.nanoTime()
        val result = FaceUtils.dealign(genMat, tgtBgr, tgtBack, maskMat)
        nsDealign += System.nanoTime() - t

        genMat.release(); maskMat.release(); alignedTgt.release(); tgtBack.release()
        frameCount++
        return result
    }

    fun close() {
        detector.close()
        idnSession?.close()
        keptTensors.forEach { it.close() }
    }

    // ---- 前处理辅助 ----
    private fun matToArcInput(bgr112: Mat): FloatArray {
        val rgb = Mat(); Imgproc.cvtColor(bgr112, rgb, Imgproc.COLOR_BGR2RGB)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val out = FloatArray(3 * 112 * 112); val px = ByteArray(3)
        for (y in 0 until 112) for (x in 0 until 112) {
            rgb.get(y, x, px)
            for (c in 0 until 3) {
                val v = (px[c].toInt() and 0xFF) / 255f
                out[c * 112 * 112 + y * 112 + x] = (v - mean[c]) / std[c]
            }
        }
        rgb.release(); return out
    }

    private fun matToIdnInput(bgr224: Mat): FloatArray {
        val rgb = Mat()
        Imgproc.cvtColor(bgr224, rgb, Imgproc.COLOR_BGR2RGB)
        val n = 224 * 224
        val bytes = ByteArray(n * 3)
        rgb.get(0, 0, bytes)
        rgb.release()
        val out = FloatArray(3 * n)
        var i = 0
        for (p in 0 until n) {
            out[p] = (bytes[i].toInt() and 0xFF) / 255f
            out[n + p] = (bytes[i + 1].toInt() and 0xFF) / 255f
            out[2 * n + p] = (bytes[i + 2].toInt() and 0xFF) / 255f
            i += 3
        }
        return out
    }

    private fun idnOutputToBgrMat(chw: FloatArray): Mat {
        val n = 224 * 224
        val bytes = ByteArray(n * 3)
        var i = 0
        for (p in 0 until n) {
            bytes[i] = (chw[p] * 255f).coerceIn(0f, 255f).toInt().toByte()
            bytes[i + 1] = (chw[n + p] * 255f).coerceIn(0f, 255f).toInt().toByte()
            bytes[i + 2] = (chw[2 * n + p] * 255f).coerceIn(0f, 255f).toInt().toByte()
            i += 3
        }
        val rgb = Mat(224, 224, CvType.CV_8UC3)
        rgb.put(0, 0, bytes)
        val bgr = Mat()
        Imgproc.cvtColor(rgb, bgr, Imgproc.COLOR_RGB2BGR)
        rgb.release()
        return bgr
    }
}