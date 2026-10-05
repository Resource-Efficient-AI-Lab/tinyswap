package com.example.miniswap

import android.content.Context
import android.util.Log
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.abs

/** 阶段三:设备端数值验证 + 注入 */
object VerifyStage2 {
    private const val TAG = "VerifyStage2"
    init {
        System.loadLibrary("opencv_java4")
    }
    /** 把asset拷到私有目录,返回绝对路径(大模型走文件,不占Java堆) */
    private fun assetToFile(ctx: Context, name: String): String {
        val outFile = File(ctx.filesDir, name)
        if (!outFile.exists() || outFile.length() == 0L) {
            ctx.assets.open(name).use { input ->
                outFile.outputStream().use { output -> input.copyTo(output) }
            }
        }
        return outFile.absolutePath
    }
    /** OpenCV冒烟测试:加载native库 + 跑一个矩阵运算 */
    fun probeOpenCV() {
        try {
            System.loadLibrary("opencv_java4")
            val a = org.opencv.core.Mat(2, 2, org.opencv.core.CvType.CV_32F)
            a.put(0, 0, 1.0, 2.0, 3.0, 4.0)
            val b = org.opencv.core.Mat(2, 2, org.opencv.core.CvType.CV_32F)
            b.put(0, 0, 10.0, 20.0, 30.0, 40.0)
            val c = org.opencv.core.Mat()
            org.opencv.core.Core.add(a, b, c)
            val out = FloatArray(4)
            c.get(0, 0, out)
            Log.i(TAG, "OpenCV测试: [1,2,3,4]+[10,20,30,40]=${out.toList()} ${if (out[3] == 44f) "PASS" else "FAIL"}")
            a.release(); b.release(); c.release()
        } catch (e: Throwable) {
            Log.i(TAG, "OpenCV加载失败: ${e.message}")
        }
    }
    private fun readBytes(ctx: Context, name: String): ByteArray =
        ctx.assets.open(name).use { it.readBytes() }

    private fun readTensor(ctx: Context, key: String): FloatArray {
        val bytes = readBytes(ctx, "verify/$key.bin")
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    private fun shapeOf(manifest: JSONObject, key: String): LongArray {
        val arr = manifest.getJSONArray(key)
        return LongArray(arr.length()) { arr.getLong(it) }
    }

    private fun maxAbsDiff(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "size mismatch: ${a.size} vs ${b.size}" }
        var m = 0f
        for (i in a.indices) {
            val d = abs(a[i] - b[i])
            if (d > m) m = d
        }
        return m
    }

    private fun tensor(env: OrtEnvironment, data: FloatArray, shape: LongArray): OnnxTensor =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape)

    private fun toFloats(t: OnnxTensor): FloatArray {
        val fb = t.floatBuffer
        val out = FloatArray(fb.remaining())
        fb.get(out)
        return out
    }

    /** 第2步:三模型数值对齐(复现Colab的三个PASS) */
    fun run(ctx: Context) {
        val env = OrtEnvironment.getEnvironment()
        val manifest = JSONObject(String(readBytes(ctx, "verify/manifest.json")))
        val contract = JSONObject(String(readBytes(ctx, "deploy_contract.json")))

        run {
            val path = assetToFile(ctx, "arcface.onnx")
            val sess = env.createSession(path, OrtSession.SessionOptions())
            val input = tensor(env, readTensor(ctx, "arcface_input"), shapeOf(manifest, "arcface_input"))
            val inputName = sess.inputNames.first()
            sess.run(mapOf(inputName to input)).use { res ->
                val emb = toFloats(res[0] as OnnxTensor)
                val fmap = toFloats(res[1] as OnnxTensor)
                val dEmb = maxAbsDiff(emb, readTensor(ctx, "id_emb"))
                val dFmap = maxAbsDiff(fmap, readTensor(ctx, "id_feature_map"))
                val ok = if (dEmb < 1e-4 && dFmap < 1e-4) "PASS" else "FAIL"
                Log.i(TAG, "[1/3] ArcFace emb=%.3e fmap=%.3e %s".format(dEmb, dFmap, ok))
            }
            input.close()
            sess.close()
        }

        run {
            val path = assetToFile(ctx, "iim.onnx")
            val sess = env.createSession(path, OrtSession.SessionOptions())
            val emb = tensor(env, readTensor(ctx, "id_emb"), shapeOf(manifest, "id_emb"))
            val fmap = tensor(env, readTensor(ctx, "id_feature_map"), shapeOf(manifest, "id_feature_map"))
            val order = contract.getJSONArray("output_order")
            sess.run(mapOf("id_emb" to emb, "id_feature_map" to fmap)).use { res ->
                var worst = 0f
                var worstName = ""
                for (i in 0 until order.length()) {
                    val name = order.getString(i)
                    val out = toFloats(res[i] as OnnxTensor)
                    val d = maxAbsDiff(out, readTensor(ctx, "iim_raw_$name"))
                    if (d > worst) {
                        worst = d
                        worstName = name
                    }
                }
                val ok = if (worst < 1e-4) "PASS" else "FAIL"
                Log.i(TAG, "[2/3] IIM worst=%s diff=%.3e %s".format(worstName, worst, ok))
            }
            emb.close()
            fmap.close()
            sess.close()
        }

        run {
            val path = assetToFile(ctx, "idn_template.onnx")
            val sess = env.createSession(path, OrtSession.SessionOptions())
            val input = tensor(env, readTensor(ctx, "idn_input"), shapeOf(manifest, "idn_input"))
            sess.run(mapOf("target_img" to input)).use { res ->
                val img = toFloats(res[0] as OnnxTensor)
                val mask = toFloats(res[1] as OnnxTensor)
                val dImg = maxAbsDiff(img, readTensor(ctx, "idn_out_img"))
                val dMask = maxAbsDiff(mask, readTensor(ctx, "idn_out_mask"))
                val ok = if (dImg < 1e-3 && dMask < 1e-3) "PASS" else "FAIL"
                Log.i(TAG, "[3/3] IDN img=%.3e mask=%.3e %s".format(dImg, dMask, ok))
            }
            input.close()
            sess.close()
        }

        Log.i(TAG, "===== Stage3 Step2 verification done =====")
    }

    /** 3a:审计Java版addInitializer是否生效(真模型覆盖enc0_dw为0,看输出变化) */
    fun probeAddInitializer(ctx: Context) {
        val env = OrtEnvironment.getEnvironment()
        val path = assetToFile(ctx, "idn_template.onnx")
        val manifest = JSONObject(String(readBytes(ctx, "verify/manifest.json")))
        val input = tensor(env, readTensor(ctx, "idn_input"), shapeOf(manifest, "idn_input"))

        val baseSess = env.createSession(path, OrtSession.SessionOptions())
        val baseImg = baseSess.run(mapOf("target_img" to input)).use {
            toFloats(it[0] as OnnxTensor)
        }
        baseSess.close()

        val so = OrtSession.SessionOptions()
        val zeroWeight = FloatArray(3 * 1 * 4 * 4) { 0f }
        val zeroTensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(zeroWeight), longArrayOf(3, 1, 4, 4)
        )
        try {
            so.addInitializer("idn_enc0_dw_weight", zeroTensor)
            val ovSess = env.createSession(path, so)
            val ovImg = ovSess.run(mapOf("target_img" to input)).use {
                toFloats(it[0] as OnnxTensor)
            }
            ovSess.close()
            val diff = maxAbsDiff(baseImg, ovImg)
            val verdict = if (diff > 1e-3) "生效,可用addInitializer" else "无变化->哑巴,走bytes_patch"
            Log.i(TAG, "3a addInitializer: 覆盖enc0_dw为0后输出变化=%.3e %s".format(diff, verdict))
        } catch (e: Exception) {
            Log.i(TAG, "3a addInitializer抛异常: ${e.message} -> 走bytes_patch")
        }
        zeroTensor.close()
        input.close()
    }

    /** 3b:完整注入验证。用verify里的18组IIM输出经addInitializer覆盖,输出应与烤死版一致 */
    fun verifyInjection(ctx: Context) {
        val env = OrtEnvironment.getEnvironment()
        val path = assetToFile(ctx, "idn_template.onnx")
        val manifest = JSONObject(String(readBytes(ctx, "verify/manifest.json")))
        val contract = JSONObject(String(readBytes(ctx, "deploy_contract.json")))
        val order = contract.getJSONArray("output_order")
        val initMap = contract.getJSONObject("idn_initializer_map")

        val input = tensor(env, readTensor(ctx, "idn_input"), shapeOf(manifest, "idn_input"))

        val so = OrtSession.SessionOptions()
        val kept = ArrayList<OnnxTensor>()   // 生命周期必须长于session
        for (i in 0 until order.length()) {
            val canon = order.getString(i)            // 如 enc0_dw
            val initName = initMap.getString(canon)   // 如 idn_enc0_dw_weight
            val raw = readTensor(ctx, "iim_raw_$canon")
            val shape = shapeOf(manifest, "iim_raw_$canon")  // dw:[1,C,k,k]  pw:[Cout,Cin,1,1]
            val injectShape = if (canon.endsWith("_dw")) {
                longArrayOf(shape[1], 1, shape[2], shape[3])  // (1,C,k,k)->(C,1,k,k)
            } else {
                shape
            }
            val t = OnnxTensor.createTensor(env, FloatBuffer.wrap(raw), injectShape)
            kept.add(t)
            so.addInitializer(initName, t)
        }

        val sess = env.createSession(path, so)
        sess.run(mapOf("target_img" to input)).use { res ->
            val img = toFloats(res[0] as OnnxTensor)
            val mask = toFloats(res[1] as OnnxTensor)
            val dImg = maxAbsDiff(img, readTensor(ctx, "idn_out_img"))
            val dMask = maxAbsDiff(mask, readTensor(ctx, "idn_out_mask"))
            val ok = if (dImg < 1e-3 && dMask < 1e-3) "PASS" else "FAIL"
            Log.i(TAG, "3b 注入验证: img=%.3e mask=%.3e %s".format(dImg, dMask, ok))
        }
        sess.close()
        kept.forEach { it.close() }
        input.close()

    }
    /** 主干:source2换到target,输出成品图并与基准比对 */
    /** 主干:source2换到target,输出成品图并与基准比对(带分阶段计时) */
    fun runFullPipeline(activity: android.app.Activity) {
        val ctx: Context = activity
        val env = OrtEnvironment.getEnvironment()
        val contract = JSONObject(String(readBytes(ctx, "deploy_contract.json")))
        val order = contract.getJSONArray("output_order")
        val initMap = contract.getJSONObject("idn_initializer_map")
        val manifest = JSONObject(String(readBytes(ctx, "verify/manifest.json")))

        Log.i(TAG, "===== 单帧主干计时开始 =====")
        val tStart = System.nanoTime()

        // 读原图
        val srcBmp = android.graphics.BitmapFactory.decodeStream(ctx.assets.open("source2.jpg"))
        val tgtBmp = android.graphics.BitmapFactory.decodeStream(ctx.assets.open("target.jpg"))
        val srcMat = org.opencv.core.Mat()
        val tgtMat = org.opencv.core.Mat()
        org.opencv.android.Utils.bitmapToMat(srcBmp, srcMat)
        org.opencv.android.Utils.bitmapToMat(tgtBmp, tgtMat)
        org.opencv.imgproc.Imgproc.cvtColor(srcMat, srcMat, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)
        org.opencv.imgproc.Imgproc.cvtColor(tgtMat, tgtMat, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)

        // ===== 注入阶段(每源一次) =====
        Log.i(TAG, "--- 注入阶段(每个源人脸一次) ---")
        val det = timed("SCRFD建session") { ScrfdDetector(env, assetToFile(ctx, "scrfd.onnx")) }
        val src2Kps = timed("SCRFD检测源脸") { det.detectBestKps(srcMat) } ?: run {
            Log.i(TAG, "source2未检测到人脸"); det.close(); return
        }

        val alignedSrc = timed("对齐源脸") {
            val (a, _) = FaceUtils.alignImg(srcMat, src2Kps, 224)
            val r = org.opencv.core.Mat()
            org.opencv.imgproc.Imgproc.resize(a, r, org.opencv.core.Size(112.0, 112.0))
            r
        }
        val arcInput = matToArcInput(alignedSrc)
        val arcSess = timed("ArcFace建session") {
            env.createSession(assetToFile(ctx, "arcface.onnx"), OrtSession.SessionOptions())
        }
        val (emb, fmap) = timed("ArcFace推理") {
            val arcTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(arcInput), longArrayOf(1, 3, 112, 112))
            val p = arcSess.run(mapOf(arcSess.inputNames.first() to arcTensor)).use {
                Pair(toFloats(it[0] as OnnxTensor), toFloats(it[1] as OnnxTensor))
            }
            arcTensor.close(); p
        }
        arcSess.close()

        val w18 = timed("IIM推理") {
            val iimSess = env.createSession(assetToFile(ctx, "iim.onnx"), OrtSession.SessionOptions())
            val embT = OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), longArrayOf(1, 512))
            val fmapT = OnnxTensor.createTensor(env, FloatBuffer.wrap(fmap), longArrayOf(1, 512, 7, 7))
            val r = iimSess.run(mapOf("id_emb" to embT, "id_feature_map" to fmapT)).use { res ->
                Array(order.length()) { i -> toFloats(res[i] as OnnxTensor) }
            }
            embT.close(); fmapT.close(); iimSess.close(); r
        }

        val idnSess = timed("注入+建IDN session") {
            val so = OrtSession.SessionOptions()
            val kept = ArrayList<OnnxTensor>()
            for (i in 0 until order.length()) {
                val canon = order.getString(i)
                val shp = shapeOf(manifest, "iim_raw_$canon")
                val injShape = if (canon.endsWith("_dw"))
                    longArrayOf(shp[1], 1, shp[2], shp[3]) else shp
                val t = OnnxTensor.createTensor(env, FloatBuffer.wrap(w18[i]), injShape)
                kept.add(t)
                so.addInitializer(initMap.getString(canon), t)
            }
            env.createSession(assetToFile(ctx, "idn_template.onnx"), so)
        }
        det.close()

        // ===== 每帧阶段(视频每帧重复) =====
        Log.i(TAG, "--- 每帧阶段(视频每帧重复) ---")
        val det2 = ScrfdDetector(env, assetToFile(ctx, "scrfd.onnx"))
        val tgtKps = timed("SCRFD检测目标脸") { det2.detectBestKps(tgtMat) } ?: run {
            Log.i(TAG, "target未检测到人脸"); det2.close(); idnSess.close(); return
        }
        det2.close()

        val (alignedTgt, tgtBack) = timed("对齐目标脸") { FaceUtils.alignImg(tgtMat, tgtKps, 224) }
        val idnInput = matToIdnInput(alignedTgt)

        val (outImg, outMask) = timed("IDN推理") {
            val idnT = OnnxTensor.createTensor(env, FloatBuffer.wrap(idnInput), longArrayOf(1, 3, 224, 224))
            val p = idnSess.run(mapOf("target_img" to idnT)).use {
                Pair(toFloats(it[0] as OnnxTensor), toFloats(it[1] as OnnxTensor))
            }
            idnT.close(); p
        }

        val result = timed("贴回") {
            val genMat = idnOutputToBgrMat(outImg)
            val maskMat = org.opencv.core.Mat(224, 224, org.opencv.core.CvType.CV_32F)
            maskMat.put(0, 0, outMask)
            FaceUtils.dealign(genMat, tgtMat, tgtBack, maskMat)
        }
        idnSess.close()

        val totalMs = (System.nanoTime() - tStart) / 1_000_000.0
        Log.i(TAG, "===== 主干总耗时 %.1f ms =====".format(totalMs))

        // 存文件 + 显示
        val outBmp = android.graphics.Bitmap.createBitmap(result.cols(), result.rows(), android.graphics.Bitmap.Config.ARGB_8888)
        val resultRgba = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(result, resultRgba, org.opencv.imgproc.Imgproc.COLOR_BGR2RGBA)
        org.opencv.android.Utils.matToBitmap(resultRgba, outBmp)
        val outFile = java.io.File(ctx.filesDir, "swap_result.png")
        outFile.outputStream().use { outBmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it) }

        Log.i(TAG, "Static pipeline done, saved to ${outFile.absolutePath}")
    }

    // ---------- 辅助:读关键点bin (5,2) ----------
    private fun readKps(ctx: Context, key: String): Array<DoubleArray> {
        val bytes = readBytes(ctx, "verify/$key.bin")
        val fb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return Array(5) { doubleArrayOf(fb.get().toDouble(), fb.get().toDouble()) }
    }

    // ---------- 辅助:112 BGR Mat -> ArcFace输入(RGB,/255,ImageNet norm,NCHW) ----------
    private fun matToArcInput(bgr112: org.opencv.core.Mat): FloatArray {
        val rgb = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(bgr112, rgb, org.opencv.imgproc.Imgproc.COLOR_BGR2RGB)
        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)
        val out = FloatArray(3 * 112 * 112)
        val px = ByteArray(3)
        for (y in 0 until 112) for (x in 0 until 112) {
            rgb.get(y, x, px)
            for (c in 0 until 3) {
                val v = (px[c].toInt() and 0xFF) / 255f
                out[c * 112 * 112 + y * 112 + x] = (v - mean[c]) / std[c]
            }
        }
        rgb.release()
        return out
    }

    // ---------- 辅助:224 BGR Mat -> IDN输入(RGB,/255,NCHW) ----------
    private fun matToIdnInput(bgr224: org.opencv.core.Mat): FloatArray {
        val rgb = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(bgr224, rgb, org.opencv.imgproc.Imgproc.COLOR_BGR2RGB)
        val out = FloatArray(3 * 224 * 224)
        val px = ByteArray(3)
        for (y in 0 until 224) for (x in 0 until 224) {
            rgb.get(y, x, px)
            for (c in 0 until 3) {
                out[c * 224 * 224 + y * 224 + x] = (px[c].toInt() and 0xFF) / 255f
            }
        }
        rgb.release()
        return out
    }

    // ---------- 辅助:IDN输出(NCHW float RGB 0~1) -> 224 BGR uint8 Mat ----------
    private fun idnOutputToBgrMat(chw: FloatArray): org.opencv.core.Mat {
        val rgb = org.opencv.core.Mat(224, 224, org.opencv.core.CvType.CV_8UC3)
        val px = ByteArray(3)
        for (y in 0 until 224) for (x in 0 until 224) {
            for (c in 0 until 3) {
                val v = (chw[c * 224 * 224 + y * 224 + x] * 255f).coerceIn(0f, 255f)
                px[c] = v.toInt().toByte()
            }
            rgb.put(y, x, px)
        }
        val bgr = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.cvtColor(rgb, bgr, org.opencv.imgproc.Imgproc.COLOR_RGB2BGR)
        rgb.release()
        return bgr
    }
    /** Part1验证:端上SCRFD关键点 vs 预存关键点 */
    fun verifyScrfd(ctx: Context) {
        val env = OrtEnvironment.getEnvironment()
        val det = ScrfdDetector(env, assetToFile(ctx, "scrfd_500m.onnx"))

        val imgNames = arrayOf("source2", "target")
        val kpsNames = arrayOf("src2_kps", "tgt_kps")

        for (idx in imgNames.indices) {
            val imgName = imgNames[idx]
            val kpsName = kpsNames[idx]

            val bmp = android.graphics.BitmapFactory.decodeStream(ctx.assets.open("$imgName.jpg"))
            val mat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(bmp, mat)
            org.opencv.imgproc.Imgproc.cvtColor(mat, mat, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)

            val kps = det.detectBestKps(mat)
            if (kps == null) {
                Log.i(TAG, "$imgName: 未检测到人脸 FAIL")
                continue
            }

            val ref = readKps(ctx, kpsName)
            var maxDiff = 0.0
            for (k in 0 until 5) {
                val dx = kotlin.math.abs(kps[k][0] - ref[k][0])
                val dy = kotlin.math.abs(kps[k][1] - ref[k][1])
                if (dx > maxDiff) maxDiff = dx
                if (dy > maxDiff) maxDiff = dy
            }
            val ok = if (maxDiff < 2.0) "PASS" else "FAIL"
            Log.i(TAG, "$imgName SCRFD关键点最大误差=%.2f像素 %s".format(maxDiff, ok))

            val detStr = StringBuilder()
            val refStr = StringBuilder()
            for (k in 0 until 5) {
                detStr.append("(%.0f,%.0f)".format(kps[k][0], kps[k][1]))
                refStr.append("(%.0f,%.0f)".format(ref[k][0], ref[k][1]))
            }
            Log.i(TAG, "  检测:$detStr")
            Log.i(TAG, "  预存:$refStr")
        }
        det.close()
    }
    /** 计时辅助:执行block并返回(结果, 毫秒) */
    private inline fun <T> timed(label: String, block: () -> T): T {
        val t0 = System.nanoTime()
        val r = block()
        val ms = (System.nanoTime() - t0) / 1_000_000.0
        Log.i(TAG, "  [耗时] %-18s %.1f ms".format(label, ms))
        return r
    }
    /** 视频换脸:assets里的test.mp4 -> 换脸 -> 输出到外部目录 */
    fun runVideoSwap(activity: android.app.Activity) {
        val ctx: Context = activity
        try {
            // 1. Copy video from assets to a readable path
            val inFile = File(ctx.filesDir, "test.mp4")
            if (!inFile.exists() || inFile.length() == 0L) {
                ctx.assets.open("test.mp4").use { input ->
                    inFile.outputStream().use { input.copyTo(it) }
                }
            }
            val outFile = File(ctx.getExternalFilesDir(null), "swapped.mp4")
            if (outFile.exists()) outFile.delete()
            Log.i(TAG, "Input:  ${inFile.absolutePath} (${inFile.length() / 1024} KB)")
            Log.i(TAG, "Output: ${outFile.absolutePath}")

            // 2. Build engine, inject source face
            val t0 = System.nanoTime()
            val engine = FaceSwapEngine(ctx)
            val srcBmp = android.graphics.BitmapFactory.decodeStream(ctx.assets.open("source2.jpg"))
            val srcMat = org.opencv.core.Mat()
            org.opencv.android.Utils.bitmapToMat(srcBmp, srcMat)
            org.opencv.imgproc.Imgproc.cvtColor(srcMat, srcMat, org.opencv.imgproc.Imgproc.COLOR_RGBA2BGR)
            val injected = engine.injectSource(srcMat)
            val injectMs = (System.nanoTime() - t0) / 1_000_000.0
            Log.i(TAG, "Source injected: ${if (injected) "OK" else "FAILED (no face detected)"} %.0f ms".format(injectMs))
            if (!injected) {
                engine.close()
                activity.runOnUiThread {
                    activity.findViewById<android.widget.TextView>(R.id.statusText).text =
                        "No face detected in source image"
                }
                return
            }

            // Show source face + target video first frame
            val retriever = android.media.MediaMetadataRetriever()
            retriever.setDataSource(inFile.absolutePath)
            val firstFrame = retriever.getFrameAtTime(0)
            retriever.release()
            activity.runOnUiThread {
                activity.findViewById<android.widget.ImageView>(R.id.imgSource).setImageBitmap(srcBmp)
                activity.findViewById<android.widget.ImageView>(R.id.imgTarget).setImageBitmap(firstFrame)
                activity.findViewById<android.widget.TextView>(R.id.statusText).text = "Starting..."
            }

            // 3. Process the video
            val tVideo = System.nanoTime()
            var lastReport = 0L
            var totalFrames = 0
            VideoSwapper(engine).process(inFile.absolutePath, outFile.absolutePath) { frameIdx ->
                totalFrames = frameIdx
                val now = System.currentTimeMillis()
                if (now - lastReport > 1000) {
                    lastReport = now
                    Log.i(TAG, "Processed $frameIdx frames...")
                    activity.runOnUiThread {
                        activity.findViewById<android.widget.TextView>(R.id.statusText).text =
                            "Processing... frame $frameIdx"
                    }
                }
            }
            val videoMs = (System.nanoTime() - tVideo) / 1_000_000.0
            engine.close()

            Log.i(TAG, "===== Video swap complete =====")
            Log.i(TAG, "Took %.1f s, output %d KB".format(videoMs / 1000, outFile.length() / 1024))

            // 4. Switch to video playback
            val frames = if (totalFrames > 0) totalFrames else 1
            activity.runOnUiThread {
                activity.findViewById<android.widget.TextView>(R.id.statusText).text =
                    "Done — %d frames, %.0fs (%.0f ms/frame)".format(frames, videoMs / 1000, videoMs / frames)

                activity.findViewById<android.widget.LinearLayout>(R.id.processingView).visibility =
                    android.view.View.GONE
                val vv = activity.findViewById<android.widget.VideoView>(R.id.videoView)
                vv.visibility = android.view.View.VISIBLE

                vv.setVideoPath(outFile.absolutePath)
                val mc = android.widget.MediaController(activity)
                mc.setAnchorView(vv)
                vv.setMediaController(mc)
                vv.setOnPreparedListener { mp ->
                    mp.isLooping = true
                    vv.start()
                }
                vv.requestFocus()
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Video swap failed: ${e.message}", e)
            activity.runOnUiThread {
                activity.findViewById<android.widget.TextView>(R.id.statusText).text = "Failed: ${e.message}"
            }
        }
    }
}