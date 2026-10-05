package com.example.miniswap

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/** 对齐 + 贴回,逐行对应官方 align_face.py */
object FaceUtils {

    /** 标准脸5点模板(size=224)。对应 REFERENCE_FACIAL_POINTS: 原始点x+8, 再*size/112 */
    private fun referencePoints(size: Double): Array<DoubleArray> {
        val ref = arrayOf(
            doubleArrayOf(30.29459953, 51.69630051),
            doubleArrayOf(65.53179932, 51.50139999),
            doubleArrayOf(48.02519989, 71.73660278),
            doubleArrayOf(33.54930115, 92.3655014),
            doubleArrayOf(62.72990036, 92.20410156)
        )
        val scale = size / 112.0
        return Array(5) { i ->
            doubleArrayOf((ref[i][0] + 8.0) * scale, (ref[i][1]) * scale)
        }
    }

    /**
     * 用5点最小二乘解仿射矩阵(对应 align_with_five_points 的 lstsq)。
     * 解 src_pts_ * A = dst_pts_,其中每个点扩展为[x,y,1]。
     * 返回2x3仿射矩阵M(把src映射到dst)。
     */
    fun computeAffine(srcPoints: Array<DoubleArray>, size: Double = 224.0): Mat {
        val dst = referencePoints(size)
        // 构造 src_ (5x3): [x,y,1]
        val srcMat = Mat(5, 3, CvType.CV_64F)
        val dstMat = Mat(5, 3, CvType.CV_64F)
        for (i in 0 until 5) {
            srcMat.put(i, 0, srcPoints[i][0], srcPoints[i][1], 1.0)
            dstMat.put(i, 0, dst[i][0], dst[i][1], 1.0)
        }
        // 最小二乘解 A (3x3): srcMat * A = dstMat  ->  A = solve(srcMat, dstMat)
        val A = Mat()
        Core.solve(srcMat, dstMat, A, Core.DECOMP_SVD)
        // A是3x3,取出仿射2x3: M = [[A00,A10,A20],[A01,A11,A21]]
        val M = Mat(2, 3, CvType.CV_64F)
        M.put(0, 0, A.get(0, 0)[0], A.get(1, 0)[0], A.get(2, 0)[0])
        M.put(1, 0, A.get(0, 1)[0], A.get(1, 1)[0], A.get(2, 1)[0])
        srcMat.release(); dstMat.release(); A.release()
        return M
    }

    /** 逆矩阵(对应 back_matrix): 2x3补成3x3,求伪逆,取前两行 */
    fun backMatrix(M: Mat): Mat {
        val full = Mat.zeros(3, 3, CvType.CV_64F)
        M.copyTo(full.submat(0, 2, 0, 3))
        full.put(2, 2, 1.0)
        val inv = Mat()
        Core.invert(full, inv, Core.DECOMP_SVD)   // pinv
        val back = Mat(2, 3, CvType.CV_64F)
        inv.submat(0, 2, 0, 3).copyTo(back)
        full.release(); inv.release()
        return back
    }

    /** 对齐:warpAffine到size×size(对应 align_img) */
    fun alignImg(imgBgr: Mat, srcPoints: Array<DoubleArray>, size: Int = 224): Pair<Mat, Mat> {
        val M = computeAffine(srcPoints, size.toDouble())
        val aligned = Mat()
        Imgproc.warpAffine(imgBgr, aligned, M, Size(size.toDouble(), size.toDouble()), Imgproc.INTER_LINEAR)
        val back = backMatrix(M)
        M.release()
        return Pair(aligned, back)
    }

    /**
     * 贴回(对应 dealign)。
     * generated: 换脸后的224 BGR图(uint8 Mat)
     * origin: 原图 BGR(uint8 Mat)
     * back: 逆仿射2x3
     * maskF: IDN输出的mask,224x224单通道float(0~1)
     */
    fun dealign(generated: Mat, origin: Mat, back: Mat, maskF: Mat): Mat {
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(11.0, 11.0))

        // mask[mask>0.001]=1
        val mask = Mat()
        Imgproc.threshold(maskF, mask, 0.001, 1.0, Imgproc.THRESH_BINARY)
        // dilate 1次, erode 2次, blur 7x7
        Imgproc.dilate(mask, mask, kernel)
        Imgproc.erode(mask, mask, kernel, org.opencv.core.Point(-1.0, -1.0), 2)
        Imgproc.blur(mask, mask, Size(7.0, 7.0))

        // mask_1: 内框10像素置1, blur 11x11
        val mask1 = Mat.zeros(mask.size(), CvType.CV_32F)
        val h = mask.rows(); val w = mask.cols()
        mask1.submat(10, h - 10, 10, w - 10).setTo(org.opencv.core.Scalar(1.0))
        Imgproc.blur(mask1, mask1, Size(11.0, 11.0))
        Core.multiply(mask, mask1, mask)

        // warp generated和mask回原图尺寸
        val originSize = Size(origin.cols().toDouble(), origin.rows().toDouble())
        val genWarp = Mat()
        Imgproc.warpAffine(generated, genWarp, back, originSize)
        val maskWarp = Mat()
        Imgproc.warpAffine(mask, maskWarp, back, originSize)

        // 转float做融合: gen*mask + origin*(1-mask)
        val genF = Mat(); genWarp.convertTo(genF, CvType.CV_32FC3)
        val oriF = Mat(); origin.convertTo(oriF, CvType.CV_32FC3)
        // maskWarp单通道 -> 三通道
        val mask3 = Mat()
        val chans = ArrayList<Mat>()
        chans.add(maskWarp); chans.add(maskWarp); chans.add(maskWarp)
        Core.merge(chans, mask3)

        val out = Mat()
        val invMask = Mat()
        Core.subtract(Mat.ones(mask3.size(), mask3.type()).apply { setTo(org.opencv.core.Scalar(1.0,1.0,1.0)) }, mask3, invMask)
        val part1 = Mat(); Core.multiply(genF, mask3, part1)
        val part2 = Mat(); Core.multiply(oriF, invMask, part2)
        Core.add(part1, part2, out)

        val result = Mat()
        out.convertTo(result, CvType.CV_8UC3)

        // 清理
        kernel.release(); mask.release(); mask1.release(); genWarp.release(); maskWarp.release()
        genF.release(); oriF.release(); mask3.release(); invMask.release(); part1.release(); part2.release(); out.release()
        return result
    }
}

