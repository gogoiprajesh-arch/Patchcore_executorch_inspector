package com.example.executorchllamademo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import org.json.JSONObject
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import kotlin.concurrent.thread
import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.tanh

private const val MODEL_PATH = "/data/local/tmp/patchcore/metal_plate_double_fp32.pte"
private const val THRESHOLD = 17.0f
private const val SIZE = 256
private const val TAG = "PatchCore"

class PatchCoreActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    PatchCoreScreen(this)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Preprocessing: antialiased bilinear (triangle filter) resize, same formula
// as PIL / torch antialias=True. Output is RGB, channel-first, floats 0..1.
// ---------------------------------------------------------------------------
private class Axis(val start: IntArray, val weights: Array<FloatArray>)

private fun buildAxis(inSize: Int, outSize: Int): Axis {
    val scale = inSize.toFloat() / outSize
    val support = max(scale, 1f)
    val start = IntArray(outSize)
    val weights = Array(outSize) { FloatArray(0) }
    for (o in 0 until outSize) {
        val center = (o + 0.5f) * scale
        val lo = max(0, floor(center - support + 0.5f).toInt())
        val hi = min(inSize, floor(center + support + 0.5f).toInt())
        val w = FloatArray(hi - lo)
        var sum = 0f
        for (i in lo until hi) {
            val v = max(0f, 1f - abs((i - center + 0.5f) / support))
            w[i - lo] = v
            sum += v
        }
        for (k in w.indices) w[k] /= sum
        start[o] = lo
        weights[o] = w
    }
    return Axis(start, weights)
}

fun preprocess(bmp: Bitmap): FloatArray {
    val sw = bmp.width
    val sh = bmp.height
    val px = IntArray(sw * sh)
    bmp.getPixels(px, 0, sw, 0, 0, sw, sh)

    val ax = buildAxis(sw, SIZE)
    val ay = buildAxis(sh, SIZE)

    val tmp = FloatArray(sh * SIZE * 3)
    for (y in 0 until sh) {
        for (ox in 0 until SIZE) {
            val w = ax.weights[ox]
            val s = ax.start[ox]
            var r = 0f; var g = 0f; var b = 0f
            for (k in w.indices) {
                val p = px[y * sw + s + k]
                val wk = w[k]
                r += ((p shr 16) and 0xFF) * wk
                g += ((p shr 8) and 0xFF) * wk
                b += (p and 0xFF) * wk
            }
            val t = (y * SIZE + ox) * 3
            tmp[t] = r; tmp[t + 1] = g; tmp[t + 2] = b
        }
    }

    val out = FloatArray(3 * SIZE * SIZE)
    val plane = SIZE * SIZE
    for (oy in 0 until SIZE) {
        val w = ay.weights[oy]
        val s = ay.start[oy]
        for (x in 0 until SIZE) {
            var r = 0f; var g = 0f; var b = 0f
            for (k in w.indices) {
                val t = ((s + k) * SIZE + x) * 3
                val wk = w[k]
                r += tmp[t] * wk
                g += tmp[t + 1] * wk
                b += tmp[t + 2] * wk
            }
            val o = oy * SIZE + x
            out[o] = r / 255f
            out[plane + o] = g / 255f
            out[2 * plane + o] = b / 255f
        }
    }
    return out
}

// ---------------------------------------------------------------------------
// Runner: loads the .pte once, runs forward, returns score + anomaly map.
// ---------------------------------------------------------------------------
class PatchCoreResult(val score: Float, val map: FloatArray)

object PatchCoreRunner {
    private var module: Module? = null

    fun ensureLoaded(): Long {
        if (module != null) return 0L
        val t0 = System.nanoTime()
        module = Module.load(MODEL_PATH)
        return (System.nanoTime() - t0) / 1_000_000
    }

    fun run(chw: FloatArray): PatchCoreResult {
        val input = Tensor.fromBlob(chw, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong()))
        val outputs = module!!.forward(EValue.from(input))
        var score = Float.NaN
        var map = FloatArray(0)
        for (e in outputs) {
            val data = e.toTensor().dataAsFloatArray
            if (data.size == 1) score = data[0] else map = data
        }
        return PatchCoreResult(score, map)
    }
    @Synchronized
    fun reset() {
        module = null
    }
}

// ---------------------------------------------------------------------------
// Postprocessing: heatmap overlay + defect record
// ---------------------------------------------------------------------------
private fun clamp01(v: Float) = max(0f, min(1f, v))

/** Transparent where the map is low, blue -> green -> yellow -> red as it
 *  approaches 2x threshold (same scaling as the C++ heatmap). */
fun heatmapBitmap(map: FloatArray, threshold: Float): Bitmap {
    val colors = IntArray(SIZE * SIZE)
    for (i in colors.indices) {
        val t = clamp01(map[i] / (2f * threshold))
        val r = clamp01(1.5f - abs(4f * t - 3f))
        val g = clamp01(1.5f - abs(4f * t - 2f))
        val b = clamp01(1.5f - abs(4f * t - 1f))
        val a = clamp01((t - 0.4f) / 0.2f) * 0.65f
        colors[i] = ((a * 255).toInt() shl 24) or
                ((r * 255).toInt() shl 16) or
                ((g * 255).toInt() shl 8) or
                (b * 255).toInt()
    }
    return Bitmap.createBitmap(colors, SIZE, SIZE, Bitmap.Config.ARGB_8888)
}

class DefectRecord(
    val type: String,
    val severity: String,
    val confidence: Float,
    val region: String,
    val box: FloatArray,   // x0, y0, x1, y1 normalised 0..1
    val areaPct: Float,
    val score: Float = 0f
){
    var logId: Long = 0L
}

/** Heuristic record from the score and the anomaly map (not a trained classifier). */
fun buildRecord(map: FloatArray, score: Float, threshold: Float): DefectRecord {
    val ratio = score / threshold
    val severity = when {
        ratio < 1.08f -> "low"
        ratio < 1.20f -> "medium"
        else -> "high"
    }
    val confidence = 0.5f + 0.5f * tanh(3f * (ratio - 1f))

    val mapMax = map.max()
    val cut = min(threshold, 0.8f * mapMax)
    var x0 = SIZE; var y0 = SIZE; var x1 = -1; var y1 = -1
    var count = 0
    for (y in 0 until SIZE) for (x in 0 until SIZE) {
        if (map[y * SIZE + x] >= cut) {
            count++
            if (x < x0) x0 = x
            if (x > x1) x1 = x
            if (y < y0) y0 = y
            if (y > y1) y1 = y
        }
    }
    if (count == 0) { x0 = 0; y0 = 0; x1 = SIZE - 1; y1 = SIZE - 1 }

    var peak = 0
    for (i in map.indices) if (map[i] > map[peak]) peak = i
    val cx = (peak % SIZE + 0.5f) / SIZE
    val cy = (peak / SIZE + 0.5f) / SIZE
    val col = if (cx < 0.33f) "left" else if (cx < 0.67f) "center" else "right"
    val row = if (cy < 0.33f) "top" else if (cy < 0.67f) "middle" else "bottom"
    val region = if (col == "center" && row == "middle") "center" else "$row-$col"

    return DefectRecord(
        type = "anomaly",
        severity = if (100f * count / (SIZE * SIZE) > 30f && severity != "high")
            (if (severity == "low") "medium" else "high") else severity,
        confidence = confidence,
        region = region,
        box = floatArrayOf(x0.toFloat() / SIZE, y0.toFloat() / SIZE,
            (x1 + 1).toFloat() / SIZE, (y1 + 1).toFloat() / SIZE),
        areaPct = 100f * count / (SIZE * SIZE),
        score = score
    )
}

private fun appendLog(activity: ComponentActivity, json: JSONObject) {
    try {
        val f = File(activity.getExternalFilesDir(null), "inspections.jsonl")
        f.appendText(json.toString() + "\n")
    } catch (e: Throwable) {
        Log.e(TAG, "log write failed", e)
    }
}

// ---------------------------------------------------------------------------
// Screen
// ---------------------------------------------------------------------------
@Composable
fun PatchCoreScreen(activity: ComponentActivity) {
    var picked by remember { mutableStateOf<Bitmap?>(null) }
    var heat by remember { mutableStateOf<Bitmap?>(null) }
    var record by remember { mutableStateOf<DefectRecord?>(null) }
    var status by remember { mutableStateOf("Pick an image to inspect") }
    var bench by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var lastChw by remember { mutableStateOf<FloatArray?>(null) }

    fun inspect(uri: Uri) {
        busy = true
        status = "Running..."
        bench = ""
        thread {
            try {
                val bmp = activity.contentResolver.openInputStream(uri).use {
                    BitmapFactory.decodeStream(it)
                } ?: throw IllegalStateException("Could not decode image")

                val loadMs = PatchCoreRunner.ensureLoaded()

                val t0 = System.nanoTime()
                val chw = preprocess(bmp)
                val t1 = System.nanoTime()
                val res = PatchCoreRunner.run(chw)
                val t2 = System.nanoTime()
                val isDefect = res.score >= THRESHOLD
                val rec = if (isDefect) buildRecord(res.map, res.score, THRESHOLD) else null
                val hm = heatmapBitmap(res.map, THRESHOLD)
                val t3 = System.nanoTime()

                val preMs = (t1 - t0) / 1e6
                val infMs = (t2 - t1) / 1e6
                val postMs = (t3 - t2) / 1e6

                val text = buildString {
                    appendLine("Result: ${if (isDefect) "DEFECT" else "OK"}")
                    appendLine("Score: %.3f  (threshold %.4f)".format(res.score, THRESHOLD))
                    appendLine("Image: ${bmp.width}x${bmp.height}")
                    appendLine("Preprocess: %.1f ms".format(preMs))
                    appendLine("Inference: %.1f ms".format(infMs))
                    appendLine("Heatmap+record: %.1f ms".format(postMs))
                    appendLine("Map min: %.4f  max: %.3f".format(res.map.min(), res.map.max()))
                    if (loadMs > 0) append("Model load: $loadMs ms (first run only)")
                }

                val json = JSONObject().apply {
                    put("time", System.currentTimeMillis())
                    put("image", uri.toString())
                    put("score", res.score.toDouble())
                    put("threshold", THRESHOLD.toDouble())
                    put("decision", if (isDefect) "DEFECT" else "OK")
                    put("preprocess_ms", preMs)
                    put("inference_ms", infMs)
                    if (rec != null) put("defect", JSONObject().apply {
                        put("type", rec.type)
                        put("severity", rec.severity)
                        put("confidence", rec.confidence.toDouble())
                        put("region", rec.region)
                        put("area_pct", rec.areaPct.toDouble())
                        put("box", org.json.JSONArray(rec.box.map { it.toDouble() }))
                    })
                }
                appendLog(activity, json)
                Log.i(TAG, text)

                picked = bmp
                heat = hm
                record = rec
                lastChw = chw
                status = text
            } catch (e: Throwable) {
                Log.e(TAG, "inspect failed", e)
                status = "Error: ${e.message}"
            } finally {
                busy = false
            }
        }
    }

    fun runBenchmark() {
        val chw = lastChw ?: return
        busy = true
        bench = "Benchmarking (3 warm-up + 20 runs)..."
        thread {
            try {
                repeat(3) { PatchCoreRunner.run(chw) }
                val times = DoubleArray(20)
                for (i in times.indices) {
                    val t = System.nanoTime()
                    PatchCoreRunner.run(chw)
                    times[i] = (System.nanoTime() - t) / 1e6
                }
                val avg = times.average()
                bench = "Benchmark (20 runs, inference only)\n" +
                        "mean %.1f ms  min %.1f  max %.1f\n".format(avg, times.min(), times.max()) +
                        "= %.2f FPS".format(1000.0 / avg)
                Log.i(TAG, bench)
            } catch (e: Throwable) {
                bench = "Benchmark error: ${e.message}"
            } finally {
                busy = false
            }
        }
    }

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri -> if (uri != null) inspect(uri) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState())
    ) {
        Row {
            Button(
                enabled = !busy,
                onClick = {
                    launcher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                }
            ) { Text("Pick image") }
            Spacer(Modifier.width(12.dp))
            Button(
                enabled = !busy && lastChw != null,
                onClick = { runBenchmark() }
            ) { Text("Benchmark") }
        }

        Spacer(Modifier.height(12.dp))
        picked?.let { bmp ->
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .aspectRatio(bmp.width.toFloat() / bmp.height)
            ) {
                Image(
                    bitmap = bmp.asImageBitmap(),
                    contentDescription = null,
                    modifier = Modifier.fillMaxSize(),
                    contentScale = ContentScale.FillBounds
                )
                heat?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = null,
                        modifier = Modifier.fillMaxSize(),
                        contentScale = ContentScale.FillBounds
                    )
                }
                record?.let { r ->
                    Canvas(modifier = Modifier.fillMaxSize()) {
                        drawRect(
                            color = Color.Red,
                            topLeft = Offset(r.box[0] * size.width, r.box[1] * size.height),
                            size = Size((r.box[2] - r.box[0]) * size.width,
                                (r.box[3] - r.box[1]) * size.height),
                            style = Stroke(width = 3f)
                        )
                    }
                }
            }
        }

        Spacer(Modifier.height(12.dp))
        Text(status)

        record?.let { r ->
            Spacer(Modifier.height(8.dp))
            Text(
                "Defect record\n" +
                        "type: ${r.type}\n" +
                        "severity: ${r.severity}\n" +
                        "location: ${r.region}  (area %.1f%%)\n".format(r.areaPct) +
                        "confidence: %.2f (from score margin)".format(r.confidence)
            )
        }
        androidx.compose.runtime.key(record) { AdvisorPanel(record) }

        if (bench.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Text(bench)
        }
    }
}
