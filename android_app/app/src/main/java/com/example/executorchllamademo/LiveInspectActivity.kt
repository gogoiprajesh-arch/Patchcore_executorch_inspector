package com.example.executorchllamademo

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size as AndroidSize
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
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
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.min

// Keep equal to THRESHOLD in LlamaAdvisor.kt and DEFAULT_THRESHOLD in InspectionLog.kt
private const val LIVE_THRESHOLD = 17.0f
private const val LTAG = "PatchCoreLive"

class LiveInspectActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        LogStore.init(applicationContext)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    LiveScreen(this)
                }
            }
        }
    }
}

@Composable
fun LiveScreen(activity: ComponentActivity) {
    val ctx = LocalContext.current
    var hasPerm by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(ctx, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { hasPerm = it }
    LaunchedEffect(Unit) { if (!hasPerm) permLauncher.launch(Manifest.permission.CAMERA) }

    var heat by remember { mutableStateOf<Bitmap?>(null) }
    var record by remember { mutableStateOf<DefectRecord?>(null) }
    var label by remember { mutableStateOf("Starting...") }
    var stats by remember { mutableStateOf("") }
    var isDefect by remember { mutableStateOf(false) }
    val paused = remember { AtomicBoolean(false) }
    var pausedUi by remember { mutableStateOf(false) }

    val previewView = remember {
        PreviewView(ctx).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    val executor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { executor.shutdown() } }

    LaunchedEffect(hasPerm) {
        if (!hasPerm) return@LaunchedEffect
        val future = ProcessCameraProvider.getInstance(ctx)
        future.addListener({
            val provider = future.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            val analysis = ImageAnalysis.Builder()
                .setTargetResolution(AndroidSize(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var lastDone = 0L
            var periodEma = 0.0
            var infEma = 0.0
            var lastLoggedDecision = ""
            var lastLogMs = 0L

            analysis.setAnalyzer(executor) { proxy ->
                var started = false
                try {
                    // Detection and Llama take turns: skip frames while Llama writes
                    if (paused.get() || InferenceGate.llamaBusy.get()) return@setAnalyzer
                    InferenceGate.detecting.set(true)
                    started = true

                    val t0 = System.nanoTime()
                    PatchCoreRunner.ensureLoaded()

                    // Frame -> upright centre square (matches the square preview)
                    val raw = proxy.toBitmap()
                    val m = Matrix().apply { postRotate(proxy.imageInfo.rotationDegrees.toFloat()) }
                    val side = min(raw.width, raw.height)
                    val sq = Bitmap.createBitmap(
                        raw, (raw.width - side) / 2, (raw.height - side) / 2, side, side, m, true
                    )

                    val chw = preprocess(sq)
                    val t1 = System.nanoTime()
                    val res = PatchCoreRunner.run(chw)
                    val t2 = System.nanoTime()

                    val defect = res.score >= LIVE_THRESHOLD
                    val rec = if (defect) buildRecord(res.map, res.score, LIVE_THRESHOLD) else null
                    val hm = heatmapBitmap(res.map, LIVE_THRESHOLD)
                    val t3 = System.nanoTime()

                    val now = System.nanoTime()
                    if (lastDone != 0L) {
                        val period = (now - lastDone) / 1e6
                        periodEma = if (periodEma == 0.0) period else 0.8 * periodEma + 0.2 * period
                    }
                    lastDone = now
                    val inf = (t2 - t1) / 1e6
                    infEma = if (infEma == 0.0) inf else 0.8 * infEma + 0.2 * inf

                    // Structured log: on every OK/DEFECT change, and at most every 3 s otherwise
                    val nowMs = System.currentTimeMillis()
                    val decisionStr = if (defect) "DEFECT" else "OK"
                    if (decisionStr != lastLoggedDecision || nowMs - lastLogMs > 3000) {
                        lastLoggedDecision = decisionStr
                        lastLogMs = nowMs
                        LogStore.logInspection(
                            "live", decisionStr, res.score, LIVE_THRESHOLD, rec,
                            inf, (t3 - t0) / 1e6, if (defect) sq else null
                        )
                    }

                    isDefect = defect
                    label = if (defect) "DEFECT" else "OK"
                    heat = if (defect) hm else null
                    record = rec
                    stats = "score %.2f (thr %.2f)\n".format(res.score, LIVE_THRESHOLD) +
                            "FPS %.1f   frame latency %.0f ms\n".format(
                                if (periodEma > 0) 1000.0 / periodEma else 0.0, (t3 - t0) / 1e6
                            ) +
                            "capture+pre %.0f ms   inference %.0f ms (avg %.0f)".format(
                                (t1 - t0) / 1e6, inf, infEma
                            )
                } catch (e: Throwable) {
                    Log.e(LTAG, "frame failed", e)
                    label = "Error"
                    stats = e.message ?: e.toString()
                } finally {
                    if (started) InferenceGate.detecting.set(false)
                    proxy.close()
                }
            }

            provider.unbindAll()
            provider.bindToLifecycle(
                activity, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis
            )
        }, ContextCompat.getMainExecutor(ctx))
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(12.dp)
    ) {
        Box(modifier = Modifier.fillMaxWidth().aspectRatio(1f).clipToBounds()) {
            AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
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
                        size = Size(
                            (r.box[2] - r.box[0]) * size.width,
                            (r.box[3] - r.box[1]) * size.height
                        ),
                        style = Stroke(width = 4f)
                    )
                }
            }
        }
        Spacer(Modifier.height(40.dp))
        Text(
            label,
            color = if (isDefect) Color(0xFFD32F2F) else Color(0xFF2E7D32),
            style = MaterialTheme.typography.headlineMedium
        )
        Text(stats)
        record?.let {
            Text("severity: ${it.severity}   location: ${it.region}   area %.1f%%".format(it.areaPct))
        }
        Spacer(Modifier.height(8.dp))
        Row {
            Button(onClick = {
                pausedUi = !pausedUi
                paused.set(pausedUi)
            }) { Text(if (pausedUi) "Resume" else "Pause") }
            Spacer(Modifier.width(12.dp))
            Button(onClick = {
                activity.startActivity(Intent(activity, InspectionLogActivity::class.java))
            }) { Text("Log") }
        }

        AdvisorPanel(record)
    }
}