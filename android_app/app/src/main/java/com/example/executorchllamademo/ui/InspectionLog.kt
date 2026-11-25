package com.example.executorchllamademo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

// ---------------------------------------------------------------------------
// One inspection = one entry. Stored as one JSON line in inspection_log.jsonl
// (adb pull /sdcard/Android/data/com.example.executorchllamademo/files/inspection_log.jsonl)
// ---------------------------------------------------------------------------
class LogEntry(
    val id: Long,                 // timestamp in ms, also used as the unique id
    val source: String,
    val decision: String,         // "OK" or "DEFECT"
    val score: Float,
    val threshold: Float,
    val record: DefectRecord?,
    val inferenceMs: Double,
    val totalMs: Double,
    val thumb: String?,           // file name inside files/captures/
    initialAdvice: String? = null,
    initialAdviceStats: String? = null
) {
    var advice by mutableStateOf(initialAdvice)
    var adviceStats by mutableStateOf(initialAdviceStats)
}

object LogStore {
    private const val FILE_NAME = "inspection_log.jsonl"
    private const val DEFAULT_THRESHOLD = 17.0f

    val entries = mutableStateListOf<LogEntry>()   // newest first
    private var dir: File? = null
    private var lastId = 0L
    private var inited = false

    fun filePath(ctx: Context): String =
        File(ctx.getExternalFilesDir(null), FILE_NAME).path

    @Synchronized
    fun init(ctx: Context) {
        if (inited) return
        inited = true
        dir = ctx.getExternalFilesDir(null)
        val f = File(dir, FILE_NAME)
        if (!f.exists()) return
        val loaded = ArrayList<LogEntry>()
        f.forEachLine { line ->
            try {
                if (line.isNotBlank()) loaded.add(fromJson(JSONObject(line)))
            } catch (_: Exception) {
            }
        }
        loaded.sortByDescending { it.id }
        entries.addAll(loaded)
        lastId = loaded.maxOfOrNull { it.id } ?: 0L
    }

    /** Adds an entry, writes it to disk and returns its id. Safe to call from any thread. */
    @Synchronized
    fun logInspection(
        source: String,
        decision: String,
        score: Float,
        threshold: Float,
        rec: DefectRecord?,
        inferenceMs: Double,
        totalMs: Double,
        frame: Bitmap?
    ): Long {
        val id = maxOf(System.currentTimeMillis(), lastId + 1)
        lastId = id
        val thumb = if (frame != null) saveThumb(id, frame) else null
        val e = LogEntry(id, source, decision, score, threshold, rec, inferenceMs, totalMs, thumb)
        rec?.logId = id
        entries.add(0, e)
        try {
            File(dir, FILE_NAME).appendText(toJson(e).toString() + "\n")
        } catch (_: Exception) {
        }
        return id
    }

    /** Attaches Llama's advice to the entry of this record (creates the entry if it was not logged yet). */
    @Synchronized
    fun attachAdvice(rec: DefectRecord, advice: String, stats: String) {
        var e = entries.firstOrNull { it.id == rec.logId }
        if (e == null) {
            val id = logInspection("live", "DEFECT", rec.score, DEFAULT_THRESHOLD, rec, 0.0, 0.0, null)
            e = entries.first { it.id == id }
        }
        e.advice = advice
        e.adviceStats = stats
        rewrite()
    }

    @Synchronized
    fun clear() {
        entries.clear()
        try {
            File(dir, FILE_NAME).delete()
            File(dir, "captures").deleteRecursively()
        } catch (_: Exception) {
        }
    }

    private fun rewrite() {
        try {
            File(dir, FILE_NAME).writeText(
                entries.reversed().joinToString("") { toJson(it).toString() + "\n" }
            )
        } catch (_: Exception) {
        }
    }

    private fun saveThumb(id: Long, frame: Bitmap): String? = try {
        val cd = File(dir, "captures")
        cd.mkdirs()
        val small = Bitmap.createScaledBitmap(frame, 192, 192, true)
        File(cd, "$id.jpg").outputStream().use { small.compress(Bitmap.CompressFormat.JPEG, 80, it) }
        "$id.jpg"
    } catch (_: Exception) {
        null
    }

    private fun toJson(e: LogEntry): JSONObject = JSONObject().apply {
        put("id", e.id)
        put("source", e.source)
        put("decision", e.decision)
        put("score", e.score.toDouble())
        put("threshold", e.threshold.toDouble())
        put("inference_ms", e.inferenceMs)
        put("total_ms", e.totalMs)
        e.record?.let { r ->
            put("defect", JSONObject().apply {
                put("type", r.type)
                put("severity", r.severity)
                put("confidence", r.confidence.toDouble())
                put("region", r.region)
                put("area_pct", r.areaPct.toDouble())
                put("box", JSONArray(r.box.map { it.toDouble() }))
            })
        }
        e.thumb?.let { put("thumb", it) }
        e.advice?.let { put("advice", it) }
        e.adviceStats?.let { put("advice_stats", it) }
    }

    private fun fromJson(j: JSONObject): LogEntry {
        val id = j.getLong("id")
        val score = j.getDouble("score").toFloat()
        val rec = j.optJSONObject("defect")?.let { d ->
            val b = d.getJSONArray("box")
            DefectRecord(
                d.getString("type"),
                d.getString("severity"),
                d.getDouble("confidence").toFloat(),
                d.getString("region"),
                floatArrayOf(
                    b.getDouble(0).toFloat(), b.getDouble(1).toFloat(),
                    b.getDouble(2).toFloat(), b.getDouble(3).toFloat()
                ),
                d.getDouble("area_pct").toFloat(),
                score
            ).also { it.logId = id }
        }
        return LogEntry(
            id,
            j.optString("source", "live"),
            j.getString("decision"),
            score,
            j.getDouble("threshold").toFloat(),
            rec,
            j.optDouble("inference_ms", 0.0),
            j.optDouble("total_ms", 0.0),
            if (j.has("thumb")) j.getString("thumb") else null,
            if (j.has("advice")) j.getString("advice") else null,
            if (j.has("advice_stats")) j.getString("advice_stats") else null
        )
    }
}

// ---------------------------------------------------------------------------
// Viewer
// ---------------------------------------------------------------------------
class InspectionLogActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        LogStore.init(applicationContext)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    LogScreen(this)
                }
            }
        }
    }
}

@Composable
fun LogScreen(activity: ComponentActivity) {
    val entries = LogStore.entries
    var confirmClear by remember { mutableStateOf(false) }
    val defects = entries.count { it.decision == "DEFECT" }

    Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {
        Text("Inspection log", style = MaterialTheme.typography.headlineSmall)
        Text("${entries.size} entries, $defects defects")
        Text(
            LogStore.filePath(LocalContext.current),
            style = MaterialTheme.typography.bodySmall
        )
        Spacer(Modifier.height(8.dp))
        Row {
            Button(onClick = { activity.finish() }) { Text("Back") }
            Spacer(Modifier.width(12.dp))
            Button(onClick = {
                if (confirmClear) {
                    LogStore.clear()
                    confirmClear = false
                } else {
                    confirmClear = true
                }
            }) { Text(if (confirmClear) "Tap again to clear" else "Clear log") }
        }
        Spacer(Modifier.height(8.dp))
        LazyColumn(modifier = Modifier.fillMaxSize()) {
            items(entries, key = { it.id }) { e -> LogRow(e) }
        }
    }
}

@Composable
fun LogRow(e: LogEntry) {
    val ctx = LocalContext.current
    val thumb = remember(e.id) {
        e.thumb?.let { name ->
            val f = File(File(ctx.getExternalFilesDir(null), "captures"), name)
            BitmapFactory.decodeFile(f.path)?.asImageBitmap()
        }
    }
    val fmt = remember { SimpleDateFormat("dd MMM HH:mm:ss", Locale.getDefault()) }
    val isDefect = e.decision == "DEFECT"

    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 8.dp)) {
        if (thumb != null) {
            Image(bitmap = thumb, contentDescription = null, modifier = Modifier.size(72.dp))
            Spacer(Modifier.width(10.dp))
        }
        Column {
            Text(
                fmt.format(Date(e.id)) + "   " + e.decision,
                color = if (isDefect) Color(0xFFD32F2F) else Color(0xFF2E7D32),
                fontWeight = FontWeight.Bold
            )
            Text("score %.2f (thr %.2f)   %.0f ms".format(e.score, e.threshold, e.totalMs))
            e.record?.let {
                Text(
                    "${it.severity} | ${it.region} | area %.1f%% | type ${it.type}".format(it.areaPct)
                )
            }
            e.advice?.let { Text("Advice: $it", style = MaterialTheme.typography.bodySmall) }
            e.adviceStats?.let { Text(it, style = MaterialTheme.typography.bodySmall) }
        }
    }
}
