package com.example.executorchllamademo

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.json.JSONObject
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/** Detection and Llama take turns: Llama waits for the frame in flight, detection skips frames while Llama writes. */
object InferenceGate {
    val llamaBusy = AtomicBoolean(false)   // Llama is generating
    val detecting = AtomicBoolean(false)   // a PatchCore frame is in flight
}

object LlamaAdvisor {
    private const val MODEL_PATH = "/data/local/tmp/llama/Llama-3.2-1B-Instruct-8da4w-gptq-2k.pte"
    private const val TOKENIZER_PATH = "/data/local/tmp/llama/tokenizer.json"
    private const val SEQ_LEN = 320          // prompt + answer, model context is 2048
    private const val TEMPERATURE = 0.2f
    private const val THRESHOLD = 17.0f      // keep equal to LIVE_THRESHOLD in LiveInspectActivity.kt

    private var module: LlmModule? = null

    @Synchronized
    private fun ensureLoaded(): Long {
        if (module != null) return 0L
        val t0 = System.currentTimeMillis()
        val m = LlmModule(ModelUtils.TEXT_MODEL, MODEL_PATH, TOKENIZER_PATH, TEMPERATURE)
        m.load()
        module = m
        return System.currentTimeMillis() - t0
    }

    private fun buildPrompt(r: DefectRecord): String {
        // The decisions are made here in code; the model only phrases them
        val action = when (r.severity) {
            "low" -> "The plate may pass, but note it for monitoring."
            "medium" -> "Remove the plate for manual inspection."
            else -> "Reject the plate and check the line."
        }
        val framing = if (r.areaPct > 60f)
            "Also tell the operator the affected area is very large, so the camera position and lighting should be checked.\n"
        else ""

        val system = "You are a quality-control assistant on a metal plate production line. " +
                "Reply in at most 3 short sentences, in plain language, without quotation marks. " +
                "Use only the facts given and do not invent a cause."

        val user = "A vision system flagged a metal plate.\n" +
                "Severity: ${r.severity}\n" +
                "Location: ${r.region} of the image\n" +
                "Affected area: %.0f%% of the image\n".format(r.areaPct) +
                "Anomaly score: %.1f (alert threshold %.1f)\n".format(r.score, THRESHOLD) +
                "The defect type was not identified.\n" +
                "Required action: $action\n" +
                framing +
                "Tell the operator the required action and briefly why, using the facts above."

        val sysPart = PromptFormat.getSystemPromptTemplate(ModelType.LLAMA_3)
            .replace(PromptFormat.SYSTEM_PLACEHOLDER, system)
        val userPart = PromptFormat.getUserPromptTemplate(ModelType.LLAMA_3)
            .replace(PromptFormat.USER_PLACEHOLDER, user)
        return sysPart + userPart
    }

    /** Blocks until generation finishes. Call from a background thread. Returns a stats line. */
    fun advise(record: DefectRecord, onToken: (String) -> Unit): String {
        InferenceGate.llamaBusy.set(true)
        val done = AtomicBoolean(false)
        try {
            // Let the PatchCore frame already running finish, so the two never overlap
            val waitStart = System.currentTimeMillis()
            while (InferenceGate.detecting.get() && System.currentTimeMillis() - waitStart < 5000) {
                Thread.sleep(20)
            }

            val loadMs = ensureLoaded()
            val m = module!!

            // Watchdog: a stuck generation must not freeze detection forever
            Thread {
                try {
                    Thread.sleep(45_000)
                    if (!done.get()) m.stop()
                } catch (_: Exception) {
                }
            }.apply { isDaemon = true }.start()

            var sawHeader = false
            var statsLine = ""
            val t0 = System.currentTimeMillis()

            m.generate(buildPrompt(record), SEQ_LEN, object : LlmCallback {
                override fun onResult(result: String) {
                    if (result == "<|eot_id|>") { m.stop(); return }
                    if (result == "<|start_header_id|>") sawHeader = true
                    if (result == "<|end_header_id|>") { sawHeader = false; return }
                    if (sawHeader) return
                    onToken(result)
                }

                override fun onStats(stats: String) {
                    try {
                        val j = JSONObject(stats)
                        val n = j.getInt("generated_tokens")
                        val end = j.getInt("inference_end_ms")
                        val pre = j.getInt("prompt_eval_end_ms")
                        statsLine = "%.1f tok/s".format(n.toFloat() / (end - pre) * 1000)
                    } catch (_: Exception) {
                    }
                }
            }, false)

            val total = (System.currentTimeMillis() - t0) / 1000.0
            return "[$statsLine, %.1f s total".format(total) +
                    (if (loadMs > 0) ", model load %.1f s".format(loadMs / 1000.0) else "") + "]"
        } finally {
            done.set(true)
//            PatchCoreRunner.reset()
            InferenceGate.llamaBusy.set(false)
        }
    }
}

/** "Advise" button + streamed answer. Enabled only when a defect record exists. */
@Composable
fun AdvisorPanel(record: DefectRecord?) {
    var text by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }

    Column {
        Spacer(Modifier.height(8.dp))
        Button(
            enabled = record != null && !busy,
            onClick = {
                record?.let { rec ->
                    busy = true
                    text = "Loading Llama..."
                    thread {
                        try {
                            val sb = StringBuilder()
                            val info = LlamaAdvisor.advise(rec) { tok ->
                                sb.append(tok)
                                text = sb.toString()
                            }
                            val advice = sb.toString().trim()
                            text = advice + "\n\n" + info
                            LogStore.attachAdvice(rec, advice, info)   // needs InspectionLog.kt
                        } catch (e: Throwable) {
                            text = "Llama error: ${e.message}"
                        } finally {
                            busy = false
                        }
                    }
                }
            }
        ) { Text(if (busy) "Generating..." else "Advise (Llama)") }
        if (text.isNotEmpty()) {
            Spacer(Modifier.height(6.dp))
            Text(text)
        }
    }
}