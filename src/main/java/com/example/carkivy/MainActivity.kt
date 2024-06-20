package com.example.carkivy

import android.os.Bundle
import androidx.activity.ComponentActivity
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import android.Manifest
import android.os.Handler
import android.os.Looper
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.app.ActivityCompat
import com.example.carkivy.databinding.ActivityMainBinding
import java.util.concurrent.Executors
import androidx.core.content.ContextCompat
import android.content.pm.PackageManager
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.tensorflow.lite.Interpreter
import android.media.MediaPlayer
import android.widget.TextView
import android.util.Log
import kotlin.math.cos
import kotlin.math.PI
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.sin
import kotlin.math.ln

class MainActivity : ComponentActivity() {
    private val RECORD_REQUEST_CODE = 101
    private val handler = Handler(Looper.getMainLooper())
    private val executor = Executors.newSingleThreadExecutor()
    private var recording = false

    private lateinit var binding: ActivityMainBinding
    private lateinit var tvConfidences: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        tvConfidences = findViewById(R.id.tvConfidences)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }

        requestPermission()

        binding.startButton.setOnClickListener {
            if (!recording) {
                recording = true
                startRecording()
            }
        }

        binding.stopButton.setOnClickListener {
            recording = false
        }
    }

    private fun requestPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.RECORD_AUDIO), RECORD_REQUEST_CODE)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == RECORD_REQUEST_CODE) {
            if ((grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED)) {
                // Permission granted, proceed with recording
            } else {
                // Permission denied, handle accordingly
            }
        }
    }

    private fun startRecording() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermission()
            return
        }

        val sampleRate = 44100
        val bufferSize = sampleRate * 4
        val audioData = ShortArray(bufferSize)

        try {
            val audioRecord = AudioRecord(MediaRecorder.AudioSource.MIC, sampleRate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufferSize)

            audioRecord.startRecording()

            executor.execute {
                while (recording) {
                    val readSize = audioRecord.read(audioData, 0, bufferSize)
                    if (readSize > 0) {
                        val features = preprocessAudio(audioData, sampleRate)
                        val result = predictAudio(features)
                        handler.post {
                            binding.resultTextView.text = result
                        }
                    }
                    Thread.sleep(4000)
                }
                audioRecord.stop()
                audioRecord.release()
            }
        } catch (e: SecurityException) {
            e.printStackTrace()
        }
    }

    private fun FloatArray.mean(): Float {
        return if (this.isNotEmpty()) this.sum() / this.size else 0f
    }

    private fun FloatArray.standardDeviation(): Float {
        val mean = this.mean()
        return sqrt(this.map { (it - mean).pow(2) }.sum() / this.size)
    }


    class FFT(val n: Int) {
        private val cosTable = FloatArray(n / 2)
        private val sinTable = FloatArray(n / 2)

        init {
            for (i in 0 until n / 2) {
                cosTable[i] = cos(2.0 * PI * i / n).toFloat()
                sinTable[i] = sin(2.0 * PI * i / n).toFloat()
            }
        }

        fun transform(real: FloatArray): FloatArray {
            val imag = FloatArray(n)
            var i = 0
            var j = 0
            while (i < n) {
                if (i < j) {
                    val tempReal = real[i]
                    real[i] = real[j]
                    real[j] = tempReal
                    val tempImag = imag[i]
                    imag[i] = imag[j]
                    imag[j] = tempImag
                }
                i++
                var bit = n / 2
                while (bit > 0 && j >= bit) {
                    j -= bit
                    bit /= 2
                }
                j += bit
            }

            var size = 2
            while (size <= n) {
                val halfSize = size / 2
                val tableStep = n / size
                i = 0
                while (i < n) {
                    var k = 0
                    for (j in i until i + halfSize) {
                        val l = j + halfSize
                        if (l < n) {
                            val tReal = real[l] * cosTable[k] + imag[l] * sinTable[k]
                            val tImag = imag[l] * cosTable[k] - real[l] * sinTable[k]
                            real[l] = real[j] - tReal
                            imag[l] = imag[j] - tImag
                            real[j] += tReal
                            imag[j] += tImag
                        }
                        k += tableStep
                    }
                    i += size
                }
                size *= 2
            }
            return real
        }
    }

    private fun logSpecgram(audio: FloatArray, sampleRate: Int, eps: Float = 1e-10f): Array<FloatArray> {
        val nperseg = 1764
        val noverlap = 441
        val window = FloatArray(nperseg) { 0.54f - 0.46f * cos(2.0 * PI * it / (nperseg - 1)).toFloat() }

        val step = nperseg - noverlap
        val segments = (audio.size - noverlap) / step
        val specgram = Array(segments) { FloatArray(nperseg / 2 + 1) }

        for (i in 0 until segments) {
            val start = i * step
            if (start + nperseg <= audio.size) {
                val segment = audio.copyOfRange(start, start + nperseg).mapIndexed { index, value -> value * window[index] }.toFloatArray()
                val fft = FFT(segment.size)
                val transformed = fft.transform(segment)
                for (j in 0 until nperseg / 2) {
                    specgram[i][j] = ln(transformed[j].pow(2) + eps).toFloat()
                }
            }
        }
        return specgram
    }

    private fun prepareData(samples: FloatArray, numOfSamples: Int = 176400): FloatArray {
        return if (samples.size >= numOfSamples) {
            samples.copyOfRange(0, numOfSamples)
        } else {
            samples + FloatArray(numOfSamples - samples.size) { 0f }
        }
    }

    private fun extractSpectrogramFeatures(data: FloatArray, sampleRate: Int = 44100): Array<FloatArray> {
        val specgram = logSpecgram(data, sampleRate)
        val features = Array(specgram.size) { FloatArray(specgram[0].size) }
        for (i in specgram.indices) {
            val mean = specgram[i].mean()
            val std = specgram[i].standardDeviation()
            features[i] = specgram[i].map { (it - mean) / std }.toFloatArray()
        }
        return features
    }

    private fun extractFeaturesFromSamples(samples: FloatArray, sampleRate: Int = 44100): Array<FloatArray> {
        val processedData = prepareData(samples)
        return extractSpectrogramFeatures(processedData, sampleRate)
    }

    private fun preprocessAudio(audioData: ShortArray, sampleRate: Int): FloatArray {
        val audioList = audioData.map { it.toFloat() }.toFloatArray()
        val features = extractFeaturesFromSamples(audioList, sampleRate)

        // Flatten and ensure the features array matches the model's expected input size
        val flattenedFeatures = features.flatMap { it.toList() }.toFloatArray()

        // You can add padding or truncation logic here if necessary
        return flattenedFeatures
    }

    private fun playAlertSound() {
        val mediaPlayer = MediaPlayer.create(this, R.raw.beep)
        mediaPlayer.setOnCompletionListener { mp ->
            mp.release()
        }
        mediaPlayer.start()

        // 1초 후에 소리를 중지
        Handler(Looper.getMainLooper()).postDelayed({
            if (mediaPlayer.isPlaying) {
                mediaPlayer.stop()
            }
            mediaPlayer.release()
        }, 1000)
    }

    private fun predictAudio(features: FloatArray): String {
        val interpreter = getInterpreter()
        try {
            val inputDetails = interpreter.getInputTensor(0)
            val outputDetails = interpreter.getOutputTensor(0)

            val inputShape = inputDetails.shape()
            val inputSize = inputShape.reduce { acc, i -> acc * i }
            Log.d("MainActivity", "Expected input size: $inputSize, Actual features size: ${features.size}")

            if (features.size != inputSize) {
                throw IllegalArgumentException("Features size (${features.size}) does not match the expected input size ($inputSize)")
            }

            val inputData = ByteBuffer.allocateDirect(4 * inputSize).order(ByteOrder.nativeOrder())
            val floatBuffer = inputData.asFloatBuffer()
            floatBuffer.put(features)

            Log.d("MainActivity", "Input ByteBuffer - capacity: ${inputData.capacity()}, limit: ${inputData.limit()}, position: ${inputData.position()}")
            Log.d("MainActivity", "FloatBuffer - limit: ${floatBuffer.limit()}, position: ${floatBuffer.position()}")

            // Rewind the inputData buffer before reading
            inputData.rewind()
            Log.d("MainActivity", "Input ByteBuffer after rewind - capacity: ${inputData.capacity()}, limit: ${inputData.limit()}, position: ${inputData.position()}")

            // Output buffer size should match the number of labels (8 in this case)
            val outputBuffer = ByteBuffer.allocateDirect(4 * 8).order(ByteOrder.nativeOrder())

            interpreter.run(inputData, outputBuffer)

            // Rewind the outputBuffer before reading
            outputBuffer.rewind()
            Log.d("MainActivity", "Output ByteBuffer after rewind - capacity: ${outputBuffer.capacity()}, limit: ${outputBuffer.limit()}, position: ${outputBuffer.position()}")

            val outputArray = FloatArray(8)  // Update size to 8 labels
            outputBuffer.asFloatBuffer().get(outputArray)
            Log.d("MainActivity", "Output Array - values: ${outputArray.joinToString(", ")}")

            val softmaxOutput = softmax(outputArray)
            val confidences = softmaxOutput.withIndex().joinToString("\n") { (index, confidence) ->
                "Label $index: $confidence"
            }

            // 모든 라벨의 confidence 출력
            runOnUiThread {
                tvConfidences.text = confidences
            }

            // 차량 소리 인식 여부 결정
            val predictionMessage = if (softmaxOutput[2] >= 0.25) {
                "차량 소리가 인식되었습니다."
            } else {
                "차량 소리가 인식되지 않았습니다."
            }

            // 경고음 재생 (차량 소리 인식된 경우)
            if (softmaxOutput[2] >= 0.25) {
                playAlertSound()
            }

            return predictionMessage
        } finally {
            interpreter.close()
        }
    }


    private fun getInterpreter(): Interpreter {
        val assetManager = assets
        val modelDescriptor = assetManager.openFd("model1.tflite")
        modelDescriptor.createInputStream().use { inputStream ->
            val modelBuffer = inputStream.channel.map(
                java.nio.channels.FileChannel.MapMode.READ_ONLY,
                modelDescriptor.startOffset,
                modelDescriptor.declaredLength
            )
            return Interpreter(modelBuffer)
        }
    }

    private fun softmax(logits: FloatArray): FloatArray {
        val expValues = logits.map { Math.exp(it.toDouble()).toFloat() }
        val sumExpValues = expValues.sum()
        return expValues.map { it / sumExpValues }.toFloatArray()
    }
}