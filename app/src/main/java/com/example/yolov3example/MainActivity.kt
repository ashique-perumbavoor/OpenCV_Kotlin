package com.example.yolov3example

import android.content.pm.PackageManager
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.core.content.ContextCompat
import com.mapzen.speakerbox.Speakerbox
import kotlinx.android.synthetic.main.activity_main.*
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch
import org.opencv.android.OpenCVLoader
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.time.LocalTime
import java.util.ArrayList
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    // ***IMPORTANT*** You have to follow some instruction in order for the application to work properly
    // ***IMPORTANT*** You can find it here https://drive.google.com/drive/folders/17lYLHJ8E_gzQFE6iltPVudfSvxBHKe6a?usp=sharing
    // Or contact me on +91 9633127111

    private val permissions = arrayOf(
            android.Manifest.permission.WRITE_EXTERNAL_STORAGE,
            android.Manifest.permission.READ_EXTERNAL_STORAGE
    )
    private val classes = arrayListOf(
            "person",
            "bicycle",
            "car",
            "motorbike",
            "aeroplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "sofa",
            "pottedplant",
            "bed",
            "diningtable",
            "toilet",
            "tvmonitor",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush"
    )
    private var configurationFile = "/yolov3-spp.cfg"
    private var weightsFile = "/yolov3-spp.weights"
    private var inputWidth = 416
    private var inputHeight = 416
    private lateinit var frame: Mat
    private lateinit var net: Net
    private var time = 0
    @RequiresApi(Build.VERSION_CODES.O)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        requestPermissions(permissions, 1)

        btDetect.setOnClickListener {
            if (hasNoPermission()) {
                requestPermissions(permissions, 1)
            } else {
                time = LocalTime.now().toSecondOfDay()
                Log.d("hello", time.toString())
                progressBar.visibility = View.VISIBLE
                GlobalScope.launch {
                    OpenCVLoader.initDebug()
                    configurationFile = Environment.getExternalStorageDirectory().path + configurationFile
                    weightsFile = Environment.getExternalStorageDirectory().path + weightsFile
                    net = Dnn.readNetFromDarknet(configurationFile, weightsFile)
                    net.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV)
                    net.setPreferableTarget(Dnn.DNN_TARGET_CPU)
                    frame = Imgcodecs.imread(Environment.getExternalStorageDirectory().path + "/test.jpg")
                    val blob = Dnn.blobFromImage(frame, 1 / 255.0, Size(inputWidth.toDouble(),
                        inputHeight.toDouble()
                    ), Scalar(0.0, 0.0, 0.0, 0.0), true, false)
                    net.setInput(blob)
                    val outs = ArrayList<Mat>()
                    net.forward(outs, getOutputsNames(net))
                    postProcess(frame, outs)
                    Imgcodecs.imwrite(Environment.getExternalStorageDirectory().path + "/testOutput.jpg", frame)
                    val finishedTime = LocalTime.now().toSecondOfDay()
                    Log.d("hello", (finishedTime - time).toString())
                }
            }
        }
    }

    private fun hasNoPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
                android.Manifest.permission.READ_EXTERNAL_STORAGE
        ) != PackageManager.PERMISSION_GRANTED ||
                ContextCompat.checkSelfPermission(
                    this,
                    android.Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
    }

    private fun getOutputsNames(net: Net?): List<String> {
        val names = ArrayList<String>()
        if (names.size == 0) {
            val outLayers = net!!.unconnectedOutLayers.toList()
            val layersNames = net.layerNames
            for (i in outLayers.indices) {
                val layer = layersNames[outLayers[i].toInt() - 1]
                names.add(layer)
            }
        }
        return names
    }

    private fun postProcess(frame: Mat?, outs: List<Mat>) {
        val classIds: MutableList<Int> = ArrayList()
        val confidences: MutableList<Float> = ArrayList()
        val boxes: MutableList<Rect> = ArrayList()
        val objconf: MutableList<Float> = ArrayList()
        for (i in outs.indices) {
            for (j in 0 until outs[i].rows()) {
                val scores = outs[i].row(j).colRange(5, outs[i].row(j).cols())
                val r = Core.minMaxLoc(scores)
                if (r.maxVal > 0.5f) {
                    val bb = outs[i].row(j).colRange(0, 5)
                    val data = FloatArray(1)
                    bb[0, 0, data]
                    val centerX = (data[0] * frame!!.cols()).toInt()
                    bb[0, 1, data]
                    val centerY = (data[0] * frame.rows()).toInt()
                    bb[0, 2, data]
                    val width = (data[0] * frame.cols()).toInt()
                    bb[0, 3, data]
                    val height = (data[0] * frame.rows()).toInt()
                    val left = centerX - width / 2
                    val top = centerY - height / 2
                    bb[0, 4, data]
                    objconf.add(data[0])
                    confidences.add(r.maxVal.toFloat())
                    classIds.add(r.maxLoc.x.toInt())
                    boxes.add(Rect(left, top, width, height))
                }
            }
        }
        val boxers = MatOfRect()
        boxers.fromList(boxes)
        val configs = MatOfFloat()
        configs.fromList(objconf)
        val indexes = MatOfInt()
        Dnn.NMSBoxes(boxers, configs, 0.5f, 0.4f, indexes)
        if (indexes.total() > 0) {
            val indices = indexes.toArray()
            for (i in indices.indices) {
                val idx = indices[i]
                val box = boxes[idx]
                drawBoxes(
                        classIds[idx], confidences[idx], box.x, box.y,
                        box.x + box.width, box.y + box.height, frame
                )
            }
        }
    }

    private fun drawBoxes(
        classId: Int,
        conf: Float,
        left: Int,
        top: Int,
        right: Int,
        bottom: Int,
        frame: Mat?
    ) {
        //Draw a rectangle displaying the bounding box
        var top = top
        Imgproc.rectangle(
            frame, Point(
                left.toDouble(),
                top.toDouble()
            ), Point(right.toDouble(), bottom.toDouble()), Scalar(255.0, 178.0, 50.0), 3
        )

        //Get the label for the class name and its confidence
        var label = String.format("%.2f", conf)
        if (classes.size > 0) {
            label = classes[classId] + ":" + label
            Log.d("hello", classes[classId])
        }

        //Display the label at the top of the bounding box
        val baseLine = IntArray(1)
        val labelSize = Imgproc.getTextSize(label, Core.FONT_HERSHEY_SIMPLEX, 0.5, 1, baseLine)
        top = top.coerceAtLeast(labelSize.height.toInt())
        Imgproc.rectangle(
            frame, Point(
                left.toDouble(),
                (top - (1.5 * labelSize.height).roundToInt()).toDouble()
            ),
            Point(
                (left + (1.5 * labelSize.width).roundToInt()).toDouble(),
                (top + baseLine[0]).toDouble()
            ), Scalar(255.0, 255.0, 255.0), Core.FILLED
        )
        Imgproc.putText(
            frame, label, Point(
                left.toDouble(),
                top.toDouble()
            ), Core.FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0.0, 0.0, 0.0), 1
        )
        val speakerBox = Speakerbox(application)
        speakerBox.play("Detected objects is $label percentage")
        runOnUiThread {
            Toast.makeText(this, "Output is saved in the file in the name testOutput.jpg", Toast.LENGTH_LONG).show()
            progressBar.visibility = View.GONE
        }
    }
}