package com.example.aifer


import android.content.ActivityNotFoundException
import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.*
import com.example.aifer.ml.Effb0FerMeta
import org.tensorflow.lite.support.image.TensorImage
import java.io.IOException
import android.graphics.drawable.BitmapDrawable
import java.util.concurrent.Executors
import kotlin.math.roundToInt


class MainActivity : AppCompatActivity() {

    // Cache the model to prevent repeated disk loading during inference
    private val model by lazy { Effb0FerMeta.newInstance(this) }

    // Executor for background tasks
    private val executor = Executors.newSingleThreadExecutor()

    // Cache views and adapter to prevent repeated findViewById calls and allocations
    private val imageView: ImageView by lazy { findViewById(R.id.imageView) }
    private val listAdapter: ArrayAdapter<String> by lazy {
        ArrayAdapter(this, android.R.layout.simple_list_item_1, ArrayList())
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Bind the cached adapter to the ListView
        findViewById<ListView>(R.id.listView).adapter = listAdapter
        findViewById<Button>(R.id.btn_photo).setOnClickListener {
            //建立一個要進行影像獲取的 Intent 物件

            val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
            //用 try-catch 避免例外產生，若產生則顯示 Toast
            try {
                startActivityForResult(intent, 0) //發送 Intent

            } catch (e: ActivityNotFoundException) {
                Toast.makeText(
                    this,
                    "error", Toast.LENGTH_SHORT
                ).show()
            }
        }

        findViewById<Button>(R.id.btn_album).setOnClickListener {
            //建立一個要進行影像獲取的 Intent 物件
            val intent =
                Intent(Intent.ACTION_GET_CONTENT).setType("image/*")
            //用 try-catch 避免例外產生，若產生則顯示 Toast
            try {
                startActivityForResult(intent, 1) //發送 Intent
            } catch (e: ActivityNotFoundException) {
                Toast.makeText(
                    this,
                    "error", Toast.LENGTH_SHORT
                ).show()
            }
        }
    }

    // 接收結果
    override fun onActivityResult(requestCode: Int,
                                  resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        //識別返回對象及執行結果
        if (requestCode == 0 && resultCode == RESULT_OK) {
            val image = data?.extras?.get("data") ?: return //取得資料
            val bitmap = image as Bitmap //將資料轉換成 Bitmap
            imageView.setImageBitmap(bitmap) //使用 Bitmap 設定圖像
            imageView.rotation = 90f //使 ImageView 旋轉順時針90度
            recognizeImage(bitmap) //使用 Bitmap 進行辨識

        }
        if (requestCode == 1 && resultCode == RESULT_OK) {
            val uri = data!!.data
            imageView.setImageURI(uri)
            imageView.rotation = 0f
            val drawable = imageView.drawable as BitmapDrawable //從imageView取得資料，轉換成Bitmap
            val bitmap = drawable.bitmap
            recognizeImage(bitmap) //使用 Bitmap 進行辨識
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        // Queue the model close on the single thread executor to ensure it doesn't
        // happen concurrently with a running inference, avoiding native crashes.
        executor.execute {
            model.close()
        }
        // Shut down the background executor
        executor.shutdown()
    }

    // 辨識圖像
    private fun recognizeImage(bitmap: Bitmap) {
        // Run inference in a background thread to prevent blocking the UI
        executor.execute {
            try {
                // Creates inputs for reference.
                val tensorImage = TensorImage.fromBitmap(bitmap)

                // Runs model inference and gets result.
                val outputs = model.process(tensorImage)
                    .probabilityAsCategoryList.apply {
                        sortByDescending { it.score } // 排序，由高到低
                    }

                //取得辨識結果與可信度
                val result = ArrayList<String>(outputs.size) // Pre-allocate capacity
                for (output in outputs) {
                    val label = output.label
                    val score: Int = (output.score * 100).roundToInt()
                    result.add("表情是 $label 的可能性為 $score %")
                }

                //將結果顯示於 ListView
                runOnUiThread {
                    listAdapter.clear()
                    listAdapter.addAll(result)
                    listAdapter.notifyDataSetChanged()
                }
            } catch (e: IOException) {
                e.printStackTrace()
            }
        }
    }
}