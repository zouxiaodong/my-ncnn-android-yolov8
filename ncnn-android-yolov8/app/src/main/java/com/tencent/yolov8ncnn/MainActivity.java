// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.yolov8ncnn;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.os.Build;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;

import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.util.ArrayList;
import java.util.List;


public class MainActivity extends Activity implements SurfaceHolder.Callback, INativeCallback {
    public static final int REQUEST_CAMERA = 100;

    private Yolov8Ncnn yolov8ncnn = new Yolov8Ncnn();
    private int facing = 1;

    private Spinner spinnerModel;
    private Spinner spinnerCPUGPU;
    private int current_model = 0;
    private int current_cpugpu = 0;

    private SurfaceView cameraView;
    private Mat mat;

    private float zoomRatio = 1;

    /**
     * Called when the activity is first created.
     */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        OpenCVLoader.initDebug();
        mat = new Mat();
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        cameraView = (SurfaceView) findViewById(R.id.cameraview);

        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        Button buttonSwitchCamera = (Button) findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {

                int new_facing = 1 - facing;

                yolov8ncnn.closeCamera();

                yolov8ncnn.openCamera(new_facing);

                facing = new_facing;
            }
        });

        spinnerModel = (Spinner) findViewById(R.id.spinnerModel);
        spinnerModel.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_model) {
                    current_model = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        spinnerCPUGPU = (Spinner) findViewById(R.id.spinnerCPUGPU);
        spinnerCPUGPU.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> arg0, View arg1, int position, long id) {
                if (position != current_cpugpu) {
                    current_cpugpu = position;
                    reload();
                }
            }

            @Override
            public void onNothingSelected(AdapterView<?> arg0) {
            }
        });

        Button zoomIn = (Button) findViewById(R.id.zoomIn);
        zoomIn.setOnClickListener(view -> {
            if (zoomRatio <= 10.0f) {
                zoomRatio = zoomRatio + 0.5f;
                yolov8ncnn.zoom(zoomRatio);
            }
        });
        Button zoomOut = (Button) findViewById(R.id.zoomOut);
        zoomOut.setOnClickListener(view -> {
            if (zoomRatio > 1.5f) {
                zoomRatio = zoomRatio - 0.5f;
                yolov8ncnn.zoom(zoomRatio);
            }
        });

        reload();
    }

    private void reload() {
        boolean ret_init = yolov8ncnn.loadModel(getAssets(), current_model, current_cpugpu);
        if (!ret_init) {
            Log.e("MainActivity", "yolov8ncnn loadModel failed");
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        yolov8ncnn.setOutputWindow(holder.getSurface(), mat.getNativeObjAddr(), this);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    @Override
    public void onResume() {
        super.onResume();

        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }

        yolov8ncnn.openCamera(facing);
    }

    @Override
    public void onPause() {
        super.onPause();

        yolov8ncnn.closeCamera();
    }

    @Override
    public void onClassify(float[] possibles) {

    }

    @Override
    public void onSegmentation(ArrayList<float[]> segmentationOutput, ArrayList<float[]> detectOutput) {

    }


    @Override
    public void onDetect(ArrayList<float[]> output) {
        if (output.isEmpty()) {
            return;
        }
        List<YoloResult> results = new ArrayList<>();
        for (float[] it : output) {
            YoloResult yolo = new YoloResult();

            float[] array = new float[4];
            array[0] = dp2px(it[0]);
            array[1] = dp2px(it[1]);
            array[2] = dp2px(it[2]);
            array[3] = dp2px(it[3]);
            yolo.setPosition(array);

            yolo.setType((int) it[4]);

            //保留两位有效小数
            yolo.setProb(String.format("%.2f", it[5]) + "%");
            results.add(yolo);
        }
        Log.e("YOLO", output.toString());
        //binding.getDetectView().updateTargetPosition(results);

        if (mat.width() > 0 || mat.height() > 0) {
            Bitmap bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(mat, bitmap, true);
            saveBitmap(bitmap);
        } else {
            Log.d("YOLOV8", "width: " + mat.width() + ", height: " + mat.height());
        }
    }

    private void saveBitmap(Bitmap bitmap) {
        File fileDirectory = getApplicationContext().getFilesDir();
        String filePath = fileDirectory.getAbsolutePath() + "/image.jpg";

        try (FileOutputStream outputStream = new FileOutputStream(filePath)) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream);
        } catch (Exception e) {
            e.printStackTrace();
        }

    }


    public float getScreenDensity(Context context) {
        WindowManager windowManager = (WindowManager) context.getSystemService(Context.WINDOW_SERVICE);
        DisplayMetrics displayMetrics = new DisplayMetrics();
        Display display;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            display = context.getDisplay();
        } else {
            display = windowManager.getDefaultDisplay();
        }
        if (display == null) {
            return 1f;
        }
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            return context.getResources().getDisplayMetrics().density;
        } else {
            display.getMetrics(displayMetrics);
            return displayMetrics.density;
        }
    }

    public float px2dp(float px) {
        return px / getScreenDensity(this);
    }

    /**
     * dp转px
     */
    public float dp2px(float dp) {
        return dp * getScreenDensity(this);
    }
}
