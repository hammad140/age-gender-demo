import React, { useRef, useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as ort from 'onnxruntime-web';

export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const [session, setSession] = useState(null);
  const [prediction, setPrediction] = useState({ age: '', gender: '' });

  // 1) Load model & start webcam
  useEffect(() => {
    (async () => {
      try {
        const model = await ort.InferenceSession.create(
          'https://huggingface.co/hammad140/age-gender-browser-model/resolve/main/model.onnx'
        );
        console.log('✅ Model loaded');
        setSession(model);
      } catch (e) {
        console.error('❌ Model load error:', e);
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const video = videoRef.current;
        video.srcObject = stream;
        await new Promise(resolve => {
          video.onloadedmetadata = () => {
            video.play().catch(() => {});
            resolve();
          };
        });
        console.log('✅ Webcam started');
      } catch (e) {
        console.error('❌ Webcam error:', e);
      }
    })();
  }, []);

  // 2) Inference loop
  useEffect(() => {
    let cancelled = false;
    async function frameLoop() {
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (session && video.readyState >= 2) {
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        // preprocess NHWC -> NCHW
        const img = tf.browser.fromPixels(video)
          .resizeBilinear([224, 224])
          .toFloat()
          .div(255.0)
          .expandDims(0)         // [1,224,224,3]
          .transpose([0, 3, 1, 2]); // [1,3,224,224]
        const data = await img.data();

        try {
          const inputTensor = new ort.Tensor('float32', data, [1, 3, 224, 224]);
          const outputs = await session.run({ input_image: inputTensor });
          const age    = outputs.predicted_age.data[0].toFixed(1);
          const logits = outputs.predicted_gender_logits.data;
          const gender = logits[0] > logits[1] ? 'Male' : 'Female';
          setPrediction({ age, gender });
        } catch (e) {
          console.error('❌ Inference error:', e);
        }
      }
      if (!cancelled) requestAnimationFrame(frameLoop);
    }
    frameLoop();
    return () => { cancelled = true; };
  }, [session]);

  return (
    <div style={{ textAlign: 'center' }}>
      <h1>Age & Gender Demo</h1>
      <div style={{ position: 'relative', width: 640, height: 480, margin: '0 auto' }}>
        <video
          ref={videoRef}
          width={640}
          height={480}
          muted
          playsInline
          style={{ position: 'absolute', top: 0, left: 0, zIndex: 1 }}
        />
        <canvas
          ref={canvasRef}
          width={640}
          height={480}
          style={{ position: 'absolute', top: 0, left: 0, zIndex: 2 }}
        />
        <div
          style={{
            position: 'absolute',
            top: 10,
            left: 10,
            background: 'rgba(0, 0, 0, 0.6)',
            color: '#fff',
            fontSize: '24px',
            padding: '5px 10px',
            borderRadius: '4px',
            zIndex: 3
          }}
        >
          Age: {prediction.age || '--'} | Gender: {prediction.gender || '--'}
        </div>
      </div>
    </div>
  );
}
