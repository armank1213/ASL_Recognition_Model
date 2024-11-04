'use client';

import { useState, useRef } from 'react';
import { Camera } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

const ASLRecognition = () => {
  const [prediction, setPrediction] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [isStreamActive, setIsStreamActive] = useState(false);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreamActive(true);
        setError(null);
      }
    } catch (err) {
      setError(`Failed to access camera: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsStreamActive(false);
    }
  };

  const captureImage = async () => {
    if (!videoRef.current || !videoRef.current.videoWidth) return;

    const canvas = document.createElement('canvas');
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    ctx.drawImage(videoRef.current, 0, 0);

    // Convert the canvas to a blob
    canvas.toBlob(async (blob) => {
      if (!blob) {
        setError('Failed to capture image');
        return;
      }

      const formData = new FormData();
      formData.append('file', blob);

      try {
        console.log('Sending request to backend...');
        const response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          body: formData,
        });

        const contentType = response.headers.get('content-type');
        console.log('Response content type:', contentType);

        if (!response.ok) {
          const errorText = await response.text();
          console.error('Server error response:', errorText);
          throw new Error(`Server error: ${errorText}`);
        }

        const data = await response.json();
        console.log('Received response:', data);
        
        if (data.error) {
          throw new Error(data.error);
        }
        
        setPrediction(data.prediction);
        setError(null);
      } catch (err) {
        console.error('Full error object:', err);
        setError(`Prediction failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }, 'image/jpeg', 0.95);
  };

  return (
    <div className="max-w-md mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">ASL Letter Recognition</h1>
      
      <div className="space-y-4">
        <div className="relative aspect-video bg-gray-100 rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-full object-cover"
          />
        </div>

        <div className="flex gap-4">
          {!isStreamActive ? (
            <button
              onClick={startCamera}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
            >
              <Camera className="w-5 h-5" />
              Start Camera
            </button>
          ) : (
            <>
              <button
                onClick={captureImage}
                className="flex-1 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600"
              >
                Capture & Predict
              </button>
              <button
                onClick={stopCamera}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
              >
                Stop Camera
              </button>
            </>
          )}
        </div>

        {prediction && (
          <div className="text-center p-4 bg-green-100 rounded-lg">
            <h2 className="text-xl font-semibold">Predicted Letter:</h2>
            <p className="text-4xl font-bold text-green-700">{prediction}</p>
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </div>
    </div>
  );
};

export default ASLRecognition;