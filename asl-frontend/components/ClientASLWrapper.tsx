'use client';

import dynamic from 'next/dynamic';

const ASLRecognition = dynamic(
  () => import('@/components/ASL_Recognition').then(mod => mod.default),
  { ssr: false }
);

export default function ClientASLWrapper() {
  return <ASLRecognition />;
} 