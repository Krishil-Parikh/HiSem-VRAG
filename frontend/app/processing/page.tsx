'use client';

import React from "react"

import { useEffect, useState, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Play, CheckCircle, Zap, Database, Sparkles, AlertCircle, Loader2 } from 'lucide-react';
import { API_BASE } from '@/lib/api';

interface ProcessingStep {
  id: string;
  title: string;
  description: string;
  status: 'pending' | 'processing' | 'complete' | 'error';
  icon: React.ReactNode;
}

interface VideoStatus {
  video_id: string;
  status: string;
  current_phase: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export default function ProcessingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const videoId = searchParams.get('videoId');
  const allVideoIds = searchParams.get('all')?.split(',') || [];

  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [videoStatus, setVideoStatus] = useState<VideoStatus | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [overallProgress, setOverallProgress] = useState(0);

  const [steps, setSteps] = useState<ProcessingStep[]>([
    {
      id: 'upload',
      title: 'Video Uploaded',
      description: 'Video file received and queued for processing',
      status: 'complete',
      icon: <Zap className="w-6 h-6" />,
    },
    {
      id: 'audio',
      title: 'Extracting Audio',
      description: 'Converting video to WAV audio format',
      status: 'pending',
      icon: <Sparkles className="w-6 h-6" />,
    },
    {
      id: 'transcribe',
      title: 'Transcribing Audio',
      description: 'Using WhisperX for speech-to-text',
      status: 'pending',
      icon: <Database className="w-6 h-6" />,
    },
    {
      id: 'chunks',
      title: 'Creating Chunks',
      description: 'Segmenting transcript into semantic chunks',
      status: 'pending',
      icon: <Database className="w-6 h-6" />,
    },
    {
      id: 'topics',
      title: 'Topic Segmentation',
      description: 'Identifying topic boundaries',
      status: 'pending',
      icon: <Sparkles className="w-6 h-6" />,
    },
    {
      id: 'frames',
      title: 'Extracting Frames',
      description: 'Capturing key frames from video',
      status: 'pending',
      icon: <Database className="w-6 h-6" />,
    },
    {
      id: 'embeddings',
      title: 'Creating Embeddings',
      description: 'Generating visual and text embeddings',
      status: 'pending',
      icon: <Sparkles className="w-6 h-6" />,
    },
    {
      id: 'scenes',
      title: 'Scene Clustering',
      description: 'Grouping similar frames into scenes',
      status: 'pending',
      icon: <Database className="w-6 h-6" />,
    },
    {
      id: 'memory',
      title: 'Building Memory',
      description: 'Creating searchable memory store',
      status: 'pending',
      icon: <Sparkles className="w-6 h-6" />,
    },
  ]);

  const updateStepsFromPhase = useCallback((phase: string | null, status: string) => {
    if (status === 'completed') {
      setSteps(prev => prev.map(s => ({ ...s, status: 'complete' as const })));
      setOverallProgress(100);
      return;
    }

    if (status === 'failed') {
      setHasError(true);
      return;
    }

    if (!phase) return;

    const phaseLower = phase.toLowerCase();
    
    // Map phase messages to step IDs
    const phaseMapping: Record<string, string[]> = {
      'audio': ['audio'],
      'extract': ['audio', 'frames'],
      'transcrib': ['transcribe'],
      'whisper': ['transcribe'],
      'chunk': ['chunks'],
      'embed': ['embeddings'],
      'topic': ['topics'],
      'segment': ['topics'],
      'frame': ['frames'],
      'scene': ['scenes'],
      'cluster': ['scenes'],
      'memory': ['memory'],
      'build': ['memory'],
    };

    let currentStepIndex = 0;
    
    for (const [keyword, stepIds] of Object.entries(phaseMapping)) {
      if (phaseLower.includes(keyword)) {
        const stepIndex = steps.findIndex(s => stepIds.includes(s.id));
        if (stepIndex > currentStepIndex) {
          currentStepIndex = stepIndex;
        }
      }
    }

    setSteps(prev => prev.map((s, idx) => ({
      ...s,
      status: idx < currentStepIndex ? 'complete' as const :
              idx === currentStepIndex ? 'processing' as const : 'pending' as const
    })));

    setOverallProgress(Math.round(((currentStepIndex + 1) / steps.length) * 100));
  }, [steps.length]);

  const pollStatus = useCallback(async () => {
    if (!videoId) return;

    try {
      const response = await fetch(`${API_BASE}/api/video/status/${videoId}`);
      if (!response.ok) {
        throw new Error('Failed to fetch status');
      }

      const data: VideoStatus = await response.json();
      setVideoStatus(data);

      updateStepsFromPhase(data.current_phase, data.status);

      if (data.status === 'completed') {
        setIsComplete(true);
        setTimeout(() => {
          router.push(`/chat?videoId=${videoId}`);
        }, 2000);
      } else if (data.status === 'failed') {
        setHasError(true);
      }

      return data.status;
    } catch (error) {
      console.error('Error polling status:', error);
      return null;
    }
  }, [videoId, router, updateStepsFromPhase]);

  useEffect(() => {
    if (!videoId) {
      router.push('/dashboard');
      return;
    }

    // Initial poll
    pollStatus();

    // Set up polling interval
    const interval = setInterval(async () => {
      const status = await pollStatus();
      if (status === 'completed' || status === 'failed') {
        clearInterval(interval);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [videoId, router, pollStatus]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-card dark:from-background dark:via-background dark:to-card/50 flex flex-col">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-accent to-primary rounded-lg flex items-center justify-center">
            <Play className="w-5 h-5 text-primary-foreground" fill="currentColor" />
          </div>
          <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">VideoRAG</span>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 flex items-center justify-center px-4 sm:px-6 lg:px-8 py-12">
        <div className="w-full max-w-2xl">
          {/* Title */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-foreground mb-2">Processing Your Video</h1>
            <p className="text-foreground/60">
              {videoStatus?.current_phase || 'Starting pipeline...'}
            </p>
            {allVideoIds.length > 1 && (
              <p className="text-sm text-primary mt-2">
                Video {currentVideoIndex + 1} of {allVideoIds.length}
              </p>
            )}
          </div>

          {/* Overall Progress */}
          <div className="mb-12">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-semibold text-foreground">Overall Progress</span>
              <span className="text-sm font-semibold text-primary">{overallProgress}%</span>
            </div>
            <div className="w-full h-2 bg-background rounded-full overflow-hidden border border-border/50">
              <div
                className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-500"
                style={{ width: `${overallProgress}%` }}
              />
            </div>
          </div>

          {/* Processing Steps */}
          <div className="space-y-4">
            {steps.map((step) => (
              <div
                key={step.id}
                className={`p-4 rounded-xl border transition-all ${
                  step.status === 'complete'
                    ? 'border-primary/30 bg-primary/5'
                    : step.status === 'processing'
                      ? 'border-accent bg-accent/5'
                      : step.status === 'error'
                        ? 'border-red-500/30 bg-red-500/5'
                        : 'border-border/50 bg-card/50'
                }`}
              >
                <div className="flex items-start gap-4">
                  {/* Icon */}
                  <div className={`flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center ${
                    step.status === 'complete'
                      ? 'bg-primary/20 text-primary'
                      : step.status === 'processing'
                        ? 'bg-accent/20 text-accent'
                        : step.status === 'error'
                          ? 'bg-red-500/20 text-red-500'
                          : 'bg-muted text-muted-foreground'
                  }`}>
                    {step.status === 'complete' ? (
                      <CheckCircle className="w-6 h-6" />
                    ) : step.status === 'processing' ? (
                      <Loader2 className="w-6 h-6 animate-spin" />
                    ) : step.status === 'error' ? (
                      <AlertCircle className="w-6 h-6" />
                    ) : (
                      step.icon
                    )}
                  </div>

                  {/* Content */}
                  <div className="flex-1">
                    <h3 className={`font-semibold mb-1 ${
                      step.status === 'pending'
                        ? 'text-foreground/50'
                        : 'text-foreground'
                    }`}>
                      {step.title}
                    </h3>
                    <p className={`text-sm ${
                      step.status === 'pending'
                        ? 'text-foreground/40'
                        : 'text-foreground/60'
                    }`}>
                      {step.description}
                    </p>
                  </div>

                  {/* Status Badge */}
                  <div className="flex-shrink-0">
                    {step.status === 'complete' && (
                      <span className="text-xs font-semibold px-2 py-1 rounded-full bg-primary/20 text-primary">
                        Done
                      </span>
                    )}
                    {step.status === 'processing' && (
                      <div className="flex gap-1">
                        {[0, 1, 2].map((i) => (
                          <div
                            key={i}
                            className="w-1.5 h-1.5 rounded-full bg-accent animate-bounce"
                            style={{ animationDelay: `${i * 150}ms` }}
                          />
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Complete Message */}
          {isComplete && (
            <div className="mt-12 p-6 rounded-xl bg-gradient-to-br from-primary/10 to-accent/10 border border-primary/20 text-center">
              <CheckCircle className="w-12 h-12 text-primary mx-auto mb-3" />
              <h2 className="text-xl font-bold text-foreground mb-2">Ready to Chat!</h2>
              <p className="text-foreground/60">Your video has been processed and is ready for analysis. Redirecting you now...</p>
            </div>
          )}

          {/* Error Message */}
          {hasError && (
            <div className="mt-12 p-6 rounded-xl bg-red-500/10 border border-red-500/20 text-center">
              <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-3" />
              <h2 className="text-xl font-bold text-foreground mb-2">Processing Failed</h2>
              <p className="text-foreground/60 mb-4">{videoStatus?.error || 'An error occurred during processing.'}</p>
              <button
                onClick={() => router.push('/dashboard')}
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Back to Dashboard
              </button>
            </div>
          )}

          {/* Tips */}
          {!isComplete && !hasError && (
            <div className="mt-12 p-4 rounded-lg bg-card border border-border/50">
              <p className="text-xs text-foreground/60 text-center">
                Tip: Processing may take 5-15 minutes depending on video length. You can leave this page and check back later.
              </p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
