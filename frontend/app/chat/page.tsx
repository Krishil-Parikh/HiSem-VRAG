'use client';

import React from "react"
import { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ChatMessage } from '@/components/chat-message';
import { VideoSidebar } from '@/components/video-sidebar';
import { Play, Send, Menu, X, Plus, LogOut, Loader2, Maximize2, Layers, ArrowRight, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';
import { API_BASE, API_ENDPOINTS, isAuthenticated, logout } from '@/lib/api';

interface SceneSimilarity {
  video_id: string;
  scene_id: number;
  score: number;
}

interface QueryMetrics {
  num_queries_formed: number;
  queries: string[];
  num_videos_retrieved: number;
  num_topics_retrieved: number;
  num_scenes_retrieved: number;
  num_frames_retrieved: number;
  scene_similarities: SceneSimilarity[];
}

interface DisplaySceneItem {
  video_id: string;
  type: 'single' | 'combined';
  scene_ids: number[];
  start_time: number;
  end_time: number;
  text: string;
  visual_description: string;
  frame_url: string;
  score: number;
  original_score?: number;
  popularity_score?: number;
}

interface Evidence {
  video_id: string;
  topic_id: number;
  scene_id: number;
  text: string;
  start_time: number;
  end_time: number;
  score: number;
  frame_url: string;
  visual_description: string;
  original_score?: number;
  popularity_score?: number;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  evidence?: Evidence[];
  metrics?: QueryMetrics;
  display_scenes?: DisplaySceneItem[];
}

interface Video {
  video_id: string;
  filename: string;
  status: string;
  has_memory: boolean;
  created_at: string;
  source: 'uploaded' | 'dataset';
}

interface VideoMemory {
  video_id: string;
  num_topics: number;
  topics: Array<{
    topic_id: number;
    text: string;
    start_time: number;
    end_time: number;
    duration: number;
    scenes: Array<{
      scene_id: number;
      start_time: number;
      end_time: number;
      frame_count: number;
      frame_urls?: string[];
    }>;
  }>;
}

export default function ChatPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const videoIdParam = searchParams.get('videoId');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);
  const [videoMemory, setVideoMemory] = useState<VideoMemory | null>(null);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [isLoadingMemory, setIsLoadingMemory] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isMobile, setIsMobile] = useState(false);
  const [expandedTopic, setExpandedTopic] = useState<number | null>(null);
  const [isGlobalMode, setIsGlobalMode] = useState(true);
  const [expandedRerank, setExpandedRerank] = useState<string | null>(null);

  // Load videos from backend
  const loadVideos = useCallback(async () => {
    setIsLoadingVideos(true);
    try {
      const response = await fetch(`${API_BASE}/api/video/list`);
      if (response.ok) {
        const data = await response.json();
        setVideos(data.videos || []);
        
        if (videoIdParam && data.videos.some((v: Video) => v.video_id === videoIdParam)) {
          setSelectedVideoIds([videoIdParam]);
          setIsGlobalMode(false);
        }
      }
    } catch (error) {
      console.error('Failed to load videos:', error);
    } finally {
      setIsLoadingVideos(false);
    }
  }, [videoIdParam]);

  // Load memory for the FIRST selected video (for context panel)
  const loadVideoMemory = useCallback(async (videoId: string) => {
    if (!videoId) return;
    setIsLoadingMemory(true);
    try {
      const response = await fetch(`${API_BASE}/api/video/topics/${videoId}`);
      if (response.ok) {
        const data = await response.json();
        setVideoMemory(data);
      }
    } catch (error) {
      console.error('Failed to load video memory:', error);
    } finally {
      setIsLoadingMemory(false);
    }
  }, []);

  useEffect(() => {
    if (selectedVideoIds.length === 1) {
      loadVideoMemory(selectedVideoIds[0]);
    } else {
      setVideoMemory(null);
    }
  }, [selectedVideoIds, loadVideoMemory]);

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1024);
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!isAuthenticated()) {
      router.push('/login');
      return;
    }
    loadVideos();
  }, [loadVideos, router]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const query = inputValue;
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: query,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch(API_ENDPOINTS.query, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          query, 
          top_k: 5,
          video_ids: isGlobalMode ? null : selectedVideoIds
        }),
      });
      
      if (!response.ok) throw new Error('Query failed');
      const data = await response.json();
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        evidence: data.evidence,
        metrics: data.metrics,
        display_scenes: data.display_scenes,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages((prev) => [...prev, {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: "Sorry, I encountered an error processing your request.",
        timestamp: new Date(),
      }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleSelectVideo = (id: string) => {
    setIsGlobalMode(false);
    setSelectedVideoIds(prev => {
      if (prev.includes(id)) {
        const next = prev.filter(vid => vid !== id);
        if (next.length === 0) setIsGlobalMode(true);
        return next;
      }
      return [...prev, id];
    });
  };

  const handleSelectGlobal = () => {
    setIsGlobalMode(true);
    setSelectedVideoIds([]);
    setVideoMemory(null);
  };

  return (
    <div className="h-screen bg-background flex flex-col overflow-hidden">
      {/* Header */}
      <header className="h-16 border-b border-border/50 bg-card/50 backdrop-blur-md flex-shrink-0 z-30">
        <div className="h-full px-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
             <Button
                onClick={() => setIsSidebarOpen(!isSidebarOpen)}
                variant="ghost"
                size="sm"
                className="p-1 h-8 w-8"
              >
                {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
              </Button>
            <div className="flex items-center gap-2 cursor-pointer" onClick={() => router.push('/')}>
              <div className="w-8 h-8 bg-gradient-to-br from-accent to-primary rounded-lg flex items-center justify-center">
                <Play className="w-5 h-5 text-primary-foreground" fill="currentColor" />
              </div>
              <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent hidden sm:inline">VideoRAG</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Button onClick={() => router.push('/dashboard')} variant="ghost" size="sm" className="gap-2">
              <Plus className="w-4 h-4" />
              <span className="hidden sm:inline">Upload</span>
            </Button>
            <Button onClick={logout} variant="outline" size="sm" className="gap-2">
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">Logout</span>
            </Button>
          </div>
        </div>
      </header>

      {/* 3-Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        
        {/* Left Sidebar: Videos */}
        <aside className={`${isSidebarOpen ? 'w-72 border-r' : 'w-0'} border-border/50 bg-card transition-all duration-300 overflow-hidden flex flex-col`}>
          <VideoSidebar
            videos={videos.map(v => ({
              id: v.video_id,
              name: v.filename,
              duration: 'N/A',
              size: '...',
              source: v.source,
              status: v.status
            }))}
            selectedVideoIds={selectedVideoIds}
            isGlobalMode={isGlobalMode}
            onSelectGlobal={handleSelectGlobal}
            onSelectVideo={handleSelectVideo}
          />
        </aside>

        {/* Center Column: Chat */}
        <main className="flex-1 flex flex-col min-w-0 bg-background relative">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center pt-20 text-center opacity-40">
                  <Layers className="w-16 h-16 mb-4" />
                  <h2 className="text-xl font-bold">
                    {isGlobalMode ? "Global Database Search" : `Search across ${selectedVideoIds.length} videos`}
                  </h2>
                  <p className="text-sm max-w-xs">Ask a question to start exploring the video content.</p>
                </div>
              ) : (
                messages.map((message) => (
                  <div key={message.id} className="space-y-4">
                    <ChatMessage
                      role={message.role}
                      content={message.content}
                      timestamp={message.timestamp}
                    />
                    {message.metrics && (
                      <div className="ml-11 mb-3 bg-card border border-border/50 shadow-sm rounded-xl p-4 text-xs">
                        <div className="font-bold flex items-center gap-2 mb-3 text-foreground/80">
                          <Layers className="w-4 h-4 text-primary" />
                          VRAG Pipeline Trace
                        </div>

                        {/* Step 1: Query Decomposition */}
                        <div className="mb-4">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="w-5 h-5 rounded-full bg-blue-500/20 text-blue-400 flex items-center justify-center text-[9px] font-bold shrink-0">1</span>
                            <span className="font-semibold text-blue-400 text-[10px] uppercase tracking-wider">Query Decomposition</span>
                            <span className="text-[9px] text-foreground/40 ml-auto">{message.metrics.num_queries_formed} sub-queries</span>
                          </div>
                          <div className="ml-7 space-y-1">
                            {message.metrics.queries.map((q, qi) => (
                              <div key={qi} className="flex items-start gap-2 text-[11px]">
                                <ArrowRight className="w-3 h-3 text-blue-400/60 shrink-0 mt-0.5" />
                                <span className={qi === 0 ? "text-foreground font-medium" : "text-foreground/60"}>{q}</span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Step 2: Retrieval Summary */}
                        <div className="mb-4">
                          <div className="flex items-center gap-2 mb-2">
                            <span className="w-5 h-5 rounded-full bg-emerald-500/20 text-emerald-400 flex items-center justify-center text-[9px] font-bold shrink-0">2</span>
                            <span className="font-semibold text-emerald-400 text-[10px] uppercase tracking-wider">Hierarchical Retrieval</span>
                          </div>
                          <div className="ml-7 grid grid-cols-4 gap-2">
                            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-2 text-center">
                              <div className="text-lg font-bold text-emerald-400">{message.metrics.num_videos_retrieved}</div>
                              <div className="text-[9px] text-foreground/40">Videos</div>
                            </div>
                            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-2 text-center">
                              <div className="text-lg font-bold text-emerald-400">{message.metrics.num_topics_retrieved}</div>
                              <div className="text-[9px] text-foreground/40">Topics</div>
                            </div>
                            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-2 text-center">
                              <div className="text-lg font-bold text-emerald-400">{message.metrics.num_scenes_retrieved}</div>
                              <div className="text-[9px] text-foreground/40">Scenes</div>
                            </div>
                            <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-lg p-2 text-center">
                              <div className="text-lg font-bold text-emerald-400">{message.metrics.num_frames_retrieved}</div>
                              <div className="text-[9px] text-foreground/40">VLM Frames</div>
                            </div>
                          </div>
                        </div>

                        {/* Step 3: Scene Similarity Scores */}
                        {message.metrics.scene_similarities && message.metrics.scene_similarities.length > 0 && (
                          <div className="mb-4">
                            <div className="flex items-center gap-2 mb-2">
                              <span className="w-5 h-5 rounded-full bg-violet-500/20 text-violet-400 flex items-center justify-center text-[9px] font-bold shrink-0">3</span>
                              <span className="font-semibold text-violet-400 text-[10px] uppercase tracking-wider">SigLIP Scene Scores</span>
                              <span className="text-[9px] text-foreground/30 ml-auto">cross-modal cosine similarity</span>
                            </div>
                            <div className="ml-7 space-y-1.5">
                              {[...message.metrics.scene_similarities]
                                .sort((a, b) => b.score - a.score)
                                .map((sim, si) => (
                                <div key={si} className="flex items-center gap-2">
                                  <span className="text-[9px] font-mono text-foreground/40 w-16 shrink-0 truncate">{sim.video_id.slice(0, 11)}</span>
                                  <span className="text-[9px] text-foreground/30 w-8 shrink-0">S{sim.scene_id}</span>
                                  <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                                    <div
                                      className="h-full rounded-full bg-gradient-to-r from-violet-500 to-purple-400 transition-all"
                                      style={{ width: `${Math.min(sim.score * 100, 100)}%` }}
                                    />
                                  </div>
                                  <span className={`text-[10px] font-mono font-bold w-12 text-right shrink-0 ${
                                    sim.score >= 0.35 ? 'text-emerald-400' : sim.score >= 0.2 ? 'text-amber-400' : 'text-rose-400'
                                  }`}>
                                    {(sim.score * 100).toFixed(1)}%
                                  </span>
                                </div>
                              ))}
                            </div>
                            <p className="ml-7 mt-2 text-[9px] text-foreground/30 italic">
                              SigLIP cross-modal scores of 20-45% indicate strong semantic match between query text and video frames.
                            </p>
                          </div>
                        )}

                        {/* Step 4: VLM + Answer Gen */}
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <span className="w-5 h-5 rounded-full bg-amber-500/20 text-amber-400 flex items-center justify-center text-[9px] font-bold shrink-0">4</span>
                            <span className="font-semibold text-amber-400 text-[10px] uppercase tracking-wider">VLM Frame Description → Answer Generation</span>
                          </div>
                          <p className="ml-7 text-[10px] text-foreground/40">
                            {message.metrics.num_frames_retrieved} keyframes described by <span className="font-mono text-amber-400/80">gemma-4-26b-a4b-it</span> VLM, then answer synthesized with full evidence context.
                          </p>
                        </div>
                      </div>
                    )}

                    {/* Popularity Re-Ranking Comparison */}
                    {message.display_scenes && message.display_scenes.length > 0 && message.display_scenes.some(s => s.original_score != null) && (
                      <div className="ml-11 mb-3">
                        <button
                          onClick={() => setExpandedRerank(expandedRerank === message.id ? null : message.id)}
                          className="w-full bg-gradient-to-r from-amber-500/10 via-orange-500/10 to-rose-500/10 border border-amber-500/30 rounded-xl p-3 text-left hover:border-amber-500/50 transition-all group"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <TrendingUp className="w-4 h-4 text-amber-500" />
                              <span className="text-xs font-bold text-foreground/80">Popularity Re-Ranking</span>
                              <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-amber-500/20 text-amber-400 font-mono">10% weight</span>
                            </div>
                            {expandedRerank === message.id
                              ? <ChevronUp className="w-4 h-4 text-foreground/40" />
                              : <ChevronDown className="w-4 h-4 text-foreground/40" />
                            }
                          </div>
                          {expandedRerank !== message.id && (
                            <p className="text-[10px] text-foreground/40 mt-1">Click to see before vs. after re-ranking order</p>
                          )}
                        </button>

                        {expandedRerank === message.id && (
                          <div className="mt-2 bg-card border border-border/50 rounded-xl p-4 animate-in fade-in slide-in-from-top-2 duration-200">
                            <div className="grid grid-cols-2 gap-4">
                              {/* BEFORE Column */}
                              <div>
                                <div className="flex items-center gap-1.5 mb-3">
                                  <div className="w-2 h-2 rounded-full bg-blue-500" />
                                  <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400">Before (Similarity Only)</span>
                                </div>
                                <div className="space-y-1.5">
                                  {[...message.display_scenes]
                                    .filter(s => s.original_score != null)
                                    .sort((a, b) => (b.original_score ?? 0) - (a.original_score ?? 0))
                                    .map((scene, idx) => (
                                      <div key={`before-${idx}`} className="flex items-center gap-2 px-2 py-1.5 rounded-lg bg-blue-500/5 border border-blue-500/10">
                                        <span className="text-[10px] font-bold text-blue-400 w-4 text-center">#{idx + 1}</span>
                                        <div className="flex-1 min-w-0">
                                          <p className="text-[10px] font-mono text-foreground/70 truncate">{scene.video_id}</p>
                                          <p className="text-[9px] text-foreground/40 truncate">{scene.text.slice(0, 50)}...</p>
                                        </div>
                                        <span className="text-[10px] font-mono text-blue-400 shrink-0">{((scene.original_score ?? 0) * 100).toFixed(1)}%</span>
                                      </div>
                                    ))}
                                </div>
                              </div>

                              {/* AFTER Column */}
                              <div>
                                <div className="flex items-center gap-1.5 mb-3">
                                  <div className="w-2 h-2 rounded-full bg-emerald-500" />
                                  <span className="text-[10px] font-bold uppercase tracking-wider text-emerald-400">After (+ Popularity)</span>
                                </div>
                                <div className="space-y-1.5">
                                  {[...message.display_scenes]
                                    .filter(s => s.original_score != null)
                                    .sort((a, b) => b.score - a.score)
                                    .map((scene, idx) => {
                                      const beforeOrder = [...message.display_scenes!]
                                        .filter(s => s.original_score != null)
                                        .sort((a, b) => (b.original_score ?? 0) - (a.original_score ?? 0))
                                        .findIndex(s => s.video_id === scene.video_id && s.start_time === scene.start_time);
                                      const moved = beforeOrder !== idx;
                                      const movedUp = beforeOrder > idx;
                                      return (
                                        <div key={`after-${idx}`} className={`flex items-center gap-2 px-2 py-1.5 rounded-lg border ${
                                          moved
                                            ? movedUp
                                              ? 'bg-emerald-500/10 border-emerald-500/30'
                                              : 'bg-rose-500/5 border-rose-500/20'
                                            : 'bg-emerald-500/5 border-emerald-500/10'
                                        }`}>
                                          <span className="text-[10px] font-bold text-emerald-400 w-4 text-center">#{idx + 1}</span>
                                          <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-1">
                                              <p className="text-[10px] font-mono text-foreground/70 truncate">{scene.video_id}</p>
                                              {moved && (
                                                <span className={`text-[8px] px-1 rounded font-bold ${
                                                  movedUp ? 'text-emerald-400 bg-emerald-500/20' : 'text-rose-400 bg-rose-500/20'
                                                }`}>
                                                  {movedUp ? `\u2191${beforeOrder - idx}` : `\u2193${idx - beforeOrder}`}
                                                </span>
                                              )}
                                            </div>
                                            <p className="text-[9px] text-foreground/40 truncate">{scene.text.slice(0, 50)}...</p>
                                          </div>
                                          <div className="text-right shrink-0">
                                            <span className="text-[10px] font-mono text-emerald-400">{(scene.score * 100).toFixed(1)}%</span>
                                            <br />
                                            <span className="text-[8px] font-mono text-amber-400/60">pop: {((scene.popularity_score ?? 0) * 100).toFixed(1)}%</span>
                                          </div>
                                        </div>
                                      );
                                    })}
                                </div>
                              </div>
                            </div>

                            {/* Legend */}
                            <div className="mt-3 pt-3 border-t border-border/30 flex items-center gap-4 text-[9px] text-foreground/40">
                              <span>final = 0.9 × similarity + 0.1 × popularity</span>
                              <span className="flex items-center gap-1"><span className="text-emerald-400">\u2191</span> moved up</span>
                              <span className="flex items-center gap-1"><span className="text-rose-400">\u2193</span> moved down</span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Evidence Scene Cards */}
                    {message.display_scenes && message.display_scenes.length > 0 && (
                      <div className="ml-11 grid grid-cols-1 md:grid-cols-2 gap-4 pb-4">
                        {message.display_scenes.map((scene, i) => (
                          <div key={i} className={`bg-card/50 rounded-xl border ${scene.type === 'combined' ? 'border-primary/50 ring-1 ring-primary/20 bg-primary/5' : 'border-border/50'} overflow-hidden flex flex-col`}>
                            <div className="relative aspect-video bg-black overflow-hidden group/vid shrink-0">
                               <video 
                                 src={`${API_BASE}/api/video/stream/${scene.video_id}#t=${scene.start_time},${scene.end_time}`} 
                                 controls 
                                 preload="metadata"
                                 className="w-full h-full object-contain"
                                 poster={scene.frame_url}
                               />
                               <div className="absolute top-2 left-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-md border border-white/10 text-[9px] text-white font-bold uppercase tracking-wider pointer-events-none transition-opacity duration-300 group-hover/vid:opacity-0">
                                 {scene.video_id} {scene.type === 'combined' && <span className="text-primary ml-1">COMBINED SCENES</span>}
                               </div>
                               {/* Score Badge */}
                               <div className={`absolute top-2 right-2 px-2 py-0.5 rounded-full backdrop-blur-md border text-[10px] font-bold font-mono pointer-events-none ${
                                 scene.score >= 0.35
                                   ? 'bg-emerald-500/30 border-emerald-400/40 text-emerald-300'
                                   : scene.score >= 0.2
                                     ? 'bg-amber-500/30 border-amber-400/40 text-amber-300'
                                     : 'bg-rose-500/30 border-rose-400/40 text-rose-300'
                               }`}>
                                 {(scene.score * 100).toFixed(1)}%
                               </div>
                               {/* Timestamp Badge */}
                               <div className="absolute bottom-2 right-2 px-2 py-0.5 rounded bg-black/70 backdrop-blur-md text-[9px] text-white/80 font-mono pointer-events-none">
                                 {formatTime(scene.start_time)} – {formatTime(scene.end_time)}
                               </div>
                            </div>
                            <div className="p-3 space-y-2 flex-1 flex flex-col justify-between bg-card">
                              <p className="text-[11px] text-foreground/90 font-medium leading-relaxed line-clamp-2">{scene.text}</p>
                              {scene.visual_description && (
                                <div className="flex items-start gap-1.5 bg-violet-500/5 border border-violet-500/15 rounded-lg p-2">
                                  <span className="text-[8px] font-bold text-violet-400 uppercase tracking-wider shrink-0 mt-0.5">VLM</span>
                                  <p className="text-[10px] text-foreground/50 italic line-clamp-2">{scene.visual_description}</p>
                                </div>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
              {isLoading && (
                <div className="flex gap-3 animate-pulse">
                  <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
                    <Loader2 className="w-4 h-4 animate-spin text-accent" />
                  </div>
                  <div className="bg-card px-4 py-2 rounded-lg border border-border/50">
                    <span className="text-sm">Thinking...</span>
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          </div>

          {/* Input Area: Always centered and accessible */}
          <div className="p-4 border-t border-border/50 bg-card/30 backdrop-blur-sm">
            <div className="max-w-3xl mx-auto">
              <form onSubmit={handleSendMessage} className="relative group">
                <Input
                  ref={inputRef}
                  type="text"
                  placeholder={
                    isGlobalMode 
                      ? "Search entire database..." 
                      : `Query ${selectedVideoIds.length} selected videos...`
                  }
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  disabled={isLoading}
                  className="w-full pr-12 h-12 bg-background/50 border-primary/20 focus:border-primary/50 transition-all rounded-xl"
                />
                <Button
                  type="submit"
                  disabled={isLoading || !inputValue.trim()}
                  className="absolute right-1.5 top-1.5 h-9 w-9 p-0 bg-primary hover:bg-primary/90 text-primary-foreground rounded-lg"
                >
                  {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                </Button>
              </form>
              <div className="mt-2 text-[10px] text-foreground/40 text-center">
                Press Enter to search
              </div>
            </div>
          </div>
        </main>

        {/* Right Sidebar: Context & Frames */}
        <aside className={`${selectedVideoIds.length === 1 && !isMobile ? 'w-96 border-l' : 'w-0'} border-border/50 bg-card transition-all duration-300 overflow-hidden flex flex-col`}>
          <div className="h-full flex flex-col min-w-[384px]">
            <div className="p-4 border-b border-border/50 flex items-center justify-between bg-muted/20">
              <h2 className="font-bold flex items-center gap-2 text-sm">
                <Maximize2 className="w-4 h-4 text-primary" />
                Context & Frames
              </h2>
              {videoMemory && (
                <span className="text-[10px] text-foreground/50 font-mono">
                  {videoMemory.num_topics} Topics
                </span>
              )}
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
              {!videoMemory ? (
                <div className="h-full flex flex-col items-center justify-center opacity-30 text-center px-6">
                  <Play className="w-12 h-12 mb-4" />
                  <p className="text-xs">
                    {selectedVideoIds.length > 1 
                      ? "Select a single video to view its rich topics and frames."
                      : "Rich context view is available when a single video is focused."}
                  </p>
                </div>
              ) : (
                videoMemory.topics.map((topic) => (
                  <div key={topic.topic_id} className="rounded-xl border border-border/50 bg-card/50 overflow-hidden">
                    <button 
                      onClick={() => setExpandedTopic(expandedTopic === topic.topic_id ? null : topic.topic_id)}
                      className="w-full text-left p-3 flex flex-col gap-1 hover:bg-muted/10 transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary text-[9px] font-bold">
                          TOPIC {topic.topic_id}
                        </span>
                        <span className="text-[10px] text-foreground/40 font-mono italic">
                          {formatTime(topic.start_time)}
                        </span>
                      </div>
                      <h3 className="text-xs font-semibold truncate text-foreground/80">{topic.text}</h3>
                    </button>
                    
                    {expandedTopic === topic.topic_id && (
                      <div className="px-3 pb-3 space-y-4 animate-in fade-in slide-in-from-top-1 duration-200">
                        <p className="text-[11px] text-foreground/60 leading-relaxed bg-muted/20 p-2 rounded">
                          {topic.text}
                        </p>
                        <div className="space-y-2">
                           <h4 className="text-[9px] font-bold uppercase text-foreground/30">Key Scenes</h4>
                           <div className="flex gap-3 overflow-x-auto pb-2 snap-x no-scrollbar">
                              {topic.scenes.map(scene => (
                                <div key={scene.scene_id} className="w-40 flex-shrink-0 snap-start bg-background rounded-lg border border-border/30 overflow-hidden group/scene">
                                  <div className="relative aspect-video">
                                    <img 
                                      src={scene.frame_urls?.[0] || ''} 
                                      className="w-full h-full object-cover grayscale group-hover/scene:grayscale-0 transition-all"
                                    />
                                    <div className="absolute bottom-1 right-1 px-1 rounded bg-black/60 text-[8px] text-white">
                                      {formatTime(scene.start_time)}
                                    </div>
                                  </div>
                                </div>
                              ))}
                           </div>
                        </div>
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>

      </div>
    </div>
  );
}
