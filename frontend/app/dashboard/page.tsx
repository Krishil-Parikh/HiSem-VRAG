'use client';

import React from "react"

import { useState, useRef, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Play, Upload, X, FileVideo, Archive, Plus, RefreshCw } from 'lucide-react';
import { API_BASE, isAuthenticated, getCurrentUser, logout } from '@/lib/api';

interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: 'video' | 'zip';
  file: File;
  progress: number;
  status: 'pending' | 'uploading' | 'uploaded' | 'error';
  videoId?: string;
}

interface ProcessedVideo {
  video_id: string;
  filename: string;
  status: string;
  has_memory: boolean;
  created_at: string;
}

export default function DashboardPage() {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [processedVideos, setProcessedVideos] = useState<ProcessedVideo[]>([]);
  const [isLoadingVideos, setIsLoadingVideos] = useState(true);
  const [user, setUser] = useState<{ name: string; email: string } | null>(null);

  // Auth check on mount
  useEffect(() => {
    if (!isAuthenticated()) {
      router.push('/login');
      return;
    }
    setUser(getCurrentUser());
    loadVideos();
  }, [router]);

  const loadVideos = async () => {
    setIsLoadingVideos(true);
    try {
      const response = await fetch(`${API_BASE}/api/video/list`);
      if (response.ok) {
        const data = await response.json();
        setProcessedVideos(data.videos || []);
      }
    } catch (error) {
      console.error('Failed to load videos:', error);
    } finally {
      setIsLoadingVideos(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const processFiles = async (fileList: FileList) => {
    const videoExtensions = ['mp4', 'webm', 'avi', 'mov', 'mkv', 'flv', 'wmv'];
    const acceptedFiles: UploadedFile[] = [];

    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i];
      const extension = file.name.split('.').pop()?.toLowerCase();

      if (extension === 'zip' || videoExtensions.includes(extension || '')) {
        const newFile: UploadedFile = {
          id: Math.random().toString(36),
          name: file.name,
          size: file.size,
          type: extension === 'zip' ? 'zip' : 'video',
          file: file,
          progress: 0,
          status: 'pending',
        };
        acceptedFiles.push(newFile);
      }
    }

    if (acceptedFiles.length > 0) {
      setFiles((prev) => [...prev, ...acceptedFiles]);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    processFiles(e.dataTransfer.files);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      processFiles(e.target.files);
    }
  };

  const handleRemoveFile = (fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId));
  };

  const uploadFile = async (file: UploadedFile): Promise<string | null> => {
    const formData = new FormData();
    formData.append('file', file.file);

    try {
      // Update status to uploading
      setFiles(prev => prev.map(f => 
        f.id === file.id ? { ...f, status: 'uploading' as const, progress: 10 } : f
      ));

      const response = await fetch(`${API_BASE}/api/video/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      
      // Update status to uploaded
      setFiles(prev => prev.map(f => 
        f.id === file.id ? { ...f, status: 'uploaded' as const, progress: 100, videoId: data.video_id } : f
      ));

      return data.video_id;
    } catch (error) {
      console.error('Upload error:', error);
      setFiles(prev => prev.map(f => 
        f.id === file.id ? { ...f, status: 'error' as const } : f
      ));
      return null;
    }
  };

  const handleProcessVideos = async () => {
    if (files.length === 0) return;

    setIsUploading(true);
    const videoIds: string[] = [];

    try {
      // Upload all files
      for (const file of files) {
        if (file.status !== 'uploaded') {
          const videoId = await uploadFile(file);
          if (videoId) {
            videoIds.push(videoId);
          }
        } else if (file.videoId) {
          videoIds.push(file.videoId);
        }
      }

      if (videoIds.length > 0) {
        // Redirect to processing page with first video ID
        router.push(`/processing?videoId=${videoIds[0]}&all=${videoIds.join(',')}`);
      }
    } catch (error) {
      console.error('Upload error:', error);
    } finally {
      setIsUploading(false);
    }
  };

  const handleViewVideo = (videoId: string) => {
    router.push(`/chat?videoId=${videoId}`);
  };

  const handleDeleteVideo = async (videoId: string) => {
    try {
      await fetch(`${API_BASE}/api/video/${videoId}`, { method: 'DELETE' });
      loadVideos();
    } catch (error) {
      console.error('Delete error:', error);
    }
  };

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);
  const formattedSize =
    totalSize > 1024 * 1024
      ? `${(totalSize / (1024 * 1024)).toFixed(2)} MB`
      : `${(totalSize / 1024).toFixed(2)} KB`;

  const allUploaded = files.length > 0 && files.every(f => f.status === 'uploaded');

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-card dark:from-background dark:via-background dark:to-card/50">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/50 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-accent to-primary rounded-lg flex items-center justify-center">
              <Play className="w-5 h-5 text-primary-foreground" fill="currentColor" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">VideoRAG</span>
          </div>
          <div className="flex items-center gap-4">
            {user && (
              <span className="text-sm text-foreground/70">Welcome, {user.name}</span>
            )}
            <Button variant="outline" size="sm" onClick={logout}>
              Logout
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid lg:grid-cols-3 gap-8">
          {/* Upload Section */}
          <div className="lg:col-span-2">
            <h1 className="text-4xl font-bold text-foreground mb-2">Upload Your Videos</h1>
            <p className="text-foreground/60 mb-8">
              Upload video files (MP4, WebM, etc.) or ZIP archives containing multiple videos. We'll process them instantly.
            </p>

            {/* Upload Area */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`relative rounded-2xl border-2 border-dashed transition-all p-8 sm:p-12 text-center cursor-pointer group ${
                isDragging
                  ? 'border-primary bg-primary/5'
                  : 'border-border/50 bg-card/50 hover:border-primary/50 hover:bg-card/80'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".mp4,.webm,.avi,.mov,.mkv,.flv,.wmv,.zip"
                onChange={handleFileInputChange}
                className="hidden"
              />

              <div className="flex flex-col items-center gap-4">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center group-hover:from-primary/30 group-hover:to-accent/30 transition-colors">
                  <Upload className="w-8 h-8 text-primary" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-foreground">Drag videos here to upload</h3>
                  <p className="text-foreground/60 text-sm">or click to browse your computer</p>
                </div>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="bg-primary hover:bg-primary/90 text-primary-foreground"
                >
                  <FileVideo className="w-4 h-4 mr-2" />
                  Browse Files
                </Button>
              </div>
            </div>

            {/* Processed Videos Section */}
            <div className="mt-12">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-foreground">Your Videos</h2>
                <Button variant="ghost" size="sm" onClick={loadVideos} disabled={isLoadingVideos}>
                  <RefreshCw className={`w-4 h-4 mr-2 ${isLoadingVideos ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              </div>
              
              {isLoadingVideos ? (
                <div className="text-center py-8 text-foreground/60">Loading videos...</div>
              ) : processedVideos.length === 0 ? (
                <div className="text-center py-8 text-foreground/60">No processed videos yet. Upload one to get started!</div>
              ) : (
                <div className="grid gap-4">
                  {processedVideos.map((video) => (
                    <div key={video.video_id} className="p-4 rounded-lg bg-card border border-border/50 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <FileVideo className="w-8 h-8 text-primary" />
                        <div>
                          <p className="font-semibold text-foreground">{video.filename}</p>
                          <p className="text-sm text-foreground/60">
                            Status: <span className={
                              video.status === 'completed' ? 'text-green-500' :
                              video.status === 'processing' ? 'text-yellow-500' :
                              video.status === 'failed' ? 'text-red-500' : 'text-foreground/60'
                            }>{video.status}</span>
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {video.status === 'processing' && (
                          <Button size="sm" variant="outline" onClick={() => router.push(`/processing?videoId=${video.video_id}`)}>
                            View Progress
                          </Button>
                        )}
                        {video.has_memory && (
                          <Button size="sm" onClick={() => handleViewVideo(video.video_id)}>
                            Chat
                          </Button>
                        )}
                        <Button size="sm" variant="destructive" onClick={() => handleDeleteVideo(video.video_id)}>
                          <X className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* Files List */}
          <div className="lg:col-span-1">
            <div className="rounded-2xl border border-border/50 bg-card p-6 sticky top-24">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-bold text-foreground">Files to Upload</h2>
                <span className="px-2 py-1 rounded-full bg-primary/10 text-primary text-sm font-semibold">
                  {files.length}
                </span>
              </div>

              {files.length === 0 ? (
                <p className="text-foreground/60 text-sm text-center py-8">No files selected yet</p>
              ) : (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {files.map((file) => (
                    <div key={file.id} className="p-3 rounded-lg bg-background/50 border border-border/50">
                      <div className="flex items-start gap-2 mb-2">
                        {file.type === 'zip' ? (
                          <Archive className="w-4 h-4 text-primary flex-shrink-0 mt-1" />
                        ) : (
                          <FileVideo className="w-4 h-4 text-primary flex-shrink-0 mt-1" />
                        )}
                        <div className="flex-1 min-w-0">
                          <p className="text-sm font-medium text-foreground truncate">{file.name}</p>
                          <p className="text-xs text-foreground/60">
                            {(file.size / (1024 * 1024)).toFixed(2)} MB
                            {file.status === 'uploaded' && <span className="text-green-500 ml-2">Uploaded</span>}
                            {file.status === 'uploading' && <span className="text-yellow-500 ml-2">Uploading...</span>}
                            {file.status === 'error' && <span className="text-red-500 ml-2">Error</span>}
                          </p>
                        </div>
                        <button
                          onClick={() => handleRemoveFile(file.id)}
                          className="text-foreground/40 hover:text-foreground transition-colors flex-shrink-0"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      </div>
                      {/* Progress Bar */}
                      <div className="w-full h-1.5 bg-background rounded-full overflow-hidden">
                        <div
                          className={`h-full transition-all duration-300 ${
                            file.status === 'error' ? 'bg-red-500' : 'bg-gradient-to-r from-primary to-accent'
                          }`}
                          style={{ width: `${file.progress}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {files.length > 0 && (
                <div className="mt-6 space-y-3 pt-4 border-t border-border/50">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-foreground/60">Total size:</span>
                    <span className="font-semibold text-foreground">{formattedSize}</span>
                  </div>
                  <Button
                    onClick={handleProcessVideos}
                    disabled={isUploading || files.length === 0}
                    className="w-full bg-accent hover:bg-accent/90 text-accent-foreground"
                  >
                    {isUploading ? 'Uploading...' : (
                      <>
                        <Plus className="w-4 h-4 mr-2" />
                        Upload & Process
                      </>
                    )}
                  </Button>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
