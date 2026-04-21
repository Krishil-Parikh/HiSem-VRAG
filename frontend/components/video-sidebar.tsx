import React from "react";
import { Video, Maximize2, Play, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";

interface VideoItem {
  id: string;
  name: string;
  duration: string;
  size: string;
  source?: 'uploaded' | 'dataset';
  status: string;
}

interface VideoSidebarProps {
  videos: VideoItem[];
  selectedVideoIds: string[];
  onSelectVideo?: (videoId: string) => void;
  onDeleteVideo?: (videoId: string) => void;
  onSelectGlobal?: () => void;
  isGlobalMode?: boolean;
}

export function VideoSidebar({
  videos,
  selectedVideoIds,
  onSelectVideo,
  onDeleteVideo,
  onSelectGlobal,
  isGlobalMode,
}: VideoSidebarProps) {
  return (
    <div className="w-full h-full flex flex-col bg-card">
      {/* Header with Global Search */}
      <div className="border-b border-border/50 p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Video className="w-5 h-5 text-primary" />
            <h2 className="font-bold text-foreground">Videos</h2>
          </div>
          {selectedVideoIds.length > 0 && (
            <Button 
              variant="ghost" 
              size="sm" 
              className="text-[10px] h-6 px-2 text-foreground/40 hover:text-primary"
              onClick={() => onSelectGlobal?.()}
            >
              Clear
            </Button>
          )}
        </div>

        <Button 
          variant={isGlobalMode ? "default" : "outline"} 
          className="w-full justify-start gap-2 h-10 border-dashed border-primary/30"
          onClick={() => onSelectGlobal?.()}
        >
          <Maximize2 className="w-4 h-4" />
          <span className="text-xs font-bold">
            {isGlobalMode ? "Global Database Search" : "Switch to Global"}
          </span>
        </Button>
      </div>

      {/* Videos List */}
      <div className="flex-1 overflow-y-auto custom-scrollbar">
        {videos.length === 0 ? (
          <div className="p-4 text-center">
            <p className="text-sm text-foreground/60">No videos found</p>
          </div>
        ) : (
          <div className="p-3 space-y-2">
            {videos.map((video) => {
              const isSelected = selectedVideoIds.includes(video.id);
              return (
                <div
                  key={video.id}
                  onClick={() => onSelectVideo?.(video.id)}
                  className={`p-3 rounded-lg border cursor-pointer transition-all group relative ${
                    isSelected
                      ? 'border-primary bg-primary/10 ring-1 ring-primary/30'
                      : 'border-border/50 bg-background/50 hover:border-primary/50 hover:bg-background'
                  }`}
                >
                  {/* Source Badge */}
                  {video.source && (
                    <div className={`absolute top-2 right-2 px-1.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider z-10 ${
                      video.source === 'dataset' 
                        ? 'bg-accent/20 text-accent' 
                        : 'bg-primary/20 text-primary'
                    }`}>
                      {video.source}
                    </div>
                  )}

                  {/* Thumbnail Placeholder */}
                  <div className="w-full h-20 bg-gradient-to-br from-primary/20 to-accent/20 rounded-md mb-2 flex items-center justify-center group-hover:from-primary/30 group-hover:to-accent/30 transition-colors pointer-events-none">
                    <Play className={`w-5 h-5 text-primary transition-opacity ${isSelected ? 'opacity-100' : 'opacity-50'}`} fill="currentColor" />
                  </div>

                  {/* Video Info */}
                  <h3 className="text-[11px] font-semibold text-foreground truncate mb-1 pr-12">
                    {video.name}
                  </h3>
                  <div className="flex items-center justify-between text-[10px] text-foreground/60">
                    <span>{video.duration}</span>
                    <span className={video.status === 'completed' ? 'text-green-500' : ''}>
                      {video.status}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border/50 p-3 bg-muted/30">
        <p className="text-[10px] text-foreground/50 text-center flex items-center justify-center gap-2">
          {selectedVideoIds.length > 0 ? (
             <><span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" /> Using {selectedVideoIds.length} videos</>
          ) : (
            'Searching entire database'
          )}
        </p>
      </div>
    </div>
  );
}
