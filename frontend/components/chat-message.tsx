import { User, Bot } from 'lucide-react';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: Date;
}

export function ChatMessage({ role, content, timestamp }: ChatMessageProps) {
  const isUser = role === 'user';

  // Basic markdown-like rendering for bold and line breaks
  const renderContent = (text: string) => {
    return text.split('\n').map((line, i) => {
      // Bold text handling
      const parts = line.split(/(\*\*.*?\*\*)/g);
      return (
        <div key={i} className="min-h-[1.25rem]">
          {parts.map((part, j) => {
            if (part.startsWith('**') && part.endsWith('**')) {
              return <strong key={j}>{part.slice(2, -2)}</strong>;
            }
            return part;
          })}
        </div>
      );
    });
  };

  return (
    <div className={`flex gap-3 mb-4 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
        isUser
          ? 'bg-primary/20 text-primary'
          : 'bg-accent/20 text-accent'
      }`}>
        {isUser ? (
          <User className="w-5 h-5" />
        ) : (
          <Bot className="w-5 h-5" />
        )}
      </div>

      {/* Message */}
      <div className={`flex-1 ${isUser ? 'text-right' : 'text-left'}`}>
        <div
          className={`inline-block max-w-xs sm:max-w-md lg:max-w-lg px-4 py-3 rounded-2xl ${
            isUser
              ? 'bg-primary text-primary-foreground rounded-tr-none'
              : 'bg-card border border-border/50 text-foreground rounded-tl-none shadow-sm'
          }
          `}
        >
          <div className="text-sm leading-relaxed break-words whitespace-pre-wrap">
            {renderContent(content)}
          </div>
        </div>
        {timestamp && (
          <p className="text-xs text-foreground/40 mt-1 px-1">
            {timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
          </p>
        )}
      </div>
    </div>
  );
}
