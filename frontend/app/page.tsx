'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { Play, ArrowRight, Zap, BarChart3, Lock } from 'lucide-react';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-card dark:from-background dark:via-background dark:to-card/50">
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 border-b border-border/50 bg-background/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-gradient-to-br from-accent to-primary rounded-lg flex items-center justify-center">
              <Play className="w-5 h-5 text-primary-foreground" fill="currentColor" />
            </div>
            <span className="text-xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">VideoRAG</span>
          </div>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-foreground/70 hover:text-foreground transition-colors">Features</a>
            <a href="#how-it-works" className="text-foreground/70 hover:text-foreground transition-colors">How it Works</a>
            <Link href="/login">
              <Button variant="outline" size="sm">Login</Button>
            </Link>
            <Link href="/signup">
              <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">Get Started</Button>
            </Link>
          </div>
          <div className="md:hidden">
            <Link href="/login">
              <Button size="sm" className="bg-primary hover:bg-primary/90 text-primary-foreground">Login</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-5xl mx-auto text-center">
          <div className="inline-block mb-4 px-3 py-1 rounded-full bg-accent/10 border border-accent/20">
            <span className="text-sm font-semibold text-accent">AI-Powered Video Intelligence</span>
          </div>
          <h1 className="text-5xl sm:text-6xl md:text-7xl font-bold mb-6 leading-tight text-foreground">
            Unlock Video <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">Intelligence</span> with AI
          </h1>
          <p className="text-lg sm:text-xl text-foreground/70 mb-8 max-w-2xl mx-auto leading-relaxed">
            Upload your videos, process them with advanced AI, and chat with your content. Get instant answers and insights from any video in seconds.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/signup">
              <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground w-full sm:w-auto">
                Start Free
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </Link>
            <Button size="lg" variant="outline">
              Watch Demo
              <Play className="w-4 h-4 ml-2" fill="currentColor" />
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 border-t border-border/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl sm:text-5xl font-bold mb-4 text-foreground">Powerful Features</h2>
            <p className="text-foreground/60 max-w-2xl mx-auto text-lg">Everything you need to analyze and interact with your video content</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {/* Feature 1 */}
            <div className="group p-6 rounded-2xl border border-border/50 bg-card hover:border-accent/50 transition-colors hover:bg-card/80">
              <div className="w-12 h-12 bg-gradient-to-br from-primary/20 to-accent/20 rounded-xl flex items-center justify-center mb-4 group-hover:from-primary/30 group-hover:to-accent/30 transition-colors">
                <Zap className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-foreground">Instant Processing</h3>
              <p className="text-foreground/60">Upload any video format and get it processed instantly. ZIP files, MP4, WebM - we handle it all.</p>
            </div>

            {/* Feature 2 */}
            <div className="group p-6 rounded-2xl border border-border/50 bg-card hover:border-accent/50 transition-colors hover:bg-card/80">
              <div className="w-12 h-12 bg-gradient-to-br from-primary/20 to-accent/20 rounded-xl flex items-center justify-center mb-4 group-hover:from-primary/30 group-hover:to-accent/30 transition-colors">
                <BarChart3 className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-foreground">Smart Chat Interface</h3>
              <p className="text-foreground/60">Chat freely with your videos. Ask questions, get insights, and understand your content better.</p>
            </div>

            {/* Feature 3 */}
            <div className="group p-6 rounded-2xl border border-border/50 bg-card hover:border-accent/50 transition-colors hover:bg-card/80">
              <div className="w-12 h-12 bg-gradient-to-br from-primary/20 to-accent/20 rounded-xl flex items-center justify-center mb-4 group-hover:from-primary/30 group-hover:to-accent/30 transition-colors">
                <Lock className="w-6 h-6 text-primary" />
              </div>
              <h3 className="text-xl font-bold mb-2 text-foreground">Your Data, Secure</h3>
              <p className="text-foreground/60">Enterprise-grade security and privacy. Your videos are encrypted and protected at all times.</p>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl sm:text-5xl font-bold mb-4 text-foreground">How It Works</h2>
            <p className="text-foreground/60 max-w-2xl mx-auto text-lg">Three simple steps to unlock your video intelligence</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: '1', title: 'Upload Videos', description: 'Upload any video or ZIP file containing videos. Our AI will automatically process and prepare them.' },
              { step: '2', title: 'Process & Index', description: 'Our advanced system processes your videos and creates an intelligent index for fast retrieval.' },
              { step: '3', title: 'Chat & Analyze', description: 'Chat with your videos, ask questions, and get instant AI-powered answers and insights.' }
            ].map((item) => (
              <div key={item.step} className="flex flex-col items-center text-center">
                <div className="w-16 h-16 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center mb-4">
                  <span className="text-2xl font-bold text-primary-foreground">{item.step}</span>
                </div>
                <h3 className="text-xl font-bold mb-2 text-foreground">{item.title}</h3>
                <p className="text-foreground/60">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8 border-t border-border/50">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl sm:text-5xl font-bold mb-6 text-foreground">Ready to Get Started?</h2>
          <p className="text-lg text-foreground/60 mb-8 max-w-xl mx-auto">Join thousands of users who are already leveraging AI to understand their video content.</p>
          <Link href="/signup">
            <Button size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground">
              Start Your Free Trial
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 py-8 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-gradient-to-br from-accent to-primary rounded-lg flex items-center justify-center">
              <Play className="w-4 h-4 text-primary-foreground" fill="currentColor" />
            </div>
            <span className="font-semibold text-foreground">VideoRAG</span>
          </div>
          <p className="text-foreground/60 text-sm">&copy; 2024 VideoRAG. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}
