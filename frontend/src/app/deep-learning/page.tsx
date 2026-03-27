"use client";

import React, { useState, useEffect, useRef, Suspense } from 'react';
import { 
  Play, 
  Settings, 
  Terminal as TerminalIcon, 
  Activity, 
  Cpu, 
  Database,
  RefreshCcw,
  ArrowLeft,
  ScanSearch,
  FlaskConical,
  LineChart,
  ShieldAlert
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
import { useSearchParams } from 'next/navigation';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '/api/v1';
const STREAM_API_BASE = process.env.NEXT_PUBLIC_STREAM_API_URL || API_BASE;

function DeepLearningContent() {
  type YoloPhase = 'idle' | 'queued' | 'preparing_dataset' | 'training' | 'done' | 'error';

  const searchParams = useSearchParams();
  const mode = searchParams.get('mode') || 'yolo';

  const [systemStatus, setSystemStatus] = useState<'online' | 'offline'>('offline');
  const [logs, setLogs] = useState<string[]>([]);
  const [isInitializing, setIsInitializing] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [deviceInfo, setDeviceInfo] = useState('Проверка устройства...');
  const [yoloPhase, setYoloPhase] = useState<YoloPhase>('idle');
  const [settings, setSettings] = useState({
    epochs: 1,
    fraction: 0.2,
    dataset_id: 'iqmansingh/guns-knives-object-detection',
    model_type: mode === 'yolo' ? 'yolo11n.pt' : mode === 'vision' ? 'resnet50' : mode === 'spectrum' ? '1d-cnn' : 'mlp'
  });

  const terminalEndRef = useRef<HTMLDivElement>(null);
  const failedStatusChecks = useRef(0);
  const yoloEventSourceRef = useRef<EventSource | null>(null);
  const yoloStreamClosingRef = useRef(false);

  const getLogTone = (log: string) => {
    if (log.includes('Ошибка')) return "border-rose-500 text-rose-400 bg-rose-500/5";
    if (log.includes('Успех') || log.includes('успешно завершено')) return "border-emerald-500 text-emerald-400 bg-emerald-500/5";
    if (/mAP|precision|recall/i.test(log)) return "border-cyan-500 text-cyan-300 bg-cyan-500/5";
    if (/epoch|эпоха|\d+\/\d+/i.test(log)) return "border-amber-500 text-amber-300 bg-amber-500/5";
    return "border-slate-800 text-slate-300";
  };

  const phaseStyles: Record<YoloPhase, string> = {
    idle: "bg-slate-500/10 text-slate-300 border-slate-500/20",
    queued: "bg-violet-500/10 text-violet-300 border-violet-500/20",
    preparing_dataset: "bg-blue-500/10 text-blue-300 border-blue-500/20",
    training: "bg-amber-500/10 text-amber-300 border-amber-500/20",
    done: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
    error: "bg-rose-500/10 text-rose-300 border-rose-500/20",
  };

  const phaseLabels: Record<YoloPhase, string> = {
    idle: "IDLE",
    queued: "QUEUED",
    preparing_dataset: "DATASET",
    training: "TRAINING",
    done: "DONE",
    error: "ERROR",
  };

  // Mode Configuration
  const modes = {
    vision: { title: 'Computer Vision 2D', icon: ShieldAlert, color: 'text-purple-500', initPath: '/vision/initialize' },
    yolo: { title: 'Object Detection YOLO', icon: ScanSearch, color: 'text-blue-500', initPath: '/vision/yolo/initialize' },
    spectrum: { title: 'Spectroscopy Analysis', icon: FlaskConical, color: 'text-emerald-500', initPath: '/spectrum/initialize' },
    math: { title: 'Mathematical Sandbox', icon: LineChart, color: 'text-amber-500', initPath: '/math/initialize' }
  };

  const modeInfo = modes[mode as keyof typeof modes] || modes.yolo;

  // Auto-scroll terminal
  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  useEffect(() => {
    return () => {
      yoloStreamClosingRef.current = true;
      yoloEventSourceRef.current?.close();
    };
  }, []);

  // Check system status
  useEffect(() => {
    const checkStatus = async () => {
      const controller = new AbortController();
      try {
        const response = await fetch(`${API_BASE}/status`, { signal: controller.signal, cache: 'no-store' });
        if (response.ok) {
          const data = await response.json();
          failedStatusChecks.current = 0;
          setSystemStatus(data.status === 'success' || data.status === 'online' ? 'online' : 'offline');
          setDeviceInfo(data.gpu_support ? `${data.framework}: ${data.device || 'GPU'} доступно` : `${data.framework}: ${data.device || 'CPU only'}`);
          if (typeof data.yolo_phase === 'string') setYoloPhase(data.yolo_phase as YoloPhase);
        } else {
          failedStatusChecks.current += 1;
          if (failedStatusChecks.current >= 2) {
            setSystemStatus('offline');
            setDeviceInfo('Бэкенд недоступен');
          }
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === 'AbortError') return;
        failedStatusChecks.current += 1;
        if (failedStatusChecks.current >= 2) {
          setSystemStatus('offline');
          setDeviceInfo('Бэкенд недоступен');
        }
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  const addLog = (message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs(prev => [...prev, `[${timestamp}] ${message}`]);
  };

  const handleInitialize = async () => {
    if (systemStatus === 'offline') {
      addLog("Ошибка: Бэкенд недоступен.");
      return;
    }
    setIsInitializing(true);
    addLog(`Инициализация модуля: ${modeInfo.title}...`);
    try {
      // Different payloads for different modules
      let body = {};
      if (mode === 'yolo') body = { model_type: settings.model_type };
      else if (mode === 'vision') body = { model_type: settings.model_type, num_classes: 2 };
      else if (mode === 'spectrum') body = { input_channels: 1, num_classes: 5, kernel_size: 3, dropout_rate: 0.5 };
      else body = { model_type: 'mlp', input_dim: 10, hidden_layers: [64, 32], output_dim: 1, activation: 'relu' };

      const response = await fetch(`${API_BASE}${modeInfo.initPath}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
        throw new Error(errorData.detail || 'Ошибка инициализации');
      }
      
      const data = await response.json();
      addLog(`Успех: ${data.message}`);
    } catch (err) {
      addLog(`Ошибка: ${err instanceof Error ? err.message : 'Не удалось инициализировать'}`);
    } finally {
      setIsInitializing(false);
    }
  };

  const handleStartTraining = async () => {
    if (systemStatus === 'offline') return;
    setIsTraining(true);
    addLog(`Запуск процесса обучения (${mode})...`);
    
    try {
      if (mode === 'yolo') {
        setYoloPhase('preparing_dataset');
        const response = await fetch(`${API_BASE}/vision/yolo/train`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            epochs: settings.epochs,
            fraction: settings.fraction,
            dataset_id: settings.dataset_id
          })
        });
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Ошибка запуска YOLO' }));
          throw new Error(errorData.detail || 'Ошибка запуска YOLO');
        }
        const data = await response.json();
        addLog(`Система: ${data.message}`);
        addLog("Подключение к потоку логов YOLO...");
        yoloStreamClosingRef.current = false;
        yoloEventSourceRef.current?.close();
        const eventSource = new EventSource(`${STREAM_API_BASE}/vision/yolo/stream`);
        yoloEventSourceRef.current = eventSource;

        eventSource.onmessage = (event) => {
          const streamData = JSON.parse(event.data);
          if (typeof streamData.phase === 'string') setYoloPhase(streamData.phase as YoloPhase);
          if (streamData.log) addLog(streamData.log);
          if (streamData.error) {
            addLog(`Ошибка: ${streamData.error}`);
            yoloStreamClosingRef.current = true;
            eventSource.close();
            yoloEventSourceRef.current = null;
            setIsTraining(false);
          }
          if (streamData.done) {
            yoloStreamClosingRef.current = true;
            eventSource.close();
            yoloEventSourceRef.current = null;
            setIsTraining(false);
          }
        };

        eventSource.onerror = () => {
          if (yoloStreamClosingRef.current) return;
          addLog("Ошибка соединения с потоком YOLO.");
          eventSource.close();
          yoloEventSourceRef.current = null;
          setYoloPhase('error');
          setIsTraining(false);
        };
      } else {
        addLog("Подключение к потоку метрик...");
        const eventSource = new EventSource(`${STREAM_API_BASE}/training/real_stream?mode=${mode}`);
        
        eventSource.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.log) addLog(data.log);
          if (data.error) {
            addLog(`Ошибка: ${data.error}`);
            eventSource.close();
            setIsTraining(false);
          }
          if (data.log && data.log.includes('успешно завершено')) {
            eventSource.close();
            setIsTraining(false);
          }
        };

        eventSource.onerror = () => {
          addLog("Ошибка соединения с потоком данных.");
          eventSource.close();
          setIsTraining(false);
        };
      }
    } catch (err) {
      addLog(`Ошибка: ${err instanceof Error ? err.message : 'Неизвестная ошибка'}`);
      setIsTraining(false);
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-6">
      {/* Header with Status */}
      <div className="flex items-center justify-between bg-slate-900/50 border border-slate-800 p-4 rounded-xl backdrop-blur-sm">
        <div className="flex items-center gap-4">
          <button 
            onClick={() => window.location.href = '/'}
            className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex items-center gap-3">
            <div className={cn("p-2 rounded-lg bg-opacity-10", modeInfo.color.replace('text-', 'bg-'))}>
              <modeInfo.icon className={cn("w-6 h-6", modeInfo.color)} />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight">{modeInfo.title}</h1>
              <p className="text-sm text-slate-400">Модуль управления нейронной сетью</p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className={cn(
            "flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium border",
            systemStatus === 'online' 
              ? "bg-emerald-500/10 text-emerald-500 border-emerald-500/20" 
              : "bg-rose-500/10 text-rose-500 border-rose-500/20"
          )}>
            <div className={cn(
              "w-2 h-2 rounded-full animate-pulse",
              systemStatus === 'online' ? "bg-emerald-500" : "bg-rose-500"
            )} />
            Engine: {systemStatus.toUpperCase()}
          </div>
          {mode === 'yolo' && (
            <div className={cn(
              "px-3 py-1 rounded-full text-sm font-medium border",
              phaseStyles[yoloPhase]
            )}>
              YOLO: {phaseLabels[yoloPhase]}
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Settings Panel */}
        <div className="lg:col-span-1 space-y-6">
          <div className="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden">
            <div className="p-4 border-b border-slate-800 bg-slate-800/30 flex items-center gap-2">
              <Settings className="w-4 h-4 text-slate-400" />
              <h2 className="font-semibold">Конфигурация</h2>
            </div>
            
            <div className="p-6 space-y-6">
              <div className="space-y-4">
                <label className="block text-sm font-medium text-slate-400">
                  Архитектура / Веса
                </label>
                <select 
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 outline-none"
                  value={settings.model_type}
                  onChange={(e) => setSettings({...settings, model_type: e.target.value})}
                >
                  {mode === 'yolo' && (
                    <>
                      <option value="yolo11n.pt">YOLOv11 Nano</option>
                      <option value="yolo11s.pt">YOLOv11 Small</option>
                    </>
                  )}
                  {mode === 'vision' && (
                    <>
                      <option value="resnet50">ResNet50 (Transfer Learning)</option>
                      <option value="efficientnet">EfficientNet-B0</option>
                    </>
                  )}
                  {mode === 'spectrum' && <option value="1d-cnn">1D-CNN (Chemistry)</option>}
                  {mode === 'math' && <option value="mlp">MLP Sandbox</option>}
                </select>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between">
                  <label className="text-sm font-medium text-slate-400">Epochs</label>
                  <span className="text-sm font-mono text-blue-400">{settings.epochs}</span>
                </div>
                <input 
                  type="range" min="1" max="100" step="1"
                  value={settings.epochs}
                  onChange={(e) => setSettings({...settings, epochs: parseInt(e.target.value)})}
                  className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                />
              </div>

              {mode === 'yolo' && (
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <label className="text-sm font-medium text-slate-400">Dataset Fraction</label>
                    <span className="text-sm font-mono text-blue-400">{Math.round(settings.fraction * 100)}%</span>
                  </div>
                  <input 
                    type="range" min="0.01" max="1.0" step="0.01"
                    value={settings.fraction}
                    onChange={(e) => setSettings({...settings, fraction: parseFloat(e.target.value)})}
                    className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-blue-500"
                  />
                </div>
              )}

              <div className="pt-4 space-y-3">
                <button 
                  onClick={handleInitialize}
                  disabled={isInitializing || systemStatus === 'offline'}
                  className="w-full flex items-center justify-center gap-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-white font-medium py-2.5 rounded-lg transition-all border border-slate-700"
                >
                  {isInitializing ? <RefreshCcw className="w-4 h-4 animate-spin" /> : <Database className="w-4 h-4" />}
                  Инициализировать
                </button>

                <button 
                  onClick={handleStartTraining}
                  disabled={isTraining || systemStatus === 'offline'}
                  className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white font-bold py-3 rounded-lg transition-all shadow-lg shadow-blue-500/20"
                >
                  {isTraining ? <Activity className="w-5 h-5 animate-pulse" /> : <Play className="w-5 h-5" />}
                  Запустить
                </button>
              </div>
            </div>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-4">
            <div className="p-3 bg-emerald-500/10 rounded-lg">
              <Cpu className="w-6 h-6 text-emerald-500" />
            </div>
            <div>
              <p className="text-xs text-slate-400 uppercase font-bold tracking-wider">Device Detected</p>
              <p className="font-mono text-sm">{deviceInfo}</p>
            </div>
          </div>
        </div>

        {/* Terminal Section */}
        <div className="lg:col-span-2 space-y-6">
          <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden h-[600px] flex flex-col shadow-2xl">
            <div className="p-4 border-b border-slate-800 bg-slate-900 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <TerminalIcon className="w-4 h-4 text-emerald-500" />
                <h2 className="font-mono text-sm font-bold text-emerald-500">LAB_TERMINAL_V1.0</h2>
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 font-mono text-sm space-y-1 custom-scrollbar">
              {logs.length === 0 && (
                <div className="text-slate-600 italic">Ожидание команд...</div>
              )}
              {logs.map((log, i) => (
                <div key={i} className={cn(
                  "border-l-2 pl-3 py-0.5",
                  getLogTone(log)
                )}>
                  {log}
                </div>
              ))}
              <div ref={terminalEndRef} />
            </div>

            <div className="p-3 bg-slate-900 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-500 font-mono uppercase tracking-widest">
              <span>Status: {isTraining ? 'Running' : 'Idle'}</span>
              <span>Module: {mode.toUpperCase()}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function DeepLearningPage() {
  return (
    <Suspense fallback={<div>Loading Lab...</div>}>
      <DeepLearningContent />
    </Suspense>
  );
}
