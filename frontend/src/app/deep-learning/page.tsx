'use client';

import React, { useEffect, useState, Suspense } from 'react';
import { useSearchParams, useRouter } from 'next/navigation';
import { Terminal, ArrowLeft, Activity, ShieldAlert } from 'lucide-react';

function DeepLearningConsole() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const mode = searchParams.get('mode') || 'vision';
  
  const [logs, setLogs] = useState<string[]>([]);
  const [graphViz, setGraphViz] = useState<string[] | null>(null);
  const [isFinished, setIsFinished] = useState(false);

  useEffect(() => {
    // Коннектимся к ручке бэкенда, откуда будут лететь логи РЕАЛЬНОГО обучения (SSE)
    const eventSource = new EventSource(`http://localhost:8000/api/v1/training/real_stream?mode=${mode}`);
    
    eventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      
      // Если прилетел лог - пушим его в стейт, чтобы отрисовать в консоли
      if (parsed.log) {
        setLogs(prev => [...prev, parsed.log]);
      }

      // Если прилетела структура сетки - сохраняем отдельно для красивого вывода
      if (parsed.graph) {
        setGraphViz(parsed.graph);
      }
      
      // Вырубаем коннект, если словили ошибку или обучение закончилось
      if (parsed.error || parsed.log === 'Обучение успешно завершено.') {
        setIsFinished(true);
        eventSource.close();
      }
    };

    eventSource.onerror = () => {
      setLogs(prev => [...prev, '[Система] Соединение с PyTorch прервано.']);
      setIsFinished(true);
      eventSource.close();
    };

    return () => eventSource.close();
  }, [mode]);

  return (
    <div className="min-h-screen bg-slate-950 text-emerald-400 font-mono p-6">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center justify-between mb-6 border-b border-slate-800 pb-4">
          <div className="flex items-center gap-3">
            <Terminal className="w-6 h-6 text-emerald-500" />
            <h1 className="text-xl font-bold text-slate-100">Терминал глубокого обучения: {mode.toUpperCase()}</h1>
          </div>
          <button 
            onClick={() => router.push('/')}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 rounded-lg transition-colors"
          >
            <ArrowLeft className="w-4 h-4" /> Назад к Дашборду
          </button>
        </div>
        
        <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 h-[70vh] overflow-y-auto shadow-2xl relative scrollbar-hide">
          {logs.length === 0 && (
            <div className="text-slate-500 animate-pulse">Ожидание подключения к PyTorch Engine...</div>
          )}

          {graphViz && (
            <div className="mb-8 p-6 bg-slate-950/50 border border-slate-800 rounded-lg">
              <div className="text-slate-400 mb-4 font-semibold tracking-wider text-sm">// АРХИТЕКТУРА МОДЕЛИ (ГРАФ ВЫЧИСЛЕНИЙ):</div>
              <div className="flex flex-wrap items-center gap-2">
                {graphViz.map((step, index) => (
                  <React.Fragment key={index}>
                    <div className="px-3 py-1.5 bg-fuchsia-900/30 border border-fuchsia-500/30 text-fuchsia-400 rounded text-sm whitespace-nowrap shadow-[0_0_10px_rgba(217,70,239,0.1)]">
                      {step}
                    </div>
                    {index < graphViz.length - 1 && (
                      <div className="text-slate-600 animate-pulse">→</div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            </div>
          )}
          
          {logs.map((log, i) => {
            let textColor = 'text-blue-300'; // Default / Info
            if (log.includes('Ошибка')) textColor = 'text-rose-400';
            else if (log.includes('Эпоха')) textColor = 'text-emerald-400';
            else if (log.includes('завершено')) textColor = 'text-amber-400 font-bold';

            return (
              <div key={i} className="mb-2 flex gap-4">
                <span className="text-slate-600 shrink-0">[{new Date().toLocaleTimeString()}]</span>
                <span className={textColor}>
                  {log}
                </span>
              </div>
            );
          })}

          {isFinished && (
            <div className="mt-6 text-amber-500 font-bold flex items-center gap-2 border-t border-slate-800 pt-4">
              <Activity className="w-5 h-5" /> Процесс остановлен. Графы вычислений выгружены из памяти.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function DeepLearningPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-slate-950 text-emerald-400 font-mono p-10 text-center">Загрузка терминала...</div>}>
      <DeepLearningConsole />
    </Suspense>
  );
}
