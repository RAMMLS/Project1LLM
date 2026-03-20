'use client';

import React, { useState } from 'react';
import { Settings, PlayCircle, RefreshCw, TerminalSquare } from 'lucide-react';
import { useRouter } from 'next/navigation';
import clsx from 'clsx';

interface Mode {
  id: string;
  title: string;
  icon: React.ReactNode;
}

interface DashboardProps {
  modes: Mode[];
  activeMode: string;
  onModeChange: (id: string) => void;
  isInitializing: boolean;
  isTraining: boolean;
  onStartTraining: () => void;
}

export default function Dashboard({ modes, activeMode, onModeChange, isInitializing, isTraining, onStartTraining }: DashboardProps) {
  const [isStarting, setIsStarting] = useState(false);
  const [isStartingDeep, setIsStartingDeep] = useState(false);
  const router = useRouter();

  const handleStartClick = async () => {
    setIsStarting(true);
    try {
      const response = await fetch(`http://localhost:8000/api/v1/${activeMode}/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}) 
      });
      
      if (response.ok) {
        onStartTraining();
      } else {
        const data = await response.json();
        alert(`Ошибка инициализации: ${data.detail || 'Неизвестная ошибка'}`);
      }
    } catch (error) {
      console.error('Ошибка при обращении к API:', error);
      alert('Ошибка соединения с бэкендом. Убедитесь, что сервер запущен.');
    } finally {
      setIsStarting(false);
    }
  };

  const handleDeepLearningClick = async () => {
    setIsStartingDeep(true);
    try {
      // Пингуем бэкенд, чтобы он проинициализировал модель ДО ТОГО, как мы перейдем на страницу логов
      const response = await fetch(`http://localhost:8000/api/v1/${activeMode}/initialize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}) 
      });
      
      if (response.ok) {
        // Если всё гуд - редиректим юзера в хакерский терминал
        router.push(`/deep-learning?mode=${activeMode}`);
      } else {
        const data = await response.json();
        alert(`Ошибка инициализации: ${data.detail || 'Неизвестная ошибка'}`);
        setIsStartingDeep(false);
      }
    } catch (error) {
      console.error('Ошибка при обращении к API:', error);
      alert('Ошибка соединения с бэкендом. Убедитесь, что сервер запущен.');
      setIsStartingDeep(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden">
      <div className="p-5 border-b border-slate-100 bg-slate-50/50">
        <h2 className="text-lg font-semibold text-slate-800 flex items-center gap-2">
          <Settings className="w-5 h-5 text-slate-500" />
          Режим задачи
        </h2>
      </div>
      
      <div className="p-5 space-y-4">
        <div className="space-y-2">
          {modes.map((mode) => (
            <button
              key={mode.id}
              onClick={() => onModeChange(mode.id)}
              disabled={isInitializing || isTraining}
              className={clsx(
                "w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all duration-200 border",
                activeMode === mode.id 
                  ? "bg-blue-50 border-blue-200 text-blue-700 shadow-sm" 
                  : "bg-white border-slate-100 text-slate-600 hover:bg-slate-50 hover:border-slate-200",
                (isInitializing || isTraining) && "opacity-50 cursor-not-allowed"
              )}
            >
              <div className={clsx(
                "p-2 rounded-lg",
                activeMode === mode.id ? "bg-blue-100 text-blue-600" : "bg-slate-100 text-slate-500"
              )}>
                {mode.icon}
              </div>
              <span className="font-medium text-sm">{mode.title}</span>
            </button>
          ))}
        </div>

        <div className="pt-4 border-t border-slate-100 space-y-3">
          <button
            onClick={handleStartClick}
            disabled={isInitializing || isStarting || isTraining || isStartingDeep}
            className="w-full flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-800 text-white px-4 py-3 rounded-xl font-medium transition-colors disabled:opacity-50"
          >
            {(isInitializing || isStarting || isTraining) ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <PlayCircle className="w-5 h-5" />
            )}
            {isStarting ? 'Запуск...' : isTraining ? 'Обучение идет...' : 'Демо обучение (Графики)'}
          </button>

          <button
            onClick={handleDeepLearningClick}
            disabled={isInitializing || isStarting || isTraining || isStartingDeep}
            className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-3 rounded-xl font-medium transition-colors disabled:opacity-50"
          >
            {isStartingDeep ? (
              <RefreshCw className="w-5 h-5 animate-spin" />
            ) : (
              <TerminalSquare className="w-5 h-5" />
            )}
            {isStartingDeep ? 'Инициализация...' : 'Глубокое обучение (Логи)'}
          </button>
        </div>
      </div>
    </div>
  );
}
