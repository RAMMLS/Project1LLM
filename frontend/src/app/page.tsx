'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Dashboard from '../components/Dashboard';
import Visualizer from '../components/Visualizer';
import InterpretationPanel from '../components/InterpretationPanel';
import { BrainCircuit, Activity, Settings2, Database } from 'lucide-react';
import Link from 'next/link';

export default function Home() {
  const [activeMode, setActiveMode] = useState('vision');
  const [isInitializing, setIsInitializing] = useState(false);
  const [systemStatus, setSystemStatus] = useState('offline');
  
  // Состояния для реального обучения
  const [isTraining, setIsTraining] = useState(false);
  const [trainingTrigger, setTrainingTrigger] = useState(0);

  // Определения режимов
  const modes = [
    { id: 'vision', title: 'Подделка документов (2D Зрение)', icon: <BrainCircuit className="w-5 h-5" /> },
    { id: 'spectrum', title: 'Химия (1D Спектроскопия)', icon: <Activity className="w-5 h-5" /> },
    { id: 'math', title: 'Экономика (Мат. моделирование)', icon: <Settings2 className="w-5 h-5" /> },
  ];

  // Получение статуса бэкенда при загрузке (пингуем сервак, жив ли он)
  useEffect(() => {
    fetch('http://localhost:8000/api/v1/status')
      .then(res => res.json())
      .then(data => setSystemStatus(data.status))
      .catch(() => setSystemStatus('offline'));
  }, []);

  const handleModeChange = (modeId: string) => {
    if (isTraining) return; // Если модель щас обучается, блочим переключение режимов от греха подальше
    setIsInitializing(true);
    setActiveMode(modeId);
    
    // Фейковая задержка для красоты UI (типа что-то грузится)
    setTimeout(() => {
      setIsInitializing(false);
    }, 800);
  };

  const handleStartTraining = useCallback(() => {
    setIsTraining(true);
    setTrainingTrigger(prev => prev + 1);
  }, []);

  const handleStopTraining = useCallback(() => {
    setIsTraining(false);
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Шапка */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg text-white">
              <BrainCircuit className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900">Единая ИИ Лаборатория</h1>
              <p className="text-xs text-slate-500 font-medium">Платформа для анализа и обучения</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 text-sm font-medium">
              <div className={`w-2 h-2 rounded-full ${systemStatus === 'online' ? 'bg-emerald-500' : 'bg-rose-500'}`}></div>
              {systemStatus === 'online' ? 'Движок в сети' : 'Движок отключен'}
            </div>
          </div>
        </div>
      </header>

      {/* Основной контент */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Левая панель - Управление */}
          <div className="lg:col-span-3 space-y-6">
            <Dashboard 
              modes={modes} 
              activeMode={activeMode} 
              onModeChange={handleModeChange} 
              isInitializing={isInitializing}
              isTraining={isTraining}
              onStartTraining={handleStartTraining}
            />
            <InterpretationPanel activeMode={activeMode} />
          </div>

          {/* Правая панель - Визуализатор */}
          <div className="lg:col-span-9">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden h-full min-h-[600px] flex flex-col">
              <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                <h2 className="text-lg font-semibold text-slate-800">Телеметрия обучения</h2>
                <div className="flex gap-2">
                  <span className={`px-2.5 py-1 text-xs font-semibold rounded-md border ${isTraining ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-slate-100 text-slate-600 border-slate-200'}`}>
                    {isTraining ? 'Обучение в процессе...' : 'Ожидание'}
                  </span>
                  <span className="px-2.5 py-1 bg-purple-50 text-purple-700 text-xs font-semibold rounded-md border border-purple-100">
                    PyTorch Engine
                  </span>
                </div>
              </div>
              <div className="p-6 flex-1 flex flex-col">
                <Visualizer 
                  activeMode={activeMode} 
                  isTraining={isTraining}
                  trainingTrigger={trainingTrigger}
                  onStopTraining={handleStopTraining}
                />
              </div>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
