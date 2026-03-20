'use client';

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

interface VisualizerProps {
  activeMode: string;
  isTraining: boolean;
  trainingTrigger: number;
  onStopTraining: () => void;
}

type DataPoint = {
  epoch: number;
  loss: number;
  accuracy: number;
};

export default function Visualizer({ activeMode, isTraining, trainingTrigger, onStopTraining }: VisualizerProps) {
  const [data, setData] = useState<DataPoint[]>([]);

  useEffect(() => {
    if (!isTraining) return;

    // Очищаем предыдущие данные при новом запуске
    setData([]);
    
    // Подключаемся к SSE (Server-Sent Events) эндпоинту FastAPI
    const eventSource = new EventSource(`http://localhost:8000/api/v1/training/stream?mode=${activeMode}`);

    eventSource.onmessage = (event) => {
      const parsed = JSON.parse(event.data);
      
      if (parsed.error) {
        console.error('Ошибка обучения:', parsed.error);
        eventSource.close();
        onStopTraining();
        alert(`Ошибка: ${parsed.error}`);
        return;
      }
      
      setData(prev => [...prev, parsed]);
      
      // Останавливаем, если достигли максимума эпох
      if (parsed.epoch >= 50) {
        eventSource.close();
        onStopTraining();
      }
    };

    eventSource.onerror = (err) => {
      console.error('Ошибка EventSource:', err);
      eventSource.close();
      onStopTraining();
    };

    return () => {
      eventSource.close();
    };
  }, [isTraining, trainingTrigger, activeMode, onStopTraining]);

  return (
    <div className="flex flex-col h-full space-y-6">
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-slate-50 border border-slate-100 rounded-xl p-4">
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">Потери (Loss)</div>
          <div className="text-2xl font-bold text-slate-800">
            {data.length > 0 ? data[data.length - 1].loss.toFixed(4) : '---'}
          </div>
        </div>
        <div className="bg-slate-50 border border-slate-100 rounded-xl p-4">
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">Точность (Accuracy)</div>
          <div className="text-2xl font-bold text-emerald-600">
            {data.length > 0 ? (data[data.length - 1].accuracy * 100).toFixed(1) + '%' : '---'}
          </div>
        </div>
        <div className="bg-slate-50 border border-slate-100 rounded-xl p-4">
          <div className="text-slate-500 text-xs font-medium uppercase tracking-wider mb-1">Эпоха</div>
          <div className="text-2xl font-bold text-blue-600">
            {data.length > 0 ? `${data[data.length - 1].epoch} / 50` : '0 / 50'}
          </div>
        </div>
      </div>

      <div className="flex-1 min-h-[300px] bg-white border border-slate-100 rounded-xl p-4 shadow-sm">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis yAxisId="left" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
            <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={12} tickLine={false} axisLine={false} />
            <Tooltip 
              contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
            />
            <Legend iconType="circle" wrapperStyle={{ fontSize: '12px' }} />
            <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#ef4444" strokeWidth={2} dot={false} name="Потери" />
            <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#10b981" strokeWidth={2} dot={false} name="Точность" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
