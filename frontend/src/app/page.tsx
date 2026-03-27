"use client";

import React from 'react';
import { 
  ScanSearch, 
  FlaskConical, 
  LineChart, 
  ShieldAlert, 
  ArrowRight,
  BrainCircuit,
  Activity,
  Layers,
  Cpu
} from 'lucide-react';
import { useRouter } from 'next/navigation';

const modules = [
  {
    id: 'vision',
    title: 'Computer Vision 2D',
    description: 'Классификация изображений (Кошки/Собаки) и детекция подделок документов с использованием ResNet50/EfficientNet.',
    icon: ShieldAlert,
    color: 'text-purple-500',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/20',
    path: '/deep-learning?mode=vision'
  },
  {
    id: 'yolo',
    title: 'Object Detection YOLOv11',
    description: 'Детекция опасных предметов (оружие, ножи) в реальном времени. Портировано из Jupyter Notebook.',
    icon: ScanSearch,
    color: 'text-blue-500',
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/20',
    path: '/deep-learning?mode=yolo'
  },
  {
    id: 'spectrum',
    title: 'Spectroscopy Analysis',
    description: 'Химический анализ данных спектрометрии с использованием 1D-CNN нейронных сетей.',
    icon: FlaskConical,
    color: 'text-emerald-500',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
    path: '/deep-learning?mode=spectrum'
  },
  {
    id: 'math',
    title: 'Mathematical Sandbox',
    description: 'Экономическое моделирование и функциональные аппроксимации с помощью MLP и RNN.',
    icon: LineChart,
    color: 'text-amber-500',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    path: '/deep-learning?mode=math'
  }
];

export default function Dashboard() {
  const router = useRouter();

  const handleModuleClick = (path: string) => {
    router.push(path);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200">
      {/* Hero Section */}
      <div className="relative overflow-hidden border-b border-slate-800 bg-slate-900/20 backdrop-blur-sm">
        <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px] -mr-64 -mt-64" />
        <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-purple-600/10 rounded-full blur-[120px] -ml-64 -mb-64" />
        
        <div className="max-w-7xl mx-auto px-6 py-20 relative z-10">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-blue-600 rounded-lg shadow-lg shadow-blue-500/20">
              <BrainCircuit className="w-6 h-6 text-white" />
            </div>
            <span className="text-sm font-mono font-bold tracking-[0.3em] text-blue-500 uppercase">System Core v1.0</span>
          </div>
          <h1 className="text-5xl md:text-6xl font-extrabold tracking-tight text-white mb-6">
            AI Lab & <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">Analysis Platform</span>
          </h1>
          <p className="text-xl text-slate-400 max-w-2xl leading-relaxed">
            Профессиональная среда для обучения, тестирования и развертывания нейронных сетей различных архитектур в едином интерфейсе.
          </p>
        </div>
      </div>

      {/* Grid Section */}
      <div className="max-w-7xl mx-auto px-6 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {modules.map((module) => (
            <div 
              key={module.id}
              onClick={() => handleModuleClick(module.path)}
              className="group relative bg-slate-900 border border-slate-800 rounded-2xl p-8 hover:border-slate-700 hover:bg-slate-800/50 transition-all cursor-pointer overflow-hidden shadow-xl"
            >
              <div className={`absolute top-0 right-0 w-32 h-32 ${module.bg} rounded-full blur-3xl -mr-16 -mt-16 opacity-50 group-hover:opacity-100 transition-opacity`} />
              
              <div className="relative z-10">
                <div className={`w-14 h-14 ${module.bg} ${module.border} border rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <module.icon className={`w-7 h-7 ${module.color}`} />
                </div>
                
                <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-blue-400 transition-colors">
                  {module.title}
                </h3>
                <p className="text-slate-400 leading-relaxed mb-8">
                  {module.description}
                </p>
                
                <div className="flex items-center gap-2 text-sm font-bold uppercase tracking-wider text-slate-500 group-hover:text-white transition-colors">
                  Запустить модуль
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Stats Footer */}
      <div className="max-w-7xl mx-auto px-6 pb-16">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
          {[
            { label: 'Active Engine', value: 'Online', icon: Activity, color: 'text-emerald-500' },
            { label: 'Compute Unit', value: 'RTX 4090', icon: Cpu, color: 'text-blue-500' },
            { label: 'Architecture', value: 'Unified API', icon: Layers, color: 'text-purple-500' }
          ].map((stat, i) => (
            <div key={i} className="bg-slate-900/50 border border-slate-800/50 rounded-xl p-4 flex items-center gap-4">
              <stat.icon className={`w-5 h-5 ${stat.color}`} />
              <div>
                <p className="text-[10px] font-mono text-slate-500 uppercase tracking-widest">{stat.label}</p>
                <p className="text-sm font-bold text-slate-300">{stat.value}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
