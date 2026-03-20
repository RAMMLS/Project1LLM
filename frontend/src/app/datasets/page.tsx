'use client';

import React from 'react';
import { useRouter } from 'next/navigation';
import { Database, ArrowLeft, ExternalLink, Image as ImageIcon, Activity, TrendingUp } from 'lucide-react';

export default function DatasetsPage() {
  const router = useRouter();

  const datasets = [
    {
      id: 'vision',
      title: 'MIDV-500: Mobile Identity Document Video dataset',
      domain: 'Подделка документов (2D Vision)',
      icon: <ImageIcon className="w-6 h-6 text-blue-500" />,
      description: 'Огромный датасет из 500 видеоклипов удостоверений личности из 50 разных стран (паспорта, права, ID-карты). Идеально подходит для обучения моделей распознаванию фальшивок, проверки границ документов и поиска геометрических искажений.',
      features: ['50 видов документов', 'Различные фоны и освещение', 'Искажения (блики, тени, углы)'],
      kaggleLink: 'https://github.com/fcakyon/MIDV-500' // Реальный датасет, хоть и часто хостится на github/zenodo
    },
    {
      id: 'spectrum',
      title: 'Kaggle: Soil Chemistry / FT-IR Spectroscopy',
      domain: 'Химия (1D Спектроскопия)',
      icon: <Activity className="w-6 h-6 text-fuchsia-500" />,
      description: 'Реальный датасет спектроскопии в ближней инфракрасной области (NIR/FT-IR). Используется для предсказания химического состава почвы (углерод, азот) только по одномерным спектральным кривым. Отлично тестирует 1D-CNN.',
      features: ['Одномерные спектры (1024 фичи)', 'Шумные данные', 'Задачи регрессии и классификации'],
      kaggleLink: 'https://www.kaggle.com/c/afsis-soil-properties'
    },
    {
      id: 'math',
      title: 'G-Research: M-5 Forecasting (Walmart Data)',
      domain: 'Экономика (Мат. моделирование)',
      icon: <TrendingUp className="w-6 h-6 text-emerald-500" />,
      description: 'Легендарный датасет M5 Forecasting от Walmart на Kaggle. Содержит исторические данные о ежедневных продажах тысяч товаров. Идеально для обучения рекуррентных сетей (RNN/LSTM) предсказывать спрос (ROI) с учетом сезонности и волатильности.',
      features: ['Временные ряды (Time Series)', 'Иерархические данные', 'Сильная сезонность и праздники'],
      kaggleLink: 'https://www.kaggle.com/c/m5-forecasting-accuracy'
    }
  ];

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
      {/* Шапка */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="bg-indigo-600 p-2 rounded-lg text-white">
              <Database className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-xl font-bold tracking-tight text-slate-900">Каталог Датасетов</h1>
              <p className="text-xs text-slate-500 font-medium">Реальные данные для ИИ Лаборатории</p>
            </div>
          </div>
          <button 
            onClick={() => router.push('/')}
            className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg transition-colors font-medium text-sm"
          >
            <ArrowLeft className="w-4 h-4" /> Назад к Лаборатории
          </button>
        </div>
      </header>

      {/* Основной контент */}
      <main className="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="mb-10 text-center">
          <h2 className="text-3xl font-bold text-slate-800 mb-4">На чем учатся наши модели?</h2>
          <p className="text-slate-600 max-w-2xl mx-auto text-lg">
            Чтобы нейросети не просто выдавали рандомные цифры, им нужны качественные данные. 
            Здесь собраны реальные датасеты из открытых источников (Kaggle и GitHub), 
            которые идеально подходят для проверки наших архитектур.
          </p>
        </div>

        <div className="grid gap-8">
          {datasets.map((ds) => (
            <div key={ds.id} className="bg-white rounded-2xl p-8 border border-slate-200 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex flex-col md:flex-row gap-6 items-start">
                
                <div className="shrink-0 p-4 bg-slate-50 rounded-xl border border-slate-100">
                  {ds.icon}
                </div>
                
                <div className="flex-1">
                  <div className="text-xs font-bold tracking-wider text-indigo-600 uppercase mb-2">
                    {ds.domain}
                  </div>
                  <h3 className="text-xl font-bold text-slate-800 mb-3">{ds.title}</h3>
                  <p className="text-slate-600 mb-6 leading-relaxed">
                    {ds.description}
                  </p>
                  
                  <div className="mb-6">
                    <div className="text-sm font-semibold text-slate-800 mb-2">Ключевые фичи:</div>
                    <div className="flex flex-wrap gap-2">
                      {ds.features.map((feat, i) => (
                        <span key={i} className="px-3 py-1 bg-slate-100 text-slate-600 rounded-full text-xs font-medium">
                          {feat}
                        </span>
                      ))}
                    </div>
                  </div>

                  <a 
                    href={ds.kaggleLink} 
                    target="_blank" 
                    rel="noreferrer"
                    className="inline-flex items-center gap-2 px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-lg text-sm font-medium transition-colors"
                  >
                    Посмотреть источник <ExternalLink className="w-4 h-4" />
                  </a>
                </div>

              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
