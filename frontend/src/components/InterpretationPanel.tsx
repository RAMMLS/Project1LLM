'use client';

import React from 'react';
import { Lightbulb, AlertTriangle, ShieldCheck } from 'lucide-react';

interface InterpretationPanelProps {
  activeMode: string;
}

export default function InterpretationPanel({ activeMode }: InterpretationPanelProps) {
  
  const getInsights = () => {
    switch(activeMode) {
      case 'vision':
        return {
          title: 'Анализ подделки документов',
          text: 'Модель опирается на микротекстурные изменения (растекание чернил, зернистость бумаги). Grad-CAM показывает высокую активацию вокруг подписей.',
          logic: 'Как это работает: Предобученная на миллионах картинок нейросеть (ResNet50) "замораживается". Мы заменяем только её последний слой, чтобы она научилась отличать настоящие документы от поддельных по текстуре чернил, игнорируя сам текст.',
          riskLevel: 'Высокий',
          action: 'Требуется проверка человеком ложноположительных результатов на высококачественных отпечатках.'
        };
      case 'spectrum':
        return {
          title: 'Идентификация химических групп',
          text: '1D-CNN фокусируется на спектральной полосе 1700-1750 см⁻¹. Сильный признак валентных колебаний карбонильной группы (C=O).',
          logic: 'Как это работает: Вместо плоских картинок мы подаем на вход одномерный график (спектр). Сверточная сеть (1D-CNN) скользит по этому графику "окном" (kernel) и ищет специфичные пики и впадины, которые уникальны для каждого химического вещества.',
          riskLevel: 'Средний',
          action: 'Скорректируйте kernel_size, если узкие пики сглаживаются.'
        };
      case 'math':
        return {
          title: 'Моделирование экономического ROI',
          text: 'Длина последовательности RNN захватывает 30-дневные зависимости. Всплески волатильности сильно влияют на итоговый прогноз.',
          logic: 'Как это работает: Модель получает данные не разово, а как временную шкалу (серию дней). Рекуррентная сеть (RNN) обладает "памятью" — она помнит, что было вчера, чтобы предсказать, что будет завтра, отделяя случайный шум от реальных трендов рынка.',
          riskLevel: 'Низкий',
          action: 'Следите за затуханием градиента. Рассмотрите возможность смены активации на ReLU, если обучение остановится.'
        };
      default:
        return { title: '', text: '', logic: '', riskLevel: '', action: '' };
    }
  };

  const insights = getInsights();

  return (
    <div className="bg-gradient-to-br from-indigo-50 to-blue-50 rounded-2xl border border-blue-100 overflow-hidden shadow-sm">
      <div className="p-5 border-b border-blue-100/50 flex items-center gap-2">
        <Lightbulb className="w-5 h-5 text-blue-600" />
        <h3 className="font-semibold text-blue-900">Оценка эксперта</h3>
      </div>
      
      <div className="p-5 space-y-4">
        <div>
          <h4 className="text-sm font-semibold text-blue-900 mb-1">{insights.title}</h4>
          <p className="text-sm text-blue-800/80 leading-relaxed mb-3">
            {insights.text}
          </p>
          <div className="bg-blue-100/50 p-3 rounded-lg border border-blue-200/50 text-xs text-blue-900/90 leading-relaxed">
            <span className="font-semibold">💡 Принцип работы:</span><br/>
            {insights.logic}
          </div>
        </div>

        <div className="bg-white/60 rounded-xl p-3 text-sm space-y-2">
          <div className="flex items-center gap-2">
            <span className="text-slate-500 font-medium">Уровень риска:</span>
            {insights.riskLevel === 'Высокий' ? (
              <span className="flex items-center gap-1 text-rose-600 font-semibold"><AlertTriangle className="w-4 h-4"/> Высокий</span>
            ) : insights.riskLevel === 'Средний' ? (
              <span className="flex items-center gap-1 text-amber-600 font-semibold"><AlertTriangle className="w-4 h-4"/> Средний</span>
            ) : (
              <span className="flex items-center gap-1 text-emerald-600 font-semibold"><ShieldCheck className="w-4 h-4"/> Низкий</span>
            )}
          </div>
          
          <div className="border-t border-blue-100 pt-2">
            <span className="text-slate-500 font-medium block mb-1">Рекомендуемое действие:</span>
            <span className="text-slate-700">{insights.action}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
