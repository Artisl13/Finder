# -*- coding: utf-8 -*-
"""
Обработка данных расчетов радиационной защиты (finder).
Версия 1.7. Изменения относительно 1.6:
- Исправлена работа с finder_settings.json после компиляции в EXE (sys.frozen).
- Убрано непрерывное отображение координат мыши для снижения нагрузки на CPU.
- Добавление точки среза только по клику левой кнопки мыши.
"""
import os
import sys
import re
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# =========================================================
# Файл настроек
# =========================================================
SETTINGS_FILENAME = "finder_settings.json"

def get_settings_path():
    """Файл настроек располагается рядом с исполняемым файлом (для EXE) или скриптом."""
    if getattr(sys, 'frozen', False):
        # Запуск из скомпилированного EXE
        base = os.path.dirname(sys.executable)
    else:
        # Запуск из исходного скрипта
        try:
            base = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base = os.getcwd()
    return os.path.join(base, SETTINGS_FILENAME)

# =========================================================
# Работа с файлами данных
# =========================================================
def detect_encoding(filepath):
    """
    Определяет кодировку файла путем последовательной попытки чтения.
    Возвращает первую успешную кодировку из списка: utf-8, cp1251, koi8-r, latin-1.
    """
    encodings = ['utf-8', 'cp1251', 'koi8-r', 'latin-1']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                f.read()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'utf-8'  # по умолчанию

def parse_header(filepath):
    """
    Читает заголовок файла.
    Возвращает:
    - n_cols: число столбцов (int)
    - col_names: список названий функционалов (ВЕСЬ текст строки)
    """
    encoding = detect_encoding(filepath)
    with open(filepath, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    n_cols = int(lines[1].strip().split()[0])
    col_names = []
    
    for i in range(2, 2 + n_cols):
        # Берем ВЕСЬ текст строки без каких-либо обрезок, split'ов и фильтров
        name = lines[i].strip()
        col_names.append(name)
        
    return n_cols, col_names

def load_data(filepath, n_cols):
    """
    Читает числовые данные из файла.
    Возвращает numpy-массив формы (n_rows, n_cols).
    """
    encoding = detect_encoding(filepath)
    with open(filepath, 'r', encoding=encoding) as f:
        lines = f.readlines()
    header_lines = 2 + n_cols
    data_lines = lines[header_lines:]
    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        if '//' in line:
            line = line.split('//')[0].strip()
        if not line:
            continue
        parts = line.split()
        parts = [p.replace(',', '.') for p in parts]
        try:
            row = [float(x) for x in parts[:n_cols]]
            if len(row) == n_cols:
                rows.append(row)
        except ValueError:
            continue
    return np.array(rows) if rows else np.empty((0, n_cols))

# =========================================================
# Алгоритм перестройки массива
# =========================================================
def transform_array(prob, values):
    """
    Сортируем по values, вычисляем убывающую кумулятивную вероятность.
    """
    prob = np.asarray(prob, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    values_sorted = values[order]
    prob_sorted = prob[order]
    prob_cum = np.cumsum(prob_sorted[::-1])[::-1]
    return values_sorted, prob_cum

# =========================================================
# GUI
# =========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработка данных радиационной защиты")
        self.geometry("1200x900")
        self.work_dir   = tk.StringVar(value="")
        self.file_type  = tk.StringVar(value="DF71.dat")
        self.func_name  = tk.StringVar(value="")
        self.trim_coeff = tk.StringVar(value="0.9")
        self.step_x     = tk.StringVar(value="1")
        self.step_y     = tk.StringVar(value="1")
        self.log_x      = tk.BooleanVar(value=False)
        self.log_y      = tk.BooleanVar(value=False)
        self.func_names = []
        self.dirs = []
        # {dir_name: (values_sorted, prob_cum, x_min, x_max)}
        self.loaded_data = {}
        # списки Line2D объектов для статистических кривых
        self.stat_lines_x = []
        self.stat_lines_y = []
        # результаты расчетов для сохранения в CSV
        self.x_result = None
        self.y_result = None
        # Точки среза: список кортежей (x_given, y_given)
        self.slice_points = []
        self._build_ui()
        self._load_settings()
        # при закрытии окна автоматически сохраняем настройки
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- построение интерфейса -----
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=5, pady=5)

        # строка 0: рабочая директория
        ttk.Label(top, text="Рабочая директория:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.work_dir, width=70).grid(
            row=0, column=1, columnspan=3, sticky=tk.W, padx=5)
        ttk.Button(top, text="Обзор...", command=self._choose_dir).grid(
            row=0, column=4, padx=5)

        # строка 1: тип файла
        ttk.Label(top, text="Тип файла:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_combo = ttk.Combobox(top, textvariable=self.file_type,
                                       values=["DF71.dat", "DF71Kz_pop1.dat"],
                                       state="readonly", width=30)
        self.file_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top, text="Сканировать заголовки",
                   command=self._scan_headers).grid(row=1, column=4, padx=5)

        # строка 2: выбор функционала
        ttk.Label(top, text="Функционал:").grid(row=2, column=0, sticky=tk.W)
        self.func_combo = ttk.Combobox(top, textvariable=self.func_name,
                                       state="readonly", width=50)
        self.func_combo.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5)
        ttk.Button(top, text="Загрузить данные и построить график",
                   command=self._load_and_plot).grid(row=2, column=4, padx=5)

        # строка 3: коэффициент отсева, шаг по X, расчет по X
        ttk.Label(top, text="Коэфф. отсева (0..1):").grid(
            row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(top, textvariable=self.trim_coeff, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=5)
        ttk.Label(top, text="Шаг по X:").grid(row=3, column=2, sticky=tk.E, padx=(20, 5))
        ttk.Entry(top, textvariable=self.step_x, width=6).grid(
            row=3, column=3, sticky=tk.W)
        ttk.Button(top, text="Расчет по X (срезы X -> статистика Y)",
                   command=self._compute_stat_curves_x).grid(
            row=3, column=4, padx=5, sticky=tk.W)

        # строка 4: шаг по Y, расчет по Y
        ttk.Label(top, text="Шаг по Y:").grid(row=4, column=2, sticky=tk.E, padx=(20, 5), pady=5)
        ttk.Entry(top, textvariable=self.step_y, width=6).grid(
            row=4, column=3, sticky=tk.W, pady=5)
        ttk.Button(top, text="Расчет по Y (срезы Y -> статистика X)",
                   command=self._compute_stat_curves_y).grid(
            row=4, column=4, padx=5, sticky=tk.W)

        # строка 5: сохранение CSV
        ttk.Button(top, text="Сохранить X-результат в CSV",
                   command=self._save_x_csv).grid(
            row=5, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        ttk.Button(top, text="Сохранить Y-результат в CSV",
                   command=self._save_y_csv).grid(
            row=5, column=4, sticky=tk.W, padx=5, pady=2)

        # строка 6: галочки логарифмического масштаба
        ttk.Checkbutton(top, text="Log X", variable=self.log_x,
                        command=self._apply_log_scales).grid(
            row=6, column=1, sticky=tk.W, padx=5, pady=2)
        ttk.Checkbutton(top, text="Log Y", variable=self.log_y,
                        command=self._apply_log_scales).grid(
            row=6, column=2, sticky=tk.W, padx=5, pady=2)

        # ----- Панель точек среза -----
        points_frame = ttk.LabelFrame(self, text="Точки среза (клик по графику или ручной ввод)")
        points_frame.pack(fill=tk.X, padx=5, pady=5)

        # Ввод координат
        ttk.Label(points_frame, text="X:").grid(row=0, column=0, padx=5, pady=2)
        self.x_given_entry = ttk.Entry(points_frame, width=15)
        self.x_given_entry.grid(row=0, column=1, padx=5, pady=2)

        ttk.Label(points_frame, text="Y:").grid(row=0, column=2, padx=5, pady=2)
        self.y_given_entry = ttk.Entry(points_frame, width=15)
        self.y_given_entry.grid(row=0, column=3, padx=5, pady=2)

        ttk.Button(points_frame, text="Добавить", command=self._add_point_from_entry).grid(
            row=0, column=4, padx=5, pady=2)
        ttk.Button(points_frame, text="Удалить выбранную",
                   command=self._remove_selected_point).grid(
            row=0, column=5, padx=5, pady=2)
        ttk.Button(points_frame, text="Очистить все",
                   command=self._clear_all_points).grid(
            row=0, column=6, padx=5, pady=2)

        # Listbox с точками
        self.points_listbox = tk.Listbox(points_frame, height=4, width=60)
        self.points_listbox.grid(row=1, column=0, columnspan=4, padx=5, pady=2, sticky=tk.W)

        # Кнопки выгрузки срезов
        ttk.Button(points_frame, text="Выгрузка среза по заданному X",
                   command=self._export_x_slice).grid(
            row=1, column=4, padx=5, pady=2, sticky=tk.W)
        ttk.Button(points_frame, text="Выгрузка среза по заданному Y",
                   command=self._export_y_slice).grid(
            row=1, column=5, padx=5, pady=2, sticky=tk.W)

        # область графика
        self.fig = plt.Figure(figsize=(11, 6.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas.get_tk_widget())
        toolbar.update()

        # обработчик клика по графику (только левая кнопка для добавления точки среза)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        
        # отключаем непрерывное отображение координат в статус-баре matplotlib для снижения нагрузки на CPU
        # координаты отображаются только после клика по графику
        self.ax.format_coord = lambda x, y: ""

        # ----- Прогресс-бар -----
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(fill=tk.X, padx=5, pady=2)
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack(padx=5, pady=2)
        self.progress_frame.pack_forget()  # Скрываем пока не нужен

        # статус
        self.status = tk.StringVar(value="Готово.")
        ttk.Label(self, textvariable=self.status,
                  relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # =========================================================
    # НАСТРОЙКИ: сохранение / загрузка
    # =========================================================
    def _load_settings(self):
        """Восстановить конфигурацию из finder_settings.json."""
        data = {}
        path = get_settings_path()
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                data = {}
        self.work_dir.set(str(data.get("work_dir", "")))
        ftype = str(data.get("file_type", "DF71.dat"))
        if ftype in self.file_combo["values"]:
            self.file_type.set(ftype)
        else:
            self.file_type.set("DF71.dat")
        self.func_name.set(str(data.get("func_name", "")))
        self.trim_coeff.set(str(data.get("trim_coeff", "0.9")))
        self.step_x.set(str(data.get("step_x", "1")))
        self.step_y.set(str(data.get("step_y", "1")))
        self.log_x.set(bool(data.get("log_x", False)))
        self.log_y.set(bool(data.get("log_y", False)))
        # Загрузка точек среза
        saved_points = data.get("slice_points", [])
        self.slice_points = []
        for p in saved_points:
            try:
                self.slice_points.append((float(p[0]), float(p[1])))
            except (ValueError, IndexError, TypeError):
                pass
        self._update_points_listbox()

    def _save_settings(self):
        """Сохранить текущую конфигурацию в finder_settings.json."""
        data = {
            "work_dir":   self.work_dir.get().strip(),
            "file_type":  self.file_type.get(),
            "func_name":  self.func_name.get(),
            "trim_coeff": self.trim_coeff.get().strip(),
            "step_x":     self.step_x.get().strip(),
            "step_y":     self.step_y.get().strip(),
            "log_x":      self.log_x.get(),
            "log_y":      self.log_y.get(),
            "slice_points": [[p[0], p[1]] for p in self.slice_points],
        }
        try:
            with open(get_settings_path(), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Не удалось сохранить настройки: {e}")

    def _on_close(self):
        """Обработчик закрытия окна: сохранить настройки и выйти."""
        self._save_settings()
        self.destroy()

    # ----- выбор директории -----
    def _choose_dir(self):
        d = filedialog.askdirectory(title="Выберите рабочую директорию")
        if d:
            self.work_dir.set(d)
            self.status.set(f"Выбрана директория: {d}")
            self._save_settings()

    # ----- сканирование заголовков -----
    def _scan_headers(self):
        work = self.work_dir.get().strip()
        if not work or not os.path.isdir(work):
            messagebox.showerror("Ошибка", "Укажите корректную рабочую директорию.")
            return
        subdirs = []
        for name in os.listdir(work):
            full = os.path.join(work, name)
            if os.path.isdir(full) and re.match(r'^\d+$', name):
                subdirs.append(name)
        subdirs.sort()
        self.dirs = subdirs
        if not subdirs:
            messagebox.showerror(
                "Ошибка",
                "В рабочей директории не найдено поддиректорий с числовыми именами.")
            return
        ftype = self.file_type.get()
        first_header = None
        for d in subdirs:
            d_path = os.path.join(work, d)
            for target in os.listdir(d_path):
                target_path = os.path.join(d_path, target)
                if os.path.isdir(target_path):
                    fpath = os.path.join(target_path, ftype)
                    if os.path.isfile(fpath):
                        first_header = fpath
                        break
            if first_header:
                break
        if first_header is None:
            messagebox.showerror("Ошибка", f"Файлы типа '{ftype}' не найдены.")
            return
        try:
            n_cols, col_names = parse_header(first_header)
        except Exception as e:
            messagebox.showerror("Ошибка чтения заголовка", str(e))
            return
        self.func_names = col_names[1:]

        """         
        self.func_combo['values'] = self.func_names
        if self.func_names:
            saved = self.func_name.get()
            if saved in self.func_names:
                self.func_combo.current(self.func_names.index(saved))
            else:
                self.func_combo.current(0) 
        """
        # Создаем отображаемые имена: заменяем запятую для корректного отображения
        self.display_names = [name.replace(',','_') for name in self.func_names]  # ‚ - одинарная кавычка-запятая
    
        self.func_combo['values'] = self.display_names
        if self.func_names:
            saved = self.func_name.get()
            if saved in self.func_names:
                idx = self.func_names.index(saved)
                self.func_combo.current(idx)
            else:
                self.func_combo.current(0)

        self.status.set(
            f"Найдено директорий: {len(subdirs)}. "
            f"Столбцов в файле: {n_cols}. Функционалов: {len(self.func_names)}.")
        self._save_settings()

    # ----- загрузка данных и построение исходных кривых -----
    def _load_and_plot(self):
        work = self.work_dir.get().strip()
        if not work or not self.dirs:
            messagebox.showerror("Ошибка", "Сначала выполните сканирование заголовков.")
            return
        # fname = self.func_name.get()
        selection_index = self.func_combo.current()
        if selection_index == -1:
            messagebox.showerror("Ошибка", "Выберите функционал.")
            return
        fname = self.func_names[selection_index]  # Оригинальное имя из списка данных
        if not fname:
            messagebox.showerror("Ошибка", "Выберите функционал.")
            return
        if fname not in self.func_names:
            messagebox.showerror("Ошибка", f"Функционал '{fname}' не найден в списке.")
            return
        ftype = self.file_type.get()
        self.loaded_data.clear()
        self._clear_stat_lines_x()
        self._clear_stat_lines_y()
        self.x_result = None
        self.y_result = None

        # найти первый попавшийся файл для чтения заголовка
        first_header = None
        for d in self.dirs:
            d_path = os.path.join(work, d)
            for target in os.listdir(d_path):
                target_path = os.path.join(d_path, target)
                if os.path.isdir(target_path):
                    fpath = os.path.join(target_path, ftype)
                    if os.path.isfile(fpath):
                        first_header = fpath
                        break
            if first_header:
                break
        if first_header is None:
            messagebox.showerror("Ошибка", f"Файлы типа '{ftype}' не найдены.")
            return
        try:
            n_cols, col_names = parse_header(first_header)
        except Exception as e:
            messagebox.showerror("Ошибка чтения заголовка", str(e))
            return
        try:
            col_idx = col_names.index(fname)
        except ValueError:
            messagebox.showerror("Ошибка", f"Не удалось найти столбец '{fname}'.")
            return

        # ----- Показываем прогресс-бар -----
        total_dirs = len(self.dirs)
        self.progress_bar['maximum'] = total_dirs
        self.progress_bar['value'] = 0
        self.progress_frame.pack(fill=tk.X, padx=5, pady=2, before=self.canvas.get_tk_widget())
        self.progress_label.config(text="Загрузка данных...")
        self.update_idletasks()

        loaded_count = 0
        for idx, d in enumerate(self.dirs):
            d_path = os.path.join(work, d)
            fpath = None
            for target in os.listdir(d_path):
                target_path = os.path.join(d_path, target)
                if os.path.isdir(target_path):
                    candidate = os.path.join(target_path, ftype)
                    if os.path.isfile(candidate):
                        fpath = candidate
                        break
            if fpath is None:
                self.progress_bar['value'] = idx + 1
                self.progress_label.config(text=f"Пропущено: {d}")
                self.update_idletasks()
                continue
            try:
                data = load_data(fpath, n_cols)
            except Exception as e:
                print(f"Ошибка чтения {fpath}: {e}")
                self.progress_bar['value'] = idx + 1
                self.progress_label.config(text=f"Ошибка: {d}")
                self.update_idletasks()
                continue
            if data.size == 0:
                self.progress_bar['value'] = idx + 1
                self.progress_label.config(text=f"Пустой файл: {d}")
                self.update_idletasks()
                continue
            prob = data[:, 0]
            values = data[:, col_idx]
            vs, pc = transform_array(prob, values)
            self.loaded_data[d] = (vs, pc, float(vs.min()), float(vs.max()))
            loaded_count += 1

            # Обновляем прогресс-бар
            self.progress_bar['value'] = idx + 1
            self.progress_label.config(text=f"Загружено: {d} ({loaded_count}/{total_dirs})")
            self.update_idletasks()

        # Скрываем прогресс-бар после загрузки
        self.progress_frame.pack_forget()
        self.update_idletasks()

        if not self.loaded_data:
            messagebox.showwarning(
                "Внимание", "Не удалось загрузить данные ни из одной директории.")
            return

        self.ax.clear()
        for d in sorted(self.loaded_data.keys()):
            vs, pc, _, _ = self.loaded_data[d]
            self.ax.plot(vs, pc, color='dodgerblue', alpha=0.4, linewidth=0.8)
        self.ax.set_xlabel(fname)
        self.ax.set_ylabel("Вероятность (кумулятивная убывающая)")
        self.ax.set_title(f"Кривые по директориям. Функционал: {fname}")
        self.ax.grid(True, which='both', ls=':', alpha=0.5)
        
        # ИСПРАВЛЕНИЕ: сначала tight_layout, потом subplots_adjust
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.186)
        
        self._apply_log_scales()
        self.canvas.draw()
        self.status.set(f"Загружено директорий: {len(self.loaded_data)}.")
        self._save_settings()

    # ----- очистка статистических линий -----
    def _clear_stat_lines_x(self):
        for line in self.stat_lines_x:
            try:
                line.remove()
            except Exception:
                pass
        self.stat_lines_x.clear()

    def _clear_stat_lines_y(self):
        for line in self.stat_lines_y:
            try:
                line.remove()
            except Exception:
                pass
        self.stat_lines_y.clear()

    # ----- применение логарифмического масштаба -----
    def _apply_log_scales(self):
        """Применяет логарифмический масштаб к осям на основе состояния чекбоксов."""
        try:
            self.ax.set_xscale('log' if self.log_x.get() else 'linear')
            self.ax.set_yscale('log' if self.log_y.get() else 'linear')
            # ИСПРАВЛЕНИЕ: сначала tight_layout, потом subplots_adjust
            self.fig.tight_layout()
            self.fig.subplots_adjust(bottom=0.186)
            self.canvas.draw()
        except Exception:
            pass

    # ----- валидация параметров -----
    def _validate_common_params(self):
        """Возвращает (k, True) или (None, False)."""
        if not self.loaded_data:
            messagebox.showwarning(
                "Внимание",
                "Сначала загрузите данные (кнопка 'Загрузить данные...').")
            return None, False
        try:
            k = float(self.trim_coeff.get().strip())
            if not (0.0 <= k <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Ошибка",
                "Коэффициент отсева должен быть числом от 0 до 1 (напр., 0.9).")
            return None, False
        return k, True

    # =========================================================
    # РАСЧЕТ ПО X: срезы по X -> статистика по Y
    # =========================================================
    def _compute_stat_curves_x(self):
        k, ok = self._validate_common_params()
        if not ok:
            return
        try:
            step_x = int(self.step_x.get().strip())
            if step_x < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Шаг по X должен быть целым числом >= 1.")
            return
        self._clear_stat_lines_x()
        self.x_result = None
        keys = sorted(self.loaded_data.keys())
        N = len(keys)
        all_vs = []
        x_min_arr = np.zeros(N)
        x_max_arr = np.zeros(N)
        for i, d in enumerate(keys):
            vs, pc, xmin, xmax = self.loaded_data[d]
            all_vs.append(vs)
            x_min_arr[i] = xmin
            x_max_arr[i] = xmax
        # общая сетка X
        x_grid = np.unique(np.concatenate(all_vs))
        if x_grid.size == 0:
            messagebox.showwarning("Внимание", "Не удалось построить сетку X.")
            return
        if step_x > 1:
            x_grid = x_grid[::step_x]
        # интерполяция всех кривых на общую сетку X
        y_matrix = np.zeros((N, x_grid.size))
        for i, d in enumerate(keys):
            vs, pc, _, _ = self.loaded_data[d]
            y_matrix[i, :] = np.interp(x_grid, vs, pc, left=pc[0], right=pc[-1])
        # маска валидности
        valid = (x_grid[None, :] >= x_min_arr[:, None]) & \
                (x_grid[None, :] <= x_max_arr[:, None])
        y_max       = np.full(x_grid.size, np.nan)
        y_min       = np.full(x_grid.size, np.nan)
        y_max_trim  = np.full(x_grid.size, np.nan)
        y_min_trim  = np.full(x_grid.size, np.nan)
        y_mean_trim = np.full(x_grid.size, np.nan)
        for j in range(x_grid.size):
            mask = valid[:, j]
            y_valid = y_matrix[mask, j]
            n_eff = y_valid.size
            if n_eff == 0:
                continue
            y_sorted = np.sort(y_valid)
            y_max[j] = y_sorted[-1]
            y_min[j] = y_sorted[0]
            trim_count = int(np.floor(n_eff * (1.0 - k) / 2.0))
            if trim_count > 0 and n_eff - 2 * trim_count > 0:
                y_trimmed = y_sorted[trim_count: n_eff - trim_count]
                y_max_trim[j]  = y_trimmed[-1]
                y_min_trim[j]  = y_trimmed[0]
                y_mean_trim[j] = np.mean(y_trimmed)
            else:
                y_max_trim[j]  = y_sorted[-1]
                y_min_trim[j]  = y_sorted[0]
                y_mean_trim[j] = np.mean(y_sorted)
        # отрисовка статистических кривых
        l1, = self.ax.plot(x_grid, y_max, color='red', linewidth=2.0,
                           label='X->Y: Max')
        l2, = self.ax.plot(x_grid, y_min, color='blue', linewidth=2.0,
                           label='X->Y: Min')
        l3, = self.ax.plot(x_grid, y_max_trim, color='darkorange', linewidth=2.0,
                           linestyle='--', label=f'X->Y: Max-trim (k={k})')
        l4, = self.ax.plot(x_grid, y_min_trim, color='green', linewidth=2.0,
                           linestyle='--', label=f'X->Y: Min+trim (k={k})')
        l5, = self.ax.plot(x_grid, y_mean_trim, color='purple', linewidth=2.5,
                           linestyle='-.', label=f'X->Y: Mean trimmed (k={k})')
        self.stat_lines_x.extend([l1, l2, l3, l4, l5])
        self.ax.legend(fontsize='small', loc='best')
        
        # ИСПРАВЛЕНИЕ: сначала tight_layout, потом subplots_adjust
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.186)
        
        self._apply_log_scales()
        self.canvas.draw()
        self.x_result = (x_grid.copy(), y_max.copy(), y_min.copy(),
                         y_max_trim.copy(), y_min_trim.copy(), y_mean_trim.copy())
        self.status.set(
            f"Расчет по X выполнен. Кривых: {N}, шаг X: {step_x}, "
            f"точек сетки: {x_grid.size}, коэфф. k={k}.")
        self._save_settings()

    # =========================================================
    # РАСЧЕТ ПО Y: срезы по Y -> статистика по X
    # =========================================================
    def _compute_stat_curves_y(self):
        k, ok = self._validate_common_params()
        if not ok:
            return
        try:
            step_y = int(self.step_y.get().strip())
            if step_y < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Шаг по Y должен быть целым числом >= 1.")
            return
        self._clear_stat_lines_y()
        self.y_result = None
        keys = sorted(self.loaded_data.keys())
        N = len(keys)
        all_pc = []
        y_min_arr = np.zeros(N)
        y_max_arr = np.zeros(N)
        for i, d in enumerate(keys):
            vs, pc, _, _ = self.loaded_data[d]
            all_pc.append(pc)
            y_min_arr[i] = pc.min()
            y_max_arr[i] = pc.max()
        # общая сетка Y (вероятности)
        y_grid = np.unique(np.concatenate(all_pc))
        if y_grid.size == 0:
            messagebox.showwarning("Внимание", "Не удалось построить сетку Y.")
            return
        if step_y > 1:
            y_grid = y_grid[::step_y]
        # интерполяция X по Y (pc убывающая -> переворачиваем для np.interp)
        x_matrix = np.zeros((N, y_grid.size))
        for i, d in enumerate(keys):
            vs, pc, _, _ = self.loaded_data[d]
            x_matrix[i, :] = np.interp(y_grid, pc[::-1], vs[::-1],
                                       left=vs[0], right=vs[-1])
        valid = (y_grid[None, :] >= y_min_arr[:, None]) & \
                (y_grid[None, :] <= y_max_arr[:, None])
        x_max       = np.full(y_grid.size, np.nan)
        x_min       = np.full(y_grid.size, np.nan)
        x_max_trim  = np.full(y_grid.size, np.nan)
        x_min_trim  = np.full(y_grid.size, np.nan)
        x_mean_trim = np.full(y_grid.size, np.nan)
        for j in range(y_grid.size):
            mask = valid[:, j]
            x_valid = x_matrix[mask, j]
            n_eff = x_valid.size
            if n_eff == 0:
                continue
            x_sorted = np.sort(x_valid)
            x_max[j] = x_sorted[-1]
            x_min[j] = x_sorted[0]
            trim_count = int(np.floor(n_eff * (1.0 - k) / 2.0))
            if trim_count > 0 and n_eff - 2 * trim_count > 0:
                x_trimmed = x_sorted[trim_count: n_eff - trim_count]
                x_max_trim[j]  = x_trimmed[-1]
                x_min_trim[j]  = x_trimmed[0]
                x_mean_trim[j] = np.mean(x_trimmed)
            else:
                x_max_trim[j]  = x_sorted[-1]
                x_min_trim[j]  = x_sorted[0]
                x_mean_trim[j] = np.mean(x_sorted)
        mksz = 4
        l1, = self.ax.plot(x_max, y_grid, color='red', linewidth=1.5,
                           marker='o', markersize=mksz, label='Y->X: Max')
        l2, = self.ax.plot(x_min, y_grid, color='blue', linewidth=1.5,
                           marker='s', markersize=mksz, label='Y->X: Min')
        l3, = self.ax.plot(x_max_trim, y_grid, color='darkorange', linewidth=1.5,
                           linestyle='--', marker='^', markersize=mksz,
                           label=f'Y->X: Max-trim (k={k})')
        l4, = self.ax.plot(x_min_trim, y_grid, color='green', linewidth=1.5,
                           linestyle='--', marker='v', markersize=mksz,
                           label=f'Y->X: Min+trim (k={k})')
        l5, = self.ax.plot(x_mean_trim, y_grid, color='purple', linewidth=2.0,
                           linestyle='-.', marker='D', markersize=mksz,
                           label=f'Y->X: Mean trimmed (k={k})')
        self.stat_lines_y.extend([l1, l2, l3, l4, l5])
        self.ax.legend(fontsize='small', loc='best')
        
        # ИСПРАВЛЕНИЕ: сначала tight_layout, потом subplots_adjust
        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.186)
        
        self._apply_log_scales()
        self.canvas.draw()
        self.y_result = (y_grid.copy(), x_max.copy(), x_min.copy(),
                         x_max_trim.copy(), x_min_trim.copy(), x_mean_trim.copy())
        self.status.set(
            f"Расчет по Y выполнен. Кривых: {N}, шаг Y: {step_y}, "
            f"точек сетки: {y_grid.size}, коэфф. k={k}.")
        self._save_settings()

    # =========================================================
    # ПАНЕЛЬ ТОЧЕК СРЕЗА
    # =========================================================
    def _on_canvas_click(self, event):
        """Обработчик клика по графику: добавляет точку в список среза (только левая кнопка)."""
        # Реагируем только на левую кнопку мыши (button == 1)
        if event.button != 1:
            return
        if event.inaxes != self.ax:
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        self.slice_points.append((float(x), float(y)))
        self._update_points_listbox()
        self._save_settings()

    def _add_point_from_entry(self):
        """Добавляет точку из полей ручного ввода."""
        try:
            x = float(self.x_given_entry.get().strip().replace(',', '.'))
            y = float(self.y_given_entry.get().strip().replace(',', '.'))
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения X и Y.")
            return
        self.slice_points.append((x, y))
        self._update_points_listbox()
        self._save_settings()

    def _remove_selected_point(self):
        """Удаляет выбранную точку из списка."""
        sel = self.points_listbox.curselection()
        if not sel:
            messagebox.showwarning("Внимание", "Выберите точку для удаления.")
            return
        idx = sel[0]
        if 0 <= idx < len(self.slice_points):
            self.slice_points.pop(idx)
            self._update_points_listbox()
            self._save_settings()

    def _clear_all_points(self):
        """Очищает все точки среза."""
        if not self.slice_points:
            return
        if messagebox.askyesno("Подтверждение", "Удалить все точки среза?"):
            self.slice_points.clear()
            self._update_points_listbox()
            self._save_settings()

    def _update_points_listbox(self):
        """Обновляет отображение списка точек."""
        self.points_listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(self.slice_points):
            self.points_listbox.insert(tk.END, f"#{i+1}: X={x:.6g}, Y={y:.6g}")

    # =========================================================
    # ВЫГРУЗКА СРЕЗОВ
    # =========================================================
    def _compute_trim_stats(self, values, k):
        """
        Вычисляет min, min_trim, mean_trim, max_trim, max для массива значений.
        Возвращает кортеж (v_min, v_min_trim, v_mean_trim, v_max_trim, v_max).
        """
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return (np.nan, np.nan, np.nan, np.nan, np.nan)
        sorted_arr = np.sort(valid)
        v_min = sorted_arr[0]
        v_max = sorted_arr[-1]
        n = sorted_arr.size
        trim_count = int(np.floor(n * (1.0 - k) / 2.0))
        if trim_count > 0 and n - 2 * trim_count > 0:
            trimmed = sorted_arr[trim_count:n - trim_count]
            v_min_trim = trimmed[0]
            v_max_trim = trimmed[-1]
            v_mean_trim = np.mean(trimmed)
        else:
            v_min_trim = v_min
            v_max_trim = v_max
            v_mean_trim = np.mean(sorted_arr)
        return (v_min, v_min_trim, v_mean_trim, v_max_trim, v_max)

    def _export_x_slice(self):
        """
        Выгрузка среза по заданному X.
        Для каждой точки (X_given, Y_given) вычисляет Y-значения всех кривых при X=X_given.
        Формат CSV: X_given;y_min;y_min_trim;y_mean_trim;y_max_trim;y_max;y_1;y_2;...;y_N
        """
        if not self.slice_points:
            messagebox.showwarning("Внимание", "Список точек среза пуст.")
            return
        if not self.loaded_data:
            messagebox.showwarning("Внимание", "Сначала загрузите данные.")
            return
        try:
            k = float(self.trim_coeff.get().strip())
            if not (0.0 <= k <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный коэффициент отсева.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Сохранить срез по X",
            initialfile="X_slice.csv",
        )
        if not path:
            return

        keys = sorted(self.loaded_data.keys())
        N = len(keys)

        rows = []
        for x_given, _ in self.slice_points:
            y_values = []
            for d in keys:
                vs, pc, xmin, xmax = self.loaded_data[d]
                if xmin <= x_given <= xmax:
                    y_val = float(np.interp(x_given, vs, pc))
                else:
                    y_val = np.nan
                y_values.append(y_val)
            v_min, v_min_trim, v_mean_trim, v_max_trim, v_max = self._compute_trim_stats(y_values, k)
            row = [x_given, v_min, v_min_trim, v_mean_trim, v_max_trim, v_max] + y_values
            rows.append(row)

        col_names = ["X_given", "y_min", "y_min_trim", "y_mean_trim", "y_max_trim", "y_max"]
        col_names += [f"y_{i+1}" for i in range(N)]

        try:
            self._write_csv(path, col_names, rows)
            self.status.set(f"Срез по X сохранен: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def _export_y_slice(self):
        """
        Выгрузка среза по заданному Y.
        Для каждой точки (X_given, Y_given) вычисляет X-значения всех кривых при Y=Y_given.
        Формат CSV: Y_given;x_min;x_min_trim;x_mean_trim;x_max_trim;x_max;x_1;x_2;...;x_N
        """
        if not self.slice_points:
            messagebox.showwarning("Внимание", "Список точек среза пуст.")
            return
        if not self.loaded_data:
            messagebox.showwarning("Внимание", "Сначала загрузите данные.")
            return
        try:
            k = float(self.trim_coeff.get().strip())
            if not (0.0 <= k <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректный коэффициент отсева.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Сохранить срез по Y",
            initialfile="Y_slice.csv",
        )
        if not path:
            return

        keys = sorted(self.loaded_data.keys())
        N = len(keys)

        rows = []
        for _, y_given in self.slice_points:
            x_values = []
            for d in keys:
                vs, pc, xmin, xmax = self.loaded_data[d]
                pc_min = pc.min()
                pc_max = pc.max()
                if pc_min <= y_given <= pc_max:
                    x_val = float(np.interp(y_given, pc[::-1], vs[::-1]))
                else:
                    x_val = np.nan
                x_values.append(x_val)
            v_min, v_min_trim, v_mean_trim, v_max_trim, v_max = self._compute_trim_stats(x_values, k)
            row = [y_given, v_min, v_min_trim, v_mean_trim, v_max_trim, v_max] + x_values
            rows.append(row)

        col_names = ["Y_given", "x_min", "x_min_trim", "x_mean_trim", "x_max_trim", "x_max"]
        col_names += [f"x_{i+1}" for i in range(N)]

        try:
            self._write_csv(path, col_names, rows)
            self.status.set(f"Срез по Y сохранен: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    # =========================================================
    # СОХРАНЕНИЕ В CSV  (разделитель ';', десятичный разделитель ',')
    # =========================================================
    def _write_csv(self, path, column_names, data):
        """
        Запись CSV в формате:
          - разделитель полей: ';'
          - десятичный разделитель: ','
          - NaN записываются как пустые ячейки
          - кодировка UTF-8 с BOM (корректное открытие в Excel)
        """
        with open(path, 'w', encoding='utf-8-sig', newline='') as f:
            f.write(';'.join(column_names) + '\n')
            for row in data:
                cells = []
                for v in row:
                    if isinstance(v, float) and np.isnan(v):
                        cells.append('')
                    else:
                        cells.append(format(v, '.10g').replace('.', ','))
                f.write(';'.join(cells) + '\n')

    def _save_x_csv(self):
        if self.x_result is None:
            messagebox.showwarning(
                "Внимание",
                "Сначала выполните расчет по X (кнопка 'Расчет по X...').")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Сохранить X-результат",
            initialfile="X_result.csv",
        )
        if not path:
            return
        x_grid, y_max, y_min, y_max_trim, y_min_trim, y_mean_trim = self.x_result
        # Порядок: x;y_min;y_min_trim;y_mean_trim;y_max_trim;y_max (ascending)
        data = np.column_stack([x_grid, y_min, y_min_trim, y_mean_trim, y_max_trim, y_max])
        cols = ["x", "y_min", "y_min_trim", "y_mean_trim", "y_max_trim", "y_max"]
        try:
            self._write_csv(path, cols, data)
            self.status.set(f"X-результат сохранен: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def _save_y_csv(self):
        if self.y_result is None:
            messagebox.showwarning(
                "Внимание",
                "Сначала выполните расчет по Y (кнопка 'Расчет по Y...').")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Сохранить Y-результат",
            initialfile="Y_result.csv",
        )
        if not path:
            return
        y_grid, x_max, x_min, x_max_trim, x_min_trim, x_mean_trim = self.y_result
        # Порядок: y;x_min;x_min_trim;x_mean_trim;x_max_trim;x_max (ascending)
        data = np.column_stack([y_grid, x_min, x_min_trim, x_mean_trim, x_max_trim, x_max])
        cols = ["y", "x_min", "x_min_trim", "x_mean_trim", "x_max_trim", "x_max"]
        try:
            self._write_csv(path, cols, data)
            self.status.set(f"Y-результат сохранен: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

# =========================================================
# Запуск
# =========================================================
if __name__ == "__main__":
    app = App()
    app.mainloop()