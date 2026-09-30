# -- coding: utf-8 --
"""
Обработка данных расчетов радиационной защиты (finder).
Версия 2.0.5. Возврат к стабильной логике из 2.0.1.
Убраны все "оптимизации" 2.0.3/2.0.4, которые ломали работу:
- unbind('<Motion>')
- format_coord = ""
- throttle через _schedule_redraw
- многопоточная загрузка на Этапе 3 (matplotlib не потокобезопасен)
Оставлено: time.sleep(0) в load_stage1_file — этого достаточно
для отзывчивости UI при загрузке больших файлов.
"""
import os
import sys
import re
import json
import time
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# =========================================================
# Настройка шрифтов matplotlib для кириллицы
# =========================================================
plt.rcParams['font.family'] = 'Segoe UI'
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================
# Базовые пути и настройки
# =========================================================
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        BASE_DIR = os.getcwd()

SETTINGS_FILENAME = "finder_settings.json"
SETTINGS_PATH = os.path.join(BASE_DIR, SETTINGS_FILENAME)

def load_settings():
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_settings(data):
    try:
        with open(SETTINGS_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Ошибка сохранения настроек: {e}")

# =========================================================
# Работа с исходными файлами (DF71)
# =========================================================
def detect_encoding(filepath):
    encodings = ['utf-8', 'cp1251', 'koi8-r', 'latin-1']
    for enc in encodings:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                f.read()
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'utf-8'

def parse_header(filepath):
    encoding = detect_encoding(filepath)
    with open(filepath, 'r', encoding=encoding) as f:
        lines = f.readlines()
    n_cols = int(lines[1].strip().split()[0])
    col_names = []
    for i in range(2, 2 + n_cols):
        name = lines[i].strip()
        col_names.append(name)
    return n_cols, col_names

def load_data(filepath, n_cols):
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

def transform_array(prob, values):
    prob = np.asarray(prob, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    values_sorted = values[order]
    prob_sorted = prob[order]
    prob_cum = np.cumsum(prob_sorted[::-1])[::-1]
    return values_sorted, prob_cum

def load_stage1_file(filepath, progress_callback=None):
    """
    Загружает агрегированный файл из Этапа 1.
    Каждые 10000 строк отпускает GIL через time.sleep(0) —
    этого достаточно для отзывчивости UI.
    """
    curves = {}
    current_dir = None
    x_data = []
    y_data = []
    line_count = 0
    GIL_RELEASE_INTERVAL = 10000
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            if line_count % GIL_RELEASE_INTERVAL == 0:
                time.sleep(0)
                if progress_callback:
                    progress_callback(line_count, current_dir)
            
            line = line.strip()
            if line.startswith('# DIR:'):
                if current_dir and x_data:
                    curves[current_dir] = (np.array(x_data), np.array(y_data))
                current_dir = line[6:].strip()
                x_data = []
                y_data = []
            elif line.startswith('#'):
                continue
            elif line and current_dir:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        x_data.append(float(parts[0]))
                        y_data.append(float(parts[1]))
                    except ValueError:
                        continue
    
    if current_dir and x_data:
        curves[current_dir] = (np.array(x_data), np.array(y_data))
    
    return curves

def load_stage2_csv(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(';')
            row = []
            for p in parts:
                p = p.strip().replace(',', '.')
                if not p:
                    row.append(np.nan)
                else:
                    try:
                        row.append(float(p))
                    except ValueError:
                        row.append(np.nan)
            data.append(row)
    
    if not data:
        return None
    
    arr = np.array(data)
    return (arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5])

def write_csv(path, column_names, data):
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

def compute_trim_stats(values, k):
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

# =========================================================
# GUI Приложение
# =========================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработка данных радиационной защиты (v2.0.5)")
        self.geometry("1200x900")
        
        self.settings = load_settings()
        
        self.work_dir = tk.StringVar(value=self.settings.get("work_dir", ""))
        self.file_type = tk.StringVar(value=self.settings.get("file_type", "DF71.dat"))
        self.func_name = tk.StringVar(value=self.settings.get("func_name", ""))
        self.trim_coeff = tk.StringVar(value=self.settings.get("trim_coeff", "0.9"))
        self.step_x = tk.StringVar(value=self.settings.get("step_x", "1"))
        self.step_y = tk.StringVar(value=self.settings.get("step_y", "1"))
        self.log_x = tk.BooleanVar(value=self.settings.get("log_x", False))
        self.log_y = tk.BooleanVar(value=self.settings.get("log_y", False))
        
        self.func_names = []
        self.display_names = []
        self.dirs = []
        
        self.raw_curves = {}
        
        self.stage3_curves = {}
        self.stage3_stat_x = None
        self.stage3_stat_y = None
        self.slice_points = []
        self._slice_point_artists = []
        
        self.processing = False
        self.stage2_processing = False
        self.stage2_stat_x_done = False
        self.stage2_stat_y_done = False
        
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text="1. Сниферство и агрегация")
        self.notebook.add(self.tab2, text="2. Расчет статистик")
        self.notebook.add(self.tab3, text="3. Визуализация и срез")
        self.notebook.add(self.tab4, text="4. Сохранение среза")
        
        self._build_tab1()
        self._build_tab2()
        self._build_tab3()
        self._build_tab4()
        
        self.status = tk.StringVar(value="Готово.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # ---------------------------------------------------------
    # ЭТАП 1
    # ---------------------------------------------------------
    def _build_tab1(self):
        top = ttk.Frame(self.tab1)
        top.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top, text="Рабочая директория:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.work_dir, width=70).grid(row=0, column=1, columnspan=3, sticky=tk.W, padx=5)
        ttk.Button(top, text="Обзор...", command=self._choose_dir).grid(row=0, column=4, padx=5)
        
        ttk.Label(top, text="Тип файла:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_combo = ttk.Combobox(top, textvariable=self.file_type, values=["DF71.dat", "DF71Kz_pop1.dat"], state="readonly", width=30)
        self.file_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top, text="Сканировать заголовки", command=self._scan_headers).grid(row=1, column=4, padx=5)
        
        ttk.Label(top, text="Функционал:").grid(row=2, column=0, sticky=tk.W)
        self.func_combo = ttk.Combobox(top, textvariable=self.func_name, state="readonly", width=50)
        self.func_combo.grid(row=2, column=1, columnspan=3, sticky=tk.W, padx=5)
        
        self.progress = ttk.Progressbar(top, mode='determinate', length=400)
        self.progress.grid(row=4, column=0, columnspan=5, sticky=tk.W+tk.E, pady=5, padx=5)
        
        tk.Button(top, text="ЗАГРУЗИТЬ, НОРМАЛИЗОВАТЬ И СОХРАНИТЬ В ФАЙЛ",
                  command=self._stage1_process_and_save,
                  bg="#d4edda", relief="raised",
                  font=("Segoe UI", 10, "bold")).grid(row=3, column=1, columnspan=3, pady=10)

    def _choose_dir(self):
        d = filedialog.askdirectory(title="Выберите рабочую директорию")
        if d:
            self.work_dir.set(d)
            self._sync_settings()

    def _scan_headers(self):
        work = self.work_dir.get().strip()
        if not work or not os.path.isdir(work):
            messagebox.showerror("Ошибка", "Укажите корректную рабочую директорию.")
            return
            
        subdirs = sorted([name for name in os.listdir(work) if os.path.isdir(os.path.join(work, name)) and re.match(r'^\d+$', name)])
        self.dirs = subdirs
        
        if not subdirs:
            messagebox.showerror("Ошибка", "Не найдено поддиректорий с числовыми именами.")
            return
            
        ftype = self.file_type.get()
        first_header = None
        for d in subdirs:
            for target in os.listdir(os.path.join(work, d)):
                target_path = os.path.join(work, d, target)
                if os.path.isdir(target_path):
                    fpath = os.path.join(target_path, ftype)
                    if os.path.isfile(fpath):
                        first_header = fpath
                        break
            if first_header: break
            
        if not first_header:
            messagebox.showerror("Ошибка", f"Файлы типа '{ftype}' не найдены.")
            return
            
        try:
            n_cols, col_names = parse_header(first_header)
            self.func_names = col_names[1:]
            self.display_names = [name.replace(',', '_') for name in self.func_names]
            self.func_combo['values'] = self.display_names
            if self.func_name.get() in self.func_names:
                idx = self.func_names.index(self.func_name.get())
                self.func_combo.current(idx)
            else:
                self.func_combo.current(0)
            self.status.set(f"Найдено директорий: {len(subdirs)}. Функционалов: {len(self.func_names)}.")
            self._sync_settings()
        except Exception as e:
            messagebox.showerror("Ошибка чтения заголовка", str(e))

    def _stage1_process_and_save(self):
        if self.processing:
            messagebox.showwarning("Внимание", "Обработка уже запущена. Дождитесь завершения.")
            return
            
        work = self.work_dir.get().strip()
        selection_index = self.func_combo.current()
        if selection_index == -1:
            messagebox.showerror("Ошибка", "Выберите функционал.")
            return
        
        fname = self.func_names[selection_index]
        ftype = self.file_type.get()
        
        if not self.dirs or not fname:
            messagebox.showerror("Ошибка", "Сначала выполните сканирование и выберите функционал.")
            return
        
        self.processing = True
        self.progress['value'] = 0
        self.progress['maximum'] = len(self.dirs)
        self.status.set("Идет загрузка и нормализация данных...")
        
        thread = threading.Thread(target=self._stage1_worker, args=(work, fname, ftype), daemon=True)
        thread.start()
        self._update_progress()

    def _stage1_worker(self, work, fname, ftype):
        try:
            self.raw_curves.clear()
            
            first_header = None
            for d in self.dirs:
                for target in os.listdir(os.path.join(work, d)):
                    target_path = os.path.join(work, d, target)
                    if os.path.isdir(target_path):
                        fpath = os.path.join(target_path, ftype)
                        if os.path.isfile(fpath):
                            first_header = fpath
                            break
                if first_header: break
                
            _, col_names = parse_header(first_header)
            col_idx = col_names.index(fname)
            
            for i, d in enumerate(self.dirs):
                fpath = None
                for target in os.listdir(os.path.join(work, d)):
                    target_path = os.path.join(work, d, target)
                    if os.path.isdir(target_path):
                        candidate = os.path.join(target_path, ftype)
                        if os.path.isfile(candidate):
                            fpath = candidate
                            break
                if not fpath: continue
                
                data = load_data(fpath, len(col_names))
                if data.size == 0: continue
                
                prob = data[:, 0]
                values = data[:, col_idx]
                vs, pc = transform_array(prob, values)
                self.raw_curves[d] = (vs, pc)
                
                self.after(0, self._set_progress_value, i + 1)
                
            if not self.raw_curves:
                self.after(0, messagebox.showwarning, "Внимание", "Не удалось загрузить данные.")
                self.after(0, self._processing_finished)
                return
                
            out_path = os.path.join(BASE_DIR, f"{fname}.dat")
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(f"# FUNCTIONAL: {fname}\n")
                f.write(f"# CURVES_COUNT: {len(self.raw_curves)}\n")
                for dir_name, (x, y) in self.raw_curves.items():
                    f.write(f"# DIR: {dir_name}\n")
                    for xi, yi in zip(x, y):
                        f.write(f"{xi:.10g} {yi:.10g}\n")
            
            self.raw_curves.clear()
            self.after(0, self._processing_finished, out_path)
            
        except Exception as e:
            self.after(0, messagebox.showerror, "Ошибка", str(e))
            self.after(0, self._processing_finished)

    def _set_progress_value(self, value):
        self.progress['value'] = value

    def _update_progress(self):
        if self.processing:
            self.update_idletasks()
            self.after(100, self._update_progress)

    def _processing_finished(self, out_path=None):
        self.processing = False
        if out_path:
            self.stage2_file.set(out_path)
            self._stage2_log("Автоподстановка файла из Этапа 1: " + os.path.basename(out_path))
            
            self.stage3_file.set(out_path)
            self.lbl_stage3_status.config(text=f"Файл данных: {os.path.basename(out_path)}", foreground="green")
            
            self.status.set(f"Этап 1 завершен. Файл сохранен: {out_path}. Память очищена.")
            messagebox.showinfo("Этап 1", f"Данные успешно агрегированы и сохранены в:\n{out_path}\n\nМассивы из памяти очищены.\nПереходите ко второму этапу.")
        self._sync_settings()

    # ---------------------------------------------------------
    # ЭТАП 2
    # ---------------------------------------------------------
    def _build_tab2(self):
        top = ttk.Frame(self.tab2)
        top.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top, text="Файл данных (из Этапа 1):").grid(row=0, column=0, sticky=tk.W)
        self.stage2_file = tk.StringVar()
        ttk.Entry(top, textvariable=self.stage2_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(top, text="Обзор...", command=self._stage2_choose_file).grid(row=0, column=2, padx=5)
        
        ttk.Label(top, text="Коэфф. отсева (0..1):").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(top, textvariable=self.trim_coeff, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(top, text="Шаг по X:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(top, textvariable=self.step_x, width=6).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(top, text="Шаг по Y:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(top, textvariable=self.step_y, width=6).grid(row=3, column=1, sticky=tk.W, padx=5)
        
        self.progress2 = ttk.Progressbar(top, mode='determinate', length=400)
        self.progress2.grid(row=5, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5, padx=5)
        
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        tk.Button(btn_frame, text="РАССЧИТАТЬ СТАТИСТИКИ ПО X",
                  command=self._stage2_calculate_x,
                  bg="#cce5ff", relief="raised",
                  font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_frame, text="РАССЧИТАТЬ СТАТИСТИКИ ПО Y",
                  command=self._stage2_calculate_y,
                  bg="#cce5ff", relief="raised",
                  font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.lbl_stat_x_status = ttk.Label(top, text="Статистика по X: не рассчитана", foreground="gray")
        self.lbl_stat_x_status.grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        self.lbl_stat_y_status = ttk.Label(top, text="Статистика по Y: не рассчитана", foreground="gray")
        self.lbl_stat_y_status.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)
        
        log_frame = ttk.LabelFrame(self.tab2, text="Журнал операций Этапа 2")
        log_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.txt_stage2_log = tk.Text(log_frame, height=6, wrap=tk.WORD, state=tk.DISABLED,
                                      font=("Consolas", 9))
        self.txt_stage2_log.pack(fill=tk.X, padx=5, pady=5)

    def _stage2_log(self, message):
        def _do():
            self.txt_stage2_log.config(state=tk.NORMAL)
            self.txt_stage2_log.insert(tk.END, message + "\n")
            self.txt_stage2_log.see(tk.END)
            self.txt_stage2_log.config(state=tk.DISABLED)
        self.after(0, _do)

    def _stage2_choose_file(self):
        path = filedialog.askopenfilename(initialdir=BASE_DIR, filetypes=[("DAT files", "*.dat")])
        if path:
            self.stage2_file.set(path)
            self._stage2_log(f"Выбран файл: {os.path.basename(path)}")

    def _validate_stage2_params(self):
        filepath = self.stage2_file.get().strip()
        if not filepath or not os.path.isfile(filepath):
            messagebox.showerror("Ошибка", "Укажите корректный файл данных из Этапа 1.")
            return None
        
        try:
            k = float(self.trim_coeff.get().strip())
            if not (0.0 <= k <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Коэффициент отсева должен быть числом от 0 до 1.")
            return None
        
        try:
            step_x = int(self.step_x.get().strip())
            step_y = int(self.step_y.get().strip())
            if step_x < 1 or step_y < 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Шаги должны быть целыми числами >= 1.")
            return None
        
        return (filepath, k, step_x, step_y)

    def _stage2_calculate_x(self):
        if self.stage2_processing:
            messagebox.showwarning("Внимание", "Расчет уже запущен. Дождитесь завершения.")
            return
        
        params = self._validate_stage2_params()
        if params is None:
            return
        filepath, k, step_x, step_y = params
        
        self.stage2_processing = True
        self.progress2['value'] = 0
        self._stage2_log("Запуск расчета статистик ПО X...")
        self.lbl_stat_x_status.config(text="Статистика по X: расчет...", foreground="orange")
        
        thread = threading.Thread(target=self._stage2_worker_x, args=(filepath, k, step_x), daemon=True)
        thread.start()
        self._update_progress2()

    def _stage2_calculate_y(self):
        if self.stage2_processing:
            messagebox.showwarning("Внимание", "Расчет уже запущен. Дождитесь завершения.")
            return
        
        params = self._validate_stage2_params()
        if params is None:
            return
        filepath, k, step_x, step_y = params
        
        self.stage2_processing = True
        self.progress2['value'] = 0
        self._stage2_log("Запуск расчета статистик ПО Y...")
        self.lbl_stat_y_status.config(text="Статистика по Y: расчет...", foreground="orange")
        
        thread = threading.Thread(target=self._stage2_worker_y, args=(filepath, k, step_y), daemon=True)
        thread.start()
        self._update_progress2()

    def _stage2_load_progress(self, line_count, current_dir):
        def _do():
            self._stage2_log(f"  Загрузка: строка {line_count:,}, директория: {current_dir}")
        self.after(0, _do)

    def _stage2_worker_x(self, filepath, k, step_x):
        try:
            self._stage2_log("Загрузка данных из файла (это может занять время для больших файлов)...")
            curves = load_stage1_file(filepath, progress_callback=self._stage2_load_progress)
            
            if not curves:
                self.after(0, messagebox.showerror, "Ошибка", "Не удалось загрузить данные из файла.")
                self.after(0, self._stage2_finished, False, 'x')
                return
            
            self._stage2_log(f"Загружено кривых: {len(curves)}. Расчет статистик X -> Y...")
            stat_x = self._compute_stat_x(curves, k, step_x, progress_callback=self._stage2_update_progress)
            
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            out_x = os.path.join(BASE_DIR, f"{base_name}_mean_X.csv")
            self._save_stat_csv(out_x, stat_x, 'x', ['y_min', 'y_min_trim', 'y_mean_trim', 'y_max_trim', 'y_max'])
            
            self._stage2_log(f"Статистика по X сохранена: {out_x}")
            self.after(0, self._stage2_finished, True, 'x', out_x)
            
        except Exception as e:
            self._stage2_log(f"ОШИБКА: {e}")
            self.after(0, messagebox.showerror, "Ошибка расчета", str(e))
            self.after(0, self._stage2_finished, False, 'x')

    def _stage2_worker_y(self, filepath, k, step_y):
        try:
            self._stage2_log("Загрузка данных из файла (это может занять время для больших файлов)...")
            curves = load_stage1_file(filepath, progress_callback=self._stage2_load_progress)
            
            if not curves:
                self.after(0, messagebox.showerror, "Ошибка", "Не удалось загрузить данные из файла.")
                self.after(0, self._stage2_finished, False, 'y')
                return
            
            self._stage2_log(f"Загружено кривых: {len(curves)}. Расчет статистик Y -> X...")
            stat_y = self._compute_stat_y(curves, k, step_y, progress_callback=self._stage2_update_progress)
            
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            out_y = os.path.join(BASE_DIR, f"{base_name}_mean_Y.csv")
            self._save_stat_csv(out_y, stat_y, 'y', ['x_min', 'x_min_trim', 'x_mean_trim', 'x_max_trim', 'x_max'])
            
            self._stage2_log(f"Статистика по Y сохранена: {out_y}")
            self.after(0, self._stage2_finished, True, 'y', out_y)
            
        except Exception as e:
            self._stage2_log(f"ОШИБКА: {e}")
            self.after(0, messagebox.showerror, "Ошибка расчета", str(e))
            self.after(0, self._stage2_finished, False, 'y')

    def _stage2_update_progress(self, value, maximum):
        def _do():
            self.progress2['maximum'] = maximum
            self.progress2['value'] = value
        self.after(0, _do)

    def _update_progress2(self):
        if self.stage2_processing:
            self.update_idletasks()
            self.after(100, self._update_progress2)

    def _stage2_finished(self, success, axis, out_path=None):
        self.stage2_processing = False
        if success:
            if axis == 'x':
                self.stage2_stat_x_done = True
                self.lbl_stat_x_status.config(text=f"Статистика по X: готова ({os.path.basename(out_path)})", foreground="green")
            else:
                self.stage2_stat_y_done = True
                self.lbl_stat_y_status.config(text=f"Статистика по Y: готова ({os.path.basename(out_path)})", foreground="green")
            self.status.set(f"Этап 2 ({'X' if axis=='x' else 'Y'}): расчет завершен. Файл: {out_path}")
        else:
            if axis == 'x':
                self.lbl_stat_x_status.config(text="Статистика по X: ОШИБКА", foreground="red")
            else:
                self.lbl_stat_y_status.config(text="Статистика по Y: ОШИБКА", foreground="red")
        self._sync_settings()

    def _compute_stat_x(self, curves, k, step_x, progress_callback=None):
        keys = sorted(curves.keys())
        N = len(keys)
        
        all_vs = []
        x_min_arr = np.zeros(N)
        x_max_arr = np.zeros(N)
        
        for i, d in enumerate(keys):
            vs, pc = curves[d]
            all_vs.append(vs)
            x_min_arr[i] = vs.min()
            x_max_arr[i] = vs.max()
        
        x_grid = np.unique(np.concatenate(all_vs))
        if step_x > 1:
            x_grid = x_grid[::step_x]
        
        total_steps = x_grid.size
        y_matrix = np.zeros((N, x_grid.size))
        for i, d in enumerate(keys):
            vs, pc = curves[d]
            y_matrix[i, :] = np.interp(x_grid, vs, pc, left=pc[0], right=pc[-1])
        
        valid = (x_grid[None, :] >= x_min_arr[:, None]) & (x_grid[None, :] <= x_max_arr[:, None])
        
        y_max = np.full(x_grid.size, np.nan)
        y_min = np.full(x_grid.size, np.nan)
        y_max_trim = np.full(x_grid.size, np.nan)
        y_min_trim = np.full(x_grid.size, np.nan)
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
                y_max_trim[j] = y_trimmed[-1]
                y_min_trim[j] = y_trimmed[0]
                y_mean_trim[j] = np.mean(y_trimmed)
            else:
                y_max_trim[j] = y_sorted[-1]
                y_min_trim[j] = y_sorted[0]
                y_mean_trim[j] = np.mean(y_sorted)
            
            if progress_callback and (j % 100 == 0 or j == x_grid.size - 1):
                progress_callback(j + 1, total_steps)
        
        return (x_grid, y_min, y_min_trim, y_mean_trim, y_max_trim, y_max)

    def _compute_stat_y(self, curves, k, step_y, progress_callback=None):
        keys = sorted(curves.keys())
        N = len(keys)
        
        all_pc = []
        y_min_arr = np.zeros(N)
        y_max_arr = np.zeros(N)
        
        for i, d in enumerate(keys):
            vs, pc = curves[d]
            all_pc.append(pc)
            y_min_arr[i] = pc.min()
            y_max_arr[i] = pc.max()
        
        y_grid = np.unique(np.concatenate(all_pc))
        if step_y > 1:
            y_grid = y_grid[::step_y]
        
        total_steps = y_grid.size
        x_matrix = np.zeros((N, y_grid.size))
        for i, d in enumerate(keys):
            vs, pc = curves[d]
            x_matrix[i, :] = np.interp(y_grid, pc[::-1], vs[::-1], left=vs[0], right=vs[-1])
        
        valid = (y_grid[None, :] >= y_min_arr[:, None]) & (y_grid[None, :] <= y_max_arr[:, None])
        
        x_max = np.full(y_grid.size, np.nan)
        x_min = np.full(y_grid.size, np.nan)
        x_max_trim = np.full(y_grid.size, np.nan)
        x_min_trim = np.full(y_grid.size, np.nan)
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
                x_max_trim[j] = x_trimmed[-1]
                x_min_trim[j] = x_trimmed[0]
                x_mean_trim[j] = np.mean(x_trimmed)
            else:
                x_max_trim[j] = x_sorted[-1]
                x_min_trim[j] = x_sorted[0]
                x_mean_trim[j] = np.mean(x_sorted)
            
            if progress_callback and (j % 100 == 0 or j == y_grid.size - 1):
                progress_callback(j + 1, total_steps)
        
        return (y_grid, x_min, x_min_trim, x_mean_trim, x_max_trim, x_max)

    def _save_stat_csv(self, path, stat_data, grid_name, value_names):
        grid, v_min, v_min_trim, v_mean, v_max_trim, v_max = stat_data
        data = np.column_stack([grid, v_min, v_min_trim, v_mean, v_max_trim, v_max])
        cols = [grid_name] + value_names
        write_csv(path, cols, data)

    # ---------------------------------------------------------
    # ЭТАП 3 (ЧИСТАЯ ЛОГИКА ИЗ 2.0.1 — БЕЗ "ОПТИМИЗАЦИЙ")
    # ---------------------------------------------------------
    def _build_tab3(self):
        top = ttk.Frame(self.tab3)
        top.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top, text="Файл данных:").grid(row=0, column=0, sticky=tk.W)
        self.stage3_file = tk.StringVar()
        ttk.Entry(top, textvariable=self.stage3_file, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(top, text="Обзор...", command=self._stage3_choose_file).grid(row=0, column=2, padx=5)
        tk.Button(top, text="НАРИСОВАТЬ ГРАФИК",
                  command=self._stage3_plot,
                  bg="#d4edda", relief="raised",
                  font=("Segoe UI", 9, "bold")).grid(row=0, column=3, padx=10)
        
        ttk.Checkbutton(top, text="Log X", variable=self.log_x, command=self._apply_log_scales).grid(row=0, column=4, padx=10)
        ttk.Checkbutton(top, text="Log Y", variable=self.log_y, command=self._apply_log_scales).grid(row=0, column=5, padx=10)
        
        self.lbl_stage3_status = ttk.Label(self.tab3, text="Статус: выберите файл и нажмите 'Нарисовать график'.", foreground="blue")
        self.lbl_stage3_status.pack(fill=tk.X, padx=5, pady=2)
        
        points_frame = ttk.LabelFrame(self.tab3, text="Точки среза (клик по графику или ручной ввод)")
        points_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(points_frame, text="X:").grid(row=0, column=0, padx=5, pady=2)
        self.x_entry = ttk.Entry(points_frame, width=15)
        self.x_entry.grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(points_frame, text="Y:").grid(row=0, column=2, padx=5, pady=2)
        self.y_entry = ttk.Entry(points_frame, width=15)
        self.y_entry.grid(row=0, column=3, padx=5, pady=2)
        
        ttk.Button(points_frame, text="Добавить", command=self._add_point).grid(row=0, column=4, padx=5, pady=2)
        ttk.Button(points_frame, text="Удалить выбранную", command=self._remove_point).grid(row=0, column=5, padx=5, pady=2)
        ttk.Button(points_frame, text="Очистить все", command=self._clear_points).grid(row=0, column=6, padx=5, pady=2)
        
        self.points_listbox = tk.Listbox(points_frame, height=4, width=60)
        self.points_listbox.grid(row=1, column=0, columnspan=4, padx=5, pady=2, sticky=tk.W)
        
        self.fig3 = plt.Figure(figsize=(11, 6), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.tab3)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        toolbar = NavigationToolbar2Tk(self.canvas3, self.canvas3.get_tk_widget())
        toolbar.update()
        
        # ВАЖНО: НЕ трогаем обработку событий canvas.
        # НЕ отключаем Motion, НЕ трогаем format_coord.
        # Матплотлиб сам управляет этими событиями.
        
        self.canvas3.mpl_connect('button_press_event', self._on_canvas_click)
        
        self.ax3.text(0.5, 0.5, "Этап 3: Выберите файл и нажмите 'Нарисовать график'", ha='center', va='center', fontsize=14, color='gray')
        self.ax3.set_xticks([])
        self.ax3.set_yticks([])
        self.canvas3.draw()

    def _stage3_choose_file(self):
        path = filedialog.askopenfilename(initialdir=BASE_DIR, filetypes=[("DAT files", "*.dat")])
        if path:
            self.stage3_file.set(path)
            self.lbl_stage3_status.config(text=f"Файл данных: {os.path.basename(path)}", foreground="blue")

    def _stage3_plot(self):
        """Загрузка данных и построение графика.
        Загрузка идёт в главном потоке с time.sleep(0) каждые 10000 строк —
        этого достаточно для отзывчивости UI (проверено на файлах 88+ МБ).
        """
        path = self.stage3_file.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Ошибка", "Укажите корректный файл данных.")
            return
        
        base_name = os.path.splitext(os.path.basename(path))[0]
        stat_x_path = os.path.join(BASE_DIR, f"{base_name}_mean_X.csv")
        stat_y_path = os.path.join(BASE_DIR, f"{base_name}_mean_Y.csv")
        
        has_stat_x = os.path.isfile(stat_x_path)
        has_stat_y = os.path.isfile(stat_y_path)
        
        if not has_stat_x and not has_stat_y:
            messagebox.showerror("Ошибка", f"Не найдены файлы статистик:\n{stat_x_path}\n{stat_y_path}\n\nСначала выполните Этап 2.")
            return
        
        try:
            self.lbl_stage3_status.config(text="Загрузка данных...")
            self.update_idletasks()
            
            # Загрузка в главном потоке с GIL-release
            self.stage3_curves = load_stage1_file(path)
            self.stage3_stat_x = load_stage2_csv(stat_x_path) if has_stat_x else None
            self.stage3_stat_y = load_stage2_csv(stat_y_path) if has_stat_y else None
            
            if not self.stage3_curves:
                messagebox.showerror("Ошибка", "Не удалось загрузить исходные данные.")
                return
            
            self._plot_stage3()
            
            stat_info = []
            if has_stat_x: stat_info.append("X→Y")
            if has_stat_y: stat_info.append("Y→X")
            self.status.set(f"Загружено кривых: {len(self.stage3_curves)}. Статистики: {', '.join(stat_info)}.")
            self.lbl_stage3_status.config(text=f"Загружено: {len(self.stage3_curves)} кривых. Статистики: {', '.join(stat_info)}", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Ошибка загрузки", str(e))
            self.lbl_stage3_status.config(text=f"Ошибка: {e}", foreground="red")

    def _plot_stage3(self):
        self.ax3.clear()
        
        for d in sorted(self.stage3_curves.keys()):
            vs, pc = self.stage3_curves[d]
            self.ax3.plot(vs, pc, color='dodgerblue', alpha=0.3, linewidth=0.8, rasterized=True)
        
        handles = []
        labels = []
        
        if self.stage3_stat_x is not None:
            grid_x, y_min, y_min_trim, y_mean, y_max_trim, y_max = self.stage3_stat_x
            l1, = self.ax3.plot(grid_x, y_max, color='red', linewidth=2.0)
            l2, = self.ax3.plot(grid_x, y_min, color='blue', linewidth=2.0)
            l3, = self.ax3.plot(grid_x, y_max_trim, color='darkorange', linewidth=1.8, linestyle='--')
            l4, = self.ax3.plot(grid_x, y_min_trim, color='green', linewidth=1.8, linestyle='--')
            l5, = self.ax3.plot(grid_x, y_mean, color='purple', linewidth=2.5, linestyle='-.')
            handles.extend([l1, l2, l3, l4, l5])
            labels.extend(['X→Y: Max', 'X→Y: Min', 'X→Y: Max−trim', 'X→Y: Min+trim', 'X→Y: Mean'])
        
        if self.stage3_stat_y is not None:
            grid_y, x_min, x_min_trim, x_mean, x_max_trim, x_max = self.stage3_stat_y
            l6, = self.ax3.plot(x_max, grid_y, color='red', linewidth=1.5, linestyle=':', rasterized=True)
            l7, = self.ax3.plot(x_min, grid_y, color='blue', linewidth=1.5, linestyle=':', rasterized=True)
            l8, = self.ax3.plot(x_max_trim, grid_y, color='darkorange', linewidth=1.3, linestyle='--', rasterized=True)
            l9, = self.ax3.plot(x_min_trim, grid_y, color='green', linewidth=1.3, linestyle='--', rasterized=True)
            l10, = self.ax3.plot(x_mean, grid_y, color='purple', linewidth=1.8, linestyle='-.', rasterized=True)
            handles.extend([l6, l7, l8, l9, l10])
            labels.extend(['Y→X: Max', 'Y→X: Min', 'Y→X: Max−trim', 'Y→X: Min+trim', 'Y→X: Mean'])
        
        self.ax3.set_xlabel("Значение")
        self.ax3.set_ylabel("Вероятность")
        self.ax3.set_title("Исходные кривые и статистики")
        self.ax3.grid(True, which='both', ls=':', alpha=0.5)
        if handles:
            self.ax3.legend(handles=handles, labels=labels, fontsize='small', loc='best')
        
        self.fig3.tight_layout()
        self._apply_log_scales()
        self._plot_slice_points()
        self.canvas3.draw()

    def _apply_log_scales(self):
        try:
            self.ax3.set_xscale('log' if self.log_x.get() else 'linear')
            self.ax3.set_yscale('log' if self.log_y.get() else 'linear')
            self.fig3.tight_layout()
            self.canvas3.draw_idle()
        except Exception:
            pass

    def _on_canvas_click(self, event):
        if event.button != 1:
            return
        if event.inaxes != self.ax3:
            return
        x = event.xdata
        y = event.ydata
        if x is None or y is None:
            return
        self.slice_points.append((float(x), float(y)))
        self._update_points_listbox()
        self._plot_slice_points()

    def _add_point(self):
        try:
            x = float(self.x_entry.get().strip().replace(',', '.'))
            y = float(self.y_entry.get().strip().replace(',', '.'))
        except ValueError:
            messagebox.showerror("Ошибка", "Введите корректные числовые значения.")
            return
        self.slice_points.append((x, y))
        self._update_points_listbox()
        self._plot_slice_points()

    def _remove_point(self):
        sel = self.points_listbox.curselection()
        if not sel:
            messagebox.showwarning("Внимание", "Выберите точку для удаления.")
            return
        idx = sel[0]
        if 0 <= idx < len(self.slice_points):
            self.slice_points.pop(idx)
            self._update_points_listbox()
            self._plot_slice_points()

    def _clear_points(self):
        if not self.slice_points:
            return
        if messagebox.askyesno("Подтверждение", "Удалить все точки среза?"):
            self.slice_points.clear()
            self._update_points_listbox()
            self._plot_slice_points()

    def _update_points_listbox(self):
        self.points_listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(self.slice_points):
            self.points_listbox.insert(tk.END, f"#{i+1}: X={x:.6g}, Y={y:.6g}")

    def _plot_slice_points(self):
        """Отрисовка точек среза. Прямой вызов draw_idle() без throttle."""
        for artist in self._slice_point_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._slice_point_artists.clear()
        
        if not self.slice_points:
            self.canvas3.draw_idle()
            return
        
        slice_x = [p[0] for p in self.slice_points]
        slice_y = [p[1] for p in self.slice_points]
        marker = self.ax3.scatter(slice_x, slice_y, color='red', s=50, marker='x', zorder=5)
        self._slice_point_artists.append(marker)
        self.canvas3.draw_idle()

    # ---------------------------------------------------------
    # ЭТАП 4
    # ---------------------------------------------------------
    def _build_tab4(self):
        top = ttk.Frame(self.tab4)
        top.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(top, text="Коэффициент отсева (0..1):").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Entry(top, textvariable=self.trim_coeff, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        info_frame = ttk.LabelFrame(self.tab4, text="Информация")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(info_frame, text="Точки среза (синхронизированы с Этапом 3):").pack(anchor=tk.W, padx=5, pady=2)
        
        self.txt_points = tk.Text(info_frame, height=8, width=80)
        self.txt_points.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(info_frame, text="Обновить из Этапа 3", command=self._stage4_refresh_points).pack(pady=2)
        
        btn_frame = ttk.Frame(self.tab4)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        tk.Button(btn_frame, text="СОХРАНИТЬ СРЕЗ ПО X",
                  command=self._stage4_save_x,
                  bg="#cce5ff", relief="raised",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=20, pady=5)
        
        tk.Button(btn_frame, text="СОХРАНИТЬ СРЕЗ ПО Y",
                  command=self._stage4_save_y,
                  bg="#cce5ff", relief="raised",
                  font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT, padx=20, pady=5)
        
        self.lbl_stage4_status = ttk.Label(self.tab4, text="Статус: ожидание.", foreground="blue")
        self.lbl_stage4_status.pack(pady=10)
        
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, event):
        try:
            current_tab = self.notebook.index(self.notebook.select())
            if current_tab == 3:
                self._stage4_refresh_points()
        except Exception:
            pass

    def _stage4_refresh_points(self):
        self.txt_points.delete(1.0, tk.END)
        if not self.slice_points:
            self.txt_points.insert(tk.END, "(список точек пуст — добавьте точки на Этапе 3)")
            return
        for i, (x, y) in enumerate(self.slice_points):
            self.txt_points.insert(tk.END, f"#{i+1}: X={x:.10g}, Y={y:.10g}\n")

    def _stage4_save_x(self):
        if not self.slice_points:
            messagebox.showwarning("Внимание", "Список точек среза пуст.")
            return
        if not self.stage3_curves:
            messagebox.showerror("Ошибка", "Не загружены исходные кривые. Выполните Этап 3.")
            return
        try:
            k = float(self.trim_coeff.get().strip())
            if not (0.0 <= k <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Коэффициент отсева должен быть числом от 0 до 1.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Сохранить срез по X",
            initialfile="X_slice.csv",
            initialdir=BASE_DIR,
        )
        if not path:
            return
        
        keys = sorted(self.stage3_curves.keys())
        N = len(keys)
        rows = []
        
        for x_given, _ in self.slice_points:
            y_values = []
            for d in keys:
                vs, pc = self.stage3_curves[d]
                x_min, x_max = vs.min(), vs.max()
                if x_min <= x_given <= x_max:
                    y_val = float(np.interp(x_given, vs, pc))
                else:
                    y_val = np.nan
                y_values.append(y_val)
            v_min, v_min_trim, v_mean_trim, v_max_trim, v_max = compute_trim_stats(y_values, k)
            row = [x_given, v_min, v_min_trim, v_mean_trim, v_max_trim, v_max] + y_values
            rows.append(row)
        
        col_names = ["X_given", "y_min", "y_min_trim", "y_mean_trim", "y_max_trim", "y_max"]
        col_names += [f"y_{i+1}" for i in range(N)]
        
        try:
            write_csv(path, col_names, rows)
            self.lbl_stage4_status.config(text=f"Срез по X сохранен: {path}", foreground="green")
            self.status.set(f"Срез по X сохранен: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    def _stage4_save_y(self):
        if not self.slice_points:
            messagebox.showwarning("Внимание", "Список точек среза пуст.")
            return
        if not self.stage3_curves:
            messagebox.showerror("Ошибка", "Не загружены исходные кривые. Выполните Этап 3.")
            return
        try:
            k = float(self.trim_coeff.get().strip())
            if not (0.0 <= k <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror("Ошибка", "Коэффициент отсева должен быть числом от 0 до 1.")
            return
        
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Сохранить срез по Y",
            initialfile="Y_slice.csv",
            initialdir=BASE_DIR,
        )
        if not path:
            return
        
        keys = sorted(self.stage3_curves.keys())
        N = len(keys)
        rows = []
        
        for _, y_given in self.slice_points:
            x_values = []
            for d in keys:
                vs, pc = self.stage3_curves[d]
                pc_min, pc_max = pc.min(), pc.max()
                if pc_min <= y_given <= pc_max:
                    x_val = float(np.interp(y_given, pc[::-1], vs[::-1]))
                else:
                    x_val = np.nan
                x_values.append(x_val)
            v_min, v_min_trim, v_mean_trim, v_max_trim, v_max = compute_trim_stats(x_values, k)
            row = [y_given, v_min, v_min_trim, v_mean_trim, v_max_trim, v_max] + x_values
            rows.append(row)
        
        col_names = ["Y_given", "x_min", "x_min_trim", "x_mean_trim", "x_max_trim", "x_max"]
        col_names += [f"x_{i+1}" for i in range(N)]
        
        try:
            write_csv(path, col_names, rows)
            self.lbl_stage4_status.config(text=f"Срез по Y сохранен: {path}", foreground="green")
            self.status.set(f"Срез по Y сохранен: {path}")
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))

    # ---------------------------------------------------------
    # Служебные методы
    # ---------------------------------------------------------
    def _sync_settings(self):
        self.settings.update({
            "work_dir": self.work_dir.get().strip(),
            "file_type": self.file_type.get(),
            "func_name": self.func_name.get(),
            "trim_coeff": self.trim_coeff.get().strip(),
            "step_x": self.step_x.get().strip(),
            "step_y": self.step_y.get().strip(),
            "log_x": self.log_x.get(),
            "log_y": self.log_y.get()
        })
        save_settings(self.settings)

    def _on_close(self):
        self._sync_settings()
        self.destroy()

if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем.")