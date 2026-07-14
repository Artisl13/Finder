# -*- coding: utf-8 -*-
import os
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# =========================================================
#                 Работа с файлами данных
# =========================================================

def parse_header(filepath):
    """
    Читает заголовок файла.
    Возвращает:
      - n_cols: число столбцов (int)
      - col_names: список названий функционалов (первый столбец - Probability)
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # строка 1 - название файла
    # строка 2 - число столбцов
    n_cols = int(lines[1].strip().split()[0])

    # строки 3 .. 3+n_cols-1 - названия столбцов
    col_names = []
    for i in range(2, 2 + n_cols):
        # убираем комментарий после "-" если он есть, но оставляем само имя
        name = lines[i].strip()
        # формат: "SpDf 7.5 Pop 1                      - коллективная доза..."
        if ' - ' in name:
            name = name.split(' - ')[0].strip()
        col_names.append(name)

    return n_cols, col_names


def load_data(filepath, n_cols):
    """
    Читает числовые данные из файла.
    Возвращает numpy-массив формы (n_rows, n_cols).
    Строки данных начинаются после заголовка (после 2+n_cols строк).
    Комментарий после '//' игнорируется.
    """
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    header_lines = 2 + n_cols  # 1 (title) + 1 (n_cols) + n_cols (names)
    data_lines = lines[header_lines:]

    rows = []
    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        # убираем комментарий
        if '//' in line:
            line = line.split('//')[0].strip()
        if not line:
            continue
        parts = line.split()
        # в файле могут встретиться числа через запятую в комментарии, но данные - через пробел
        # на всякий случай заменим запятые на точки (для европейских записей)
        parts = [p.replace(',', '.') for p in parts]
        try:
            row = [float(x) for x in parts[:n_cols]]
            if len(row) == n_cols:
                rows.append(row)
        except ValueError:
            continue

    return np.array(rows) if rows else np.empty((0, n_cols))


# =========================================================
#                 Алгоритм перестройки массива
# =========================================================

def transform_array(prob, values):
    """
    Алгоритм:
      а) Сортируем по values (возрастание), вместе с prob.
      б) Нулевые значения не принципиально как сортируются.
      в) Для первого (минимального) значения: вероятность = сумма всех вероятностей.
      г) Второе значение вероятности = сумма_всех - вероятность_второго значения.
         И так далее: P_i = sum(P) - sum(P[1..i]) = sum(P[i+1..end])
         Это убывающая кумулятивная вероятность P(X >= x).
    """
    prob = np.asarray(prob, dtype=float)
    values = np.asarray(values, dtype=float)

    # сортируем по значениям функционала
    order = np.argsort(values, kind='stable')
    values_sorted = values[order]
    prob_sorted = prob[order]

    total = np.sum(prob_sorted)

    # убывающая кумулятивная вероятность
    # P_cum[i] = sum(prob_sorted[i:])
    # реализуем через обратный cumsum
    prob_cum = np.cumsum(prob_sorted[::-1])[::-1]

    return values_sorted, prob_cum


# =========================================================
#                     GUI
# =========================================================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Обработка данных радиационной защиты")
        self.geometry("1100x700")

        self.work_dir = tk.StringVar(value="")
        self.file_type = tk.StringVar(value="DF71.dat")
        self.func_name = tk.StringVar(value="")

        self.func_names = []          # список функционалов (после чтения заголовка)
        self.dirs = []                # список поддиректорий (0001, 0002, ...)
        self.loaded_data = {}         # {dir_name: (values_sorted, prob_cum)}

        self._build_ui()

    # ----- построение интерфейса -----
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=5, pady=5)

        # строка 1: рабочая директория
        ttk.Label(top, text="Рабочая директория:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(top, textvariable=self.work_dir, width=70).grid(row=0, column=1, padx=5)
        ttk.Button(top, text="Обзор...", command=self._choose_dir).grid(row=0, column=2, padx=5)

        # строка 2: тип файла
        ttk.Label(top, text="Тип файла:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.file_combo = ttk.Combobox(top, textvariable=self.file_type,
                                       values=["DF71.dat", "DF71Kz_pop1.dat"],
                                       state="readonly", width=30)
        self.file_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        ttk.Button(top, text="Сканировать заголовки", command=self._scan_headers).grid(row=1, column=2, padx=5)

        # строка 3: выбор функционала
        ttk.Label(top, text="Функционал:").grid(row=2, column=0, sticky=tk.W)
        self.func_combo = ttk.Combobox(top, textvariable=self.func_name,
                                       state="readonly", width=50)
        self.func_combo.grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Button(top, text="Загрузить данные и построить график",
                   command=self._load_and_plot).grid(row=2, column=2, padx=5)

        # область графика
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas.get_tk_widget())
        toolbar.update()

        # статус
        self.status = tk.StringVar(value="Готово.")
        ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, side=tk.BOTTOM)

    # ----- выбор директории -----
    def _choose_dir(self):
        d = filedialog.askdirectory(title="Выберите рабочую директорию")
        if d:
            self.work_dir.set(d)
            self.status.set(f"Выбрана директория: {d}")

    # ----- сканирование заголовков -----
    def _scan_headers(self):
        work = self.work_dir.get().strip()
        if not work or not os.path.isdir(work):
            messagebox.showerror("Ошибка", "Укажите корректную рабочую директорию.")
            return

        # ищем поддиректории вида 0001, 0002, ...
        subdirs = []
        for name in os.listdir(work):
            full = os.path.join(work, name)
            if os.path.isdir(full) and re.match(r'^\d+$', name):
                subdirs.append(name)
        subdirs.sort()
        self.dirs = subdirs

        if not subdirs:
            messagebox.showerror("Ошибка", "В рабочей директории не найдено поддиректорий с числовыми именами.")
            return

        # читаем заголовок первого попавшегося файла выбранного типа
        ftype = self.file_type.get()
        first_header = None
        for d in subdirs:
            # ищем файл во всех подпапках Target_XXX
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

        self.func_names = col_names[1:]  # первый столбец - Probability
        self.func_combo['values'] = self.func_names
        if self.func_names:
            self.func_combo.current(0)

        self.status.set(f"Найдено директорий: {len(subdirs)}. "
                        f"Столбцов в файле: {n_cols}. Функционалов: {len(self.func_names)}.")

    # ----- загрузка данных и построение -----
    def _load_and_plot(self):
        work = self.work_dir.get().strip()
        if not work or not self.dirs:
            messagebox.showerror("Ошибка", "Сначала выполните сканирование заголовков.")
            return

        fname = self.func_name.get()
        if not fname:
            messagebox.showerror("Ошибка", "Выберите функционал.")
            return
        if fname not in self.func_names:
            messagebox.showerror("Ошибка", f"Функционал '{fname}' не найден в списке.")
            return

        ftype = self.file_type.get()
        self.loaded_data.clear()

        # индекс функционала в столбцах (0 - Probability)
        # нужно заново прочитать заголовок, чтобы получить индексы
        # но мы уже знаем порядок col_names; найдём индекс по имени
        # читаем заголовок из первого файла
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

        n_cols, col_names = parse_header(first_header)
        try:
            col_idx = col_names.index(fname)
        except ValueError:
            messagebox.showerror("Ошибка", f"Не удалось найти столбец '{fname}'.")
            return

        # проходим по всем директориям
        for d in self.dirs:
            d_path = os.path.join(work, d)
            # ищем файл в любой подпапке Target_*
            fpath = None
            for target in os.listdir(d_path):
                target_path = os.path.join(d_path, target)
                if os.path.isdir(target_path):
                    candidate = os.path.join(target_path, ftype)
                    if os.path.isfile(candidate):
                        fpath = candidate
                        break
            if fpath is None:
                continue

            try:
                data = load_data(fpath, n_cols)
            except Exception as e:
                print(f"Ошибка чтения {fpath}: {e}")
                continue

            if data.size == 0:
                continue

            prob = data[:, 0]
            values = data[:, col_idx]

            vs, pc = transform_array(prob, values)
            self.loaded_data[d] = (vs, pc)

        if not self.loaded_data:
            messagebox.showwarning("Внимание", "Не удалось загрузить данные ни из одной директории.")
            return

        # ----- построение графика -----
        self.ax.clear()
        for d in sorted(self.loaded_data.keys()):
            vs, pc = self.loaded_data[d]
            self.ax.plot(vs, pc, label=d)

        self.ax.set_xlabel(fname)
        self.ax.set_ylabel("Вероятность (кумулятивная убывающая)")
        self.ax.set_title(f"Кривые по директориям. Функционал: {fname}")
        self.ax.set_xscale('log') if np.any(np.array([v[0] for v, _ in self.loaded_data.values()]) > 0) else None
        self.ax.legend(fontsize='small', loc='best')
        self.ax.grid(True, which='both', ls=':', alpha=0.5)
        self.fig.tight_layout()
        self.canvas.draw()

        self.status.set(f"Загружено директорий: {len(self.loaded_data)}.")


# =========================================================
#                         Запуск
# =========================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()