"""Сохранение результатов прогонов."""

import csv
import datetime as dt
import os


def save_rows(rows, path, run_id=None):
    """Дописывает строки результатов в CSV, создавая файл и заголовок при нужде.

    Все строки приводятся к общему набору колонок, чтобы файл оставался
    читаемым при добавлении новых кодов с другими параметрами.
    """
    if not rows:
        return path

    stamp = dt.datetime.now(tz=dt.timezone.utc).isoformat(timespec="seconds")
    run_id = run_id or stamp
    enriched = [{"run_id": run_id, "timestamp": stamp, **row} for row in rows]

    existing = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing = next(reader, [])

    fields = list(existing)
    for row in enriched:
        for key in row:
            if key not in fields:
                fields.append(key)

    rewrite = fields != existing
    if rewrite and existing:
        with open(path, newline="", encoding="utf-8") as handle:
            old = list(csv.DictReader(handle))
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, restval="")
            writer.writeheader()
            writer.writerows(old)
    elif not existing:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writerows(enriched)
    return path


def print_table(rows, columns=("code", "n", "k", "rate", "channel", "level",
                               "fer", "fer_lo", "fer_hi", "ber",
                               "mean_iterations", "blocks")):
    """Печать результатов в виде выровненной таблицы."""
    def fmt(value):
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    table = [list(columns)]
    table += [[fmt(row.get(col, "")) for col in columns] for row in rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(columns))]
    for i, line in enumerate(table):
        print("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(line)))
        if i == 0:
            print("  ".join("-" * w for w in widths))
