import csv
import html
import io
import logging
import os
import re
import sqlite3
import statistics
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone

log = logging.getLogger("bot.analytics")

DB_FILE = os.getenv("ANALYTICS_DB", "/data/analytics.db")
UTC_OFFSET = float(os.getenv("STATS_UTC_OFFSET", "3"))
AUTO_TARGET = float(os.getenv("AUTO_SOLVE_TARGET", "40"))
TZ = timezone(timedelta(hours=UTC_OFFSET))
OFFSET_SEC = UTC_OFFSET * 3600
DEFLECT_WINDOW = 900
REACTION_WINDOW = 86400
CAPTION_LIMIT = 1000

PERIODS = {
    "today": "Сегодня",
    "7d": "7 дней",
    "30d": "30 дней",
    "all": "Всё время",
}

# Modern Analytics palette
GREEN = "#36A783"
AMBER = "#E9A23B"
RED = "#E16B73"
BLUE = "#648DE5"
GREY = "#8491A5"
GRID = "#EDF0F5"

BG = "#F4F6FA"
CARD = "#FFFFFF"
TEXT = "#253047"
MUTED = "#8994A7"

PALETTE = [
    "#648DE5",
    "#36A783",
    "#E9A23B",
    "#A58BE0",
    "#E16B73",
]


def _connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_FILE) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_FILE, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS events ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, kind TEXT NOT NULL, "
        "user_id INTEGER, outcome TEXT, latency REAL, question TEXT)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts)")
    return conn


def record(kind: str, user_id: int | None = None, outcome: str | None = None,
           latency: float | None = None, question: str | None = None, ts: float | None = None) -> None:
    try:
        with closing(_connect()) as conn:
            conn.execute(
                "INSERT INTO events (ts, kind, user_id, outcome, latency, question) VALUES (?,?,?,?,?,?)",
                (ts or time.time(), kind, user_id, outcome, latency, (question or "")[:500] or None),
            )
            conn.commit()
    except Exception as e:
        log.warning("Не удалось записать событие аналитики: %s", e)


def load_events(start: float | None, end: float) -> list[dict]:
    with closing(_connect()) as conn:
        if start is None:
            rows = conn.execute("SELECT * FROM events WHERE ts <= ? ORDER BY ts", (end,)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM events WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end)
            ).fetchall()
    return [dict(r) for r in rows]


def load_first_seen() -> dict[int, float]:
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT user_id, MIN(ts) AS first_ts FROM events "
            "WHERE kind = 'question' AND user_id IS NOT NULL GROUP BY user_id"
        ).fetchall()
    return {r["user_id"]: r["first_ts"] for r in rows}


def period_bounds(period: str, now: float | None = None):
    now = now or time.time()
    midnight = datetime.fromtimestamp(now, TZ).replace(hour=0, minute=0, second=0, microsecond=0)
    today = midnight.timestamp()
    days = {"today": 1, "7d": 7, "30d": 30}.get(period)
    if days is None:
        return None, now, None, None
    start = today - (days - 1) * 86400
    prev_start = start - days * 86400
    return start, now, prev_start, prev_start + (now - start)


def _local(ts: float) -> datetime:
    return datetime.fromtimestamp(ts, TZ)


def _bucket_size(period: str, start: float | None, end: float, first_ts: float | None) -> int:
    if period == "today":
        return 3600
    if period in ("7d", "30d"):
        return 86400
    span = end - (first_ts if first_ts else end)
    return 86400 if span <= 60 * 86400 else 7 * 86400


def _bucket_key(ts: float, size: int) -> int:
    return int((ts + OFFSET_SEC) // size)


def _bucket_label(key: int, size: int) -> str:
    moment = datetime.fromtimestamp(key * size - OFFSET_SEC, TZ)
    return moment.strftime("%H") if size == 3600 else moment.strftime("%d.%m")


def summarize(events: list[dict], first_seen: dict[int, float], start: float | None) -> dict:
    s = {
        "total": 0, "answered": 0, "not_found": 0, "errors": 0,
        "escalations": 0, "operator_replies": 0, "pos": 0, "neg": 0,
        "registrations": 0, "access_requests": 0,
        "users": set(), "latencies": [], "deflected": 0, "reactions": [],
    }
    esc_by_user: dict[int, list[float]] = defaultdict(list)
    reply_by_user: dict[int, list[float]] = defaultdict(list)
    for e in events:
        kind, user, outcome = e["kind"], e["user_id"], e["outcome"]
        if kind == "question":
            s["total"] += 1
            if outcome == "answered":
                s["answered"] += 1
            elif outcome == "not_found":
                s["not_found"] += 1
            else:
                s["errors"] += 1
            if e["latency"] is not None and outcome in ("answered", "not_found"):
                s["latencies"].append(e["latency"])
            if user is not None:
                s["users"].add(user)
        elif kind == "escalation":
            s["escalations"] += 1
            esc_by_user[user].append(e["ts"])
        elif kind == "operator_reply":
            s["operator_replies"] += 1
            reply_by_user[user].append(e["ts"])
        elif kind == "feedback":
            s["pos" if outcome == "positive" else "neg"] += 1
        elif kind == "registration":
            s["registrations"] += 1
        elif kind == "access_request":
            s["access_requests"] += 1
    for e in events:
        if e["kind"] == "question" and e["outcome"] == "answered":
            later = esc_by_user.get(e["user_id"])
            if not later:
                s["deflected"] += 1
                continue
            idx = bisect_right(later, e["ts"])
            if idx >= len(later) or later[idx] - e["ts"] > DEFLECT_WINDOW:
                s["deflected"] += 1
    for user, escalations in esc_by_user.items():
        replies = reply_by_user.get(user)
        if not replies:
            continue
        for ts in escalations:
            idx = bisect_left(replies, ts)
            if idx < len(replies) and replies[idx] - ts <= REACTION_WINDOW:
                s["reactions"].append(replies[idx] - ts)
    s["new_users"] = sum(1 for u in s["users"] if start is None or first_seen.get(u, 0) >= start)
    return s


def _pct(part: int, whole: int) -> str:
    return f"{part / whole * 100:.0f}%" if whole else "—"


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 60:
        return f"{value:.0f} с"
    if value < 3600:
        return f"{value / 60:.0f} мин"
    return f"{value / 3600:.1f} ч"


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def _delta(cur: int, prev: int | None) -> str:
    if not prev:
        return ""
    diff = (cur - prev) / prev * 100
    if abs(diff) < 1:
        return " ≈"
    return f" {'▲' if diff > 0 else '▼'} {diff:+.0f}%"


def _range_text(period: str, start: float | None, end: float, first_ts: float | None) -> str:
    if period == "today":
        return _local(end).strftime("%d.%m.%Y")
    begin = start if start is not None else first_ts
    if begin is None:
        return _local(end).strftime("%d.%m.%Y")
    return f"{_local(begin).strftime('%d.%m')} – {_local(end).strftime('%d.%m.%Y')}"


def build_caption(period: str, cur: dict, prev: dict | None, start: float | None,
                  end: float, first_ts: float | None) -> str:
    head = f"📊 <b>Аналитика бота — {PERIODS[period]}</b>\n<i>{_range_text(period, start, end, first_ts)}</i>\n\n"
    if cur["total"] == 0 and not cur["escalations"] and not cur["registrations"]:
        return head + "За выбранный период данных пока нет."
    total = cur["total"]
    lat = cur["latencies"]
    auto_share = cur["answered"] / total * 100 if total else 0
    users_line = f"Пользователей: <b>{len(cur['users'])}</b>{_delta(len(cur['users']), len(prev['users']) if prev else None)}"
    if start is not None:
        users_line += f", новых: {cur['new_users']}"
    lines = [
        f"Вопросов: <b>{total}</b>{_delta(total, prev['total'] if prev else None)}",
        users_line,
        f"Ответ найден в базе: <b>{auto_share:.0f}%</b>",
        f"Решено без оператора: <b>{cur['deflected'] / total * 100:.0f}%</b>" if total else "Решено без оператора: —",
        f"Не найдено: {cur['not_found'] / total * 100:.0f}% · Ошибок: {cur['errors'] / total * 100:.0f}%" if total else "Не найдено: — · Ошибок: —",
    ]
    esc = f"Эскалаций: <b>{cur['escalations']}</b> ({cur['escalations'] / total * 100:.0f}% вопросов)" if total else f"Эскалаций: <b>{cur['escalations']}</b>"
    if cur["reactions"]:
        esc += f", реакция оператора ≈ {_fmt_seconds(statistics.median(cur['reactions']))}"
    lines.append(esc)
    rated = cur["pos"] + cur["neg"]
    lines.append(f"Оценки: 👍 {cur['pos']} / 👎 {cur['neg']}, CSAT {_pct(cur['pos'], rated) if rated else '—'}")
    if lat:
        lines.append(
            f"Время ответа: ср. {_fmt_seconds(sum(lat) / len(lat))}, "
            f"медиана {_fmt_seconds(statistics.median(lat))}, p95 {_fmt_seconds(_percentile(lat, 0.95))}"
        )
    if cur["registrations"] or cur["access_requests"]:
        lines.append(f"Регистраций: {cur['registrations']}, запросов доступа: {cur['access_requests']}")
    if prev is not None:
        tail = "вчера к этому времени" if period == "today" else "предыдущим периодом"
        lines.append(f"\n<i>Стрелки: сравнение с {tail}.</i>")
    return head + "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def _short(text: str, limit: int = 70) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _question_stats(events: list[dict]) -> dict[str, dict]:
    items: dict[str, dict] = {}
    for e in events:
        if not e["question"]:
            continue
        key = _normalize(e["question"])
        if not key:
            continue
        item = items.setdefault(
            key, {"text": e["question"], "total": 0, "answered": 0, "not_found": 0, "errors": 0, "neg": 0}
        )
        if e["kind"] == "question":
            item["total"] += 1
            if e["outcome"] == "answered":
                item["answered"] += 1
            elif e["outcome"] == "not_found":
                item["not_found"] += 1
            else:
                item["errors"] += 1
        elif e["kind"] == "feedback" and e["outcome"] == "negative":
            item["neg"] += 1
    return items


def _top_questions(events: list[dict], limit: int) -> list[tuple[str, int, int]]:
    ranked = sorted(
        (v for v in _question_stats(events).values() if v["total"]),
        key=lambda v: v["total"], reverse=True,
    )
    return [(v["text"], v["total"], v["answered"]) for v in ranked[:limit]]


def weak_spots_text(period: str) -> str:
    start, end, _, _ = period_bounds(period)
    items = _question_stats(load_events(start, end))
    weak = [v for v in items.values() if v["not_found"] or v["neg"]]
    weak.sort(key=lambda v: v["not_found"] + v["neg"], reverse=True)
    head = f"🔍 <b>Слабые места базы знаний — {PERIODS[period]}</b>\n"
    if not weak:
        return head + "\nВопросов без ответа и с оценкой 👎 за период нет."
    head += "Вопросы, на которые бот не нашёл ответ или получил 👎. Кандидаты на пополнение документации:\n\n"
    lines = []
    for i, v in enumerate(weak[:10], 1):
        marks = []
        if v["not_found"]:
            marks.append(f"❌ {v['not_found']}")
        if v["neg"]:
            marks.append(f"👎 {v['neg']}")
        lines.append(f"{i}. <i>{html.escape(_short(v['text']))}</i> — {' · '.join(marks)}")
    return head + "\n".join(lines)


def top_questions_text(period: str) -> str:
    start, end, _, _ = period_bounds(period)
    top = _top_questions(load_events(start, end), 10)
    head = f"🏆 <b>Частые вопросы — {PERIODS[period]}</b>\n\n"
    if not top:
        return head + "За выбранный период вопросов нет."
    lines = [
        f"{i}. <i>{html.escape(_short(text))}</i> — {total} (✅ {_pct(answered, total)})"
        for i, (text, total, answered) in enumerate(top, 1)
    ]
    return head + "\n".join(lines)


def export_csv(period: str) -> bytes | None:
    start, end, _, _ = period_bounds(period)
    events = load_events(start, end)
    if not events:
        return None
    kinds = {
        "question": "вопрос", "escalation": "эскалация", "operator_reply": "ответ оператора",
        "feedback": "оценка", "registration": "регистрация", "access_request": "запрос доступа",
    }
    results = {
        "answered": "ответ найден", "not_found": "не найдено", "error": "ошибка",
        "positive": "👍", "negative": "👎",
    }
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(["Время", "Событие", "Пользователь", "Результат", "Время ответа, с", "Вопрос"])
    for e in events:
        writer.writerow([
            _local(e["ts"]).strftime("%Y-%m-%d %H:%M:%S"),
            kinds.get(e["kind"], e["kind"]),
            e["user_id"] if e["user_id"] is not None else "",
            results.get(e["outcome"], e["outcome"] or ""),
            f"{e['latency']:.1f}".replace(".", ",") if e["latency"] is not None else "",
            e["question"] or "",
        ])
    return buf.getvalue().encode("utf-8-sig")


def _style(ax, title: str) -> None:
    """Единый стиль осей для всех графиков дашборда."""
    ax.set_facecolor(CARD)
    ax.set_title(
        title,
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        pad=12,
    )

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.tick_params(
        axis="both",
        colors=MUTED,
        labelsize=8,
        length=0,
        pad=5,
    )

    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        linestyle="-",
    )
    ax.set_axisbelow(True)


def _empty(ax, text: str = "Нет данных") -> None:
    ax.text(
        0.5, 0.5, text,
        ha="center", va="center",
        color=GREY, fontsize=11,
        transform=ax.transAxes,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_dashboard(period: str, events: list[dict], cur: dict, start: float | None,
                     end: float, first_ts: float | None) -> bytes:
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    questions = [e for e in events if e["kind"] == "question"]
    size = _bucket_size(period, start, end, first_ts)
    begin = start if start is not None else (first_ts or end)

    first_key = _bucket_key(begin, size)
    last_key = _bucket_key(end, size)
    keys = list(range(first_key, last_key + 1))
    index = {k: i for i, k in enumerate(keys)}

    answered = [0] * len(keys)
    not_found = [0] * len(keys)
    errors = [0] * len(keys)
    lat_sum = [0.0] * len(keys)
    lat_n = [0] * len(keys)

    for e in questions:
        i = index.get(_bucket_key(e["ts"], size))
        if i is None:
            continue

        if e["outcome"] == "answered":
            answered[i] += 1
        elif e["outcome"] == "not_found":
            not_found[i] += 1
        else:
            errors[i] += 1

        if e["latency"] is not None and e["outcome"] in ("answered", "not_found"):
            lat_sum[i] += e["latency"]
            lat_n[i] += 1

    # Ручная сетка вместо constrained_layout: она не даёт заголовкам,
    # подписям осей и легендам самопроизвольно сдвигать соседние графики.
    fig = Figure(figsize=(16, 9), dpi=120, facecolor=BG)
    grid = fig.add_gridspec(
        2, 3,
        left=0.065, right=0.975,
        bottom=0.095, top=0.865,
        height_ratios=[1.08, 1],
        hspace=0.48, wspace=0.34,
    )

    fig.text(
        0.065, 0.965, "БАРС-Помощник",
        ha="left", va="top",
        fontsize=22, fontweight="bold", color=TEXT,
    )
    fig.text(
        0.066, 0.925,
        f"АНАЛИТИКА  /  {PERIODS[period].upper()}  /  "
        f"{_range_text(period, start, end, first_ts)}",
        ha="left", va="top", fontsize=9, color=MUTED,
    )

    # 1. Динамика вопросов
    ax = fig.add_subplot(grid[0, :2])
    _style(ax, "Динамика вопросов")

    if questions:
        xs = list(range(len(keys)))
        ax.bar(
            xs, answered, color=GREEN, width=0.62,
            label="Ответ найден", edgecolor="none", zorder=3,
        )
        ax.bar(
            xs, not_found, bottom=answered, color=AMBER, width=0.62,
            label="Не найдено", edgecolor="none", zorder=3,
        )
        bottoms = [a + n for a, n in zip(answered, not_found)]
        ax.bar(
            xs, errors, bottom=bottoms, color=RED, width=0.62,
            label="Ошибка", edgecolor="none", zorder=3,
        )

        step = max(1, len(keys) // 12)
        ax.set_xticks(xs[::step])
        ax.set_xticklabels([_bucket_label(k, size) for k in keys[::step]])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        maximum = max(
            (a + n + r for a, n, r in zip(answered, not_found, errors)),
            default=0,
        )
        ax.set_ylim(0, max(1, maximum * 1.2))
        ax.legend(
            loc="upper left",
            frameon=False,
            fontsize=8,
            ncols=3,
            labelcolor=MUTED,
        )
    else:
        _empty(ax)

    # 2. Результат ответов
    ax = fig.add_subplot(grid[0, 2])
    ax.set_facecolor(CARD)
    ax.set_title(
        "Результат ответов",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        pad=18,
    )

    values = [
        (cur["answered"], "Ответ найден", GREEN),
        (cur["not_found"], "Не найдено", AMBER),
        (cur["errors"], "Ошибка", RED),
    ]
    values = [v for v in values if v[0] > 0]

    if values:
        total = sum(v[0] for v in values)
        wedges, _ = ax.pie(
            [v[0] for v in values],
            colors=[v[2] for v in values],
            startangle=90,
            counterclock=False,
            wedgeprops={
                "width": 0.32,
                "edgecolor": CARD,
                "linewidth": 4,
            },
        )
        ax.text(
            0, 0.07, str(total),
            ha="center", va="center",
            fontsize=25, fontweight="bold", color=TEXT,
        )
        ax.text(
            0, -0.19, "вопросов",
            ha="center", va="center",
            fontsize=9, color=MUTED,
        )
        ax.legend(
            wedges,
            [f"{v[1]}  {v[0] / total * 100:.0f}%" for v in values],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.10),
            frameon=False,
            fontsize=8,
            labelcolor=MUTED,
        )
        ax.set_aspect("equal")
    else:
        _empty(ax)

    # 3. Активность по часам
    ax = fig.add_subplot(grid[1, 0])
    _style(ax, "Активность по часам суток")

    if questions:
        hours = Counter(_local(e["ts"]).hour for e in questions)
        hour_values = [hours.get(h, 0) for h in range(24)]

        ax.bar(
            range(24),
            hour_values,
            color=[BLUE if v > 0 else "#E9EDF4" for v in hour_values],
            width=0.65,
            edgecolor="none",
            zorder=3,
        )
        ax.set_xticks(range(0, 24, 3))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(-0.7, 23.7)
    else:
        _empty(ax)

    # 4. Время ответа
    ax = fig.add_subplot(grid[1, 1])
    _style(ax, "Среднее время ответа")

    points = [
        (i, lat_sum[i] / lat_n[i])
        for i in range(len(keys))
        if lat_n[i]
    ]

    if points:
        xs, ys = zip(*points)
        ax.fill_between(xs, ys, color=BLUE, alpha=0.10, zorder=2)
        ax.plot(
            xs, ys,
            color=BLUE,
            linewidth=2.5,
            marker="o",
            markersize=5,
            markerfacecolor=CARD,
            markeredgewidth=2,
            zorder=4,
        )
        ax.set_ylim(bottom=0)

        step = max(1, (len(keys) + 5) // 6)
        tick_positions = list(range(len(keys)))[::step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            [_bucket_label(keys[i], size) for i in tick_positions]
        )
        ax.set_xlim(-0.5, len(keys) - 0.5)

        if cur["latencies"]:
            median = statistics.median(cur["latencies"])
            ax.axhline(
                median,
                color=AMBER,
                linestyle=(0, (4, 3)),
                linewidth=1.5,
                alpha=0.9,
            )
            ax.text(
                0.97, 0.94,
                f"Медиана: {median:.2f} с",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color=AMBER,
                fontweight="bold",
            )
    else:
        _empty(ax)

    # 5. Топ вопросов
    ax = fig.add_subplot(grid[1, 2])
    _style(ax, "Топ-5 популярных вопросов")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)

    top = _top_questions(events, 5)

    if top:
        labels = [_short(t, 34) for t, _, _ in top]
        totals = [n for _, n, _ in top]
        ypos = list(range(len(top)))

        bars = ax.barh(
            ypos,
            totals,
            color=BLUE,
            height=0.52,
            edgecolor="none",
            zorder=3,
        )
        for i, bar in enumerate(bars):
            bar.set_alpha(1 - i * 0.10)

        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        maximum = max(totals)
        ax.set_xlim(0, maximum * 1.25 if maximum else 1)

        for y, n in zip(ypos, totals):
            ax.text(
                n + maximum * 0.025,
                y,
                str(n),
                va="center",
                fontsize=9,
                fontweight="bold",
                color=TEXT,
            )
    else:
        _empty(ax)

    # Footer
    fig.text(
        0.975, 0.025,
        f"Сформировано {_local(time.time()).strftime('%d.%m.%Y %H:%M')} "
        f"· UTC{UTC_OFFSET:+g}",
        ha="right",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=120,
        facecolor=BG,
        bbox_inches="tight",
        pad_inches=0.3,
    )
    return buf.getvalue()


def build_dashboard(period: str) -> tuple[bytes | None, str]:
    start, end, prev_start, prev_end = period_bounds(period)
    events = load_events(start, end)
    first_seen = load_first_seen()
    cur = summarize(events, first_seen, start)
    prev = summarize(load_events(prev_start, prev_end), first_seen, prev_start) if start is not None else None
    first_ts = events[0]["ts"] if events else None
    caption = build_caption(period, cur, prev, start, end, first_ts)
    png = None
    try:
        png = render_dashboard(period, events, cur, start, end, first_ts)
    except Exception:
        log.exception("Не удалось построить графики")
        caption += "\n\n⚠️ Графики сейчас недоступны."
    if len(caption) > CAPTION_LIMIT:
        caption = caption[: CAPTION_LIMIT - 1] + "…"
    return png, caption
