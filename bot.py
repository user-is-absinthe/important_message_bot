import os
import re
import io
import asyncio
import logging
import html
from datetime import datetime
from typing import List, Optional, Dict

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    InlineQuery,
    InlineQueryResultCachedPhoto,
    CallbackQuery,
    BufferedInputFile,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.fsm.context import FSMContext

import aiosqlite
from PIL import Image

import pymorphy3
morph = pymorphy3.MorphAnalyzer()

# --------- SETTINGS ----------
try:
    import settings  # файл settings.py должен лежать рядом с bot.py
except ImportError as e:
    raise RuntimeError(
        "Не найден файл settings.py рядом с bot.py. "
        "Создайте его по образцу из инструкции."
    ) from e

BOT_TOKEN = getattr(settings, "BOT_TOKEN", None)
CHANNEL_ID = int(getattr(settings, "CHANNEL_ID", 0))
DB_PATH = getattr(settings, "DB_PATH", "photos.db")

if not BOT_TOKEN or not CHANNEL_ID:
    raise RuntimeError("Укажите BOT_TOKEN и CHANNEL_ID в settings.py")

# --------- LOGGING ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inline-photo-bot")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

# --------- DB HELPERS ----------
CREATE_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    username TEXT,
    first_name TEXT,
    last_name TEXT,
    is_admin INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id TEXT NOT NULL,
    file_unique_id TEXT NOT NULL,
    channel_id INTEGER NOT NULL,
    channel_message_id INTEGER NOT NULL,
    uploader_user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    deleted_at TEXT
);

CREATE TABLE IF NOT EXISTS image_tags (
    image_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    UNIQUE(image_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_images_uploader ON images(uploader_user_id);
CREATE INDEX IF NOT EXISTS idx_images_unique ON images(file_unique_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON image_tags(tag);
"""

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        for stmt in CREATE_SQL.strip().split(";\n"):
            if stmt.strip():
                await db.execute(stmt)
        await db.commit()

def normalize_tags(text: Optional[str]) -> List[str]:
    if not text:
        return []
    raw = re.split(r"[,\n;]+|\s+", text.strip().lower())
    tags = []
    for t in raw:
        t = t.strip()
        if not t:
            continue
        if t.startswith("#"):
            t = t[1:]
        t = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9_\-]+", "", t)
        if t:
            tags.append(t)
    seen = set()
    uniq = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

async def upsert_user(m: Message):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO users(user_id, username, first_name, last_name)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET
                 username=excluded.username,
                 first_name=excluded.first_name,
                 last_name=excluded.last_name""",
            (m.from_user.id, m.from_user.username, m.from_user.first_name, m.from_user.last_name),
        )
        await db.commit()

async def get_user(user_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

async def add_image_record(
    file_id: str,
    file_unique_id: str,
    uploader_user_id: int,
    channel_id: int,
    channel_message_id: int,
    tags: List[str],
):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            """INSERT INTO images(file_id, file_unique_id, channel_id, channel_message_id, uploader_user_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (file_id, file_unique_id, channel_id, channel_message_id, uploader_user_id, datetime.utcnow().isoformat()),
        )
        image_id = cur.lastrowid
        if tags:
            await db.executemany(
                "INSERT OR IGNORE INTO image_tags(image_id, tag) VALUES (?, ?)",
                [(image_id, t) for t in tags],
            )
        await db.commit()
        return image_id

async def get_image_by_unique(file_unique_id: str) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM images WHERE file_unique_id=? AND deleted_at IS NULL", (file_unique_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

async def get_image_by_id(image_id: int) -> Optional[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM images WHERE id=? AND deleted_at IS NULL", (image_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

async def get_image_row_any(image_id: int) -> Optional[dict]:
    """Достаём запись даже если она помечена удалённой (для правки подписи в канале)."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM images WHERE id=?", (image_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

async def get_image_tags(image_id: int) -> List[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT tag FROM image_tags WHERE image_id=?", (image_id,))
        rows = await cur.fetchall()
        return [r[0] for r in rows]

async def set_image_tags(image_id: int, tags: List[str]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM image_tags WHERE image_id=?", (image_id,))
        if tags:
            await db.executemany(
                "INSERT OR IGNORE INTO image_tags(image_id, tag) VALUES (?, ?)",
                [(image_id, t) for t in tags],
            )
        await db.commit()

async def soft_delete_image(image_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("UPDATE images SET deleted_at=? WHERE id=? AND deleted_at IS NULL", (datetime.utcnow().isoformat(), image_id))
        await db.commit()

async def search_images_for_user(user_id: int, tags: List[str], limit: int, offset: int) -> List[dict]:
    if not tags:
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT * FROM images
                   WHERE uploader_user_id=? AND deleted_at IS NULL
                   ORDER BY id DESC LIMIT ? OFFSET ?""",
                (user_id, limit, offset),
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]
    else:
        placeholders = ",".join("?" for _ in tags)
        sql = f"""
            SELECT i.*
            FROM images i
            JOIN image_tags t ON t.image_id = i.id
            WHERE i.uploader_user_id=? AND i.deleted_at IS NULL AND t.tag IN ({placeholders})
            GROUP BY i.id
            HAVING COUNT(DISTINCT t.tag) = ?
            ORDER BY i.id DESC
            LIMIT ? OFFSET ?
        """
        args = [user_id, *tags, len(tags), limit, offset]
        async with aiosqlite.connect(DB_PATH) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(sql, args)
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

async def search_images_for_user_fts(user_id: int, query: str, limit: int, offset: int) -> List[dict]:
    base_tokens = normalize_tags(query)
    if not base_tokens:
        return await search_images_for_user(user_id, [], limit, offset)

    lemmas = lemmatize_tokens(base_tokens)

    groups = []
    for i, t in enumerate(base_tokens):
        lem = lemmas[i] if i < len(lemmas) else t
        t = t.lower()
        lem = lem.lower()
        groups.append(f'({t}* OR {lem}*)' if lem != t else f'{t}*')

    match = " AND ".join(groups)

    sql_bm25 = """
    SELECT i.*
    FROM image_fts
    JOIN images AS i ON i.id = image_fts.rowid
    WHERE image_fts MATCH ?
      AND i.uploader_user_id = ?
      AND i.deleted_at IS NULL
    ORDER BY bm25(image_fts), i.id DESC
    LIMIT ? OFFSET ?
    """

    sql_no_bm25 = """
    SELECT i.*
    FROM image_fts
    JOIN images AS i ON i.id = image_fts.rowid
    WHERE image_fts MATCH ?
      AND i.uploader_user_id = ?
      AND i.deleted_at IS NULL
    ORDER BY i.id DESC
    LIMIT ? OFFSET ?
    """

    import sqlite3
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        try:
            cur = await db.execute(sql_bm25, (match, user_id, limit, offset))
            rows = await cur.fetchall()
        except sqlite3.OperationalError as e:
            # Фолбэк, если в сборке SQLite нет bm25()
            if "no such function: bm25" in str(e):
                cur = await db.execute(sql_no_bm25, (match, user_id, limit, offset))
                rows = await cur.fetchall()
            else:
                raise
        return [dict(r) for r in rows]

# ---------- FTS5 ----------
CREATE_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS image_fts USING fts5(
    content,
    tokenize = "unicode61 remove_diacritics 2 tokenchars '-_'"
);
"""

async def init_fts():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_FTS_SQL)
        await db.commit()

def lemmatize_tokens(tokens: List[str]) -> List[str]:
    """Лемматизируем русские токены, прочие — оставляем как есть."""
    lemmas = []
    for t in tokens:
        # лемматизируем только кириллицу (включая ё/Ё и дефисы/нижние подчёркивания)
        if re.fullmatch(r"[а-яА-ЯёЁ][а-яА-ЯёЁ0-9_\-]*", t):
            try:
                p = morph.parse(t)[0]
                lem = p.normal_form
                if lem:
                    lemmas.append(lem.lower())
                    continue
            except Exception:
                pass
        lemmas.append(t.lower())
    # убираем дубли, сохраняя порядок
    seen, out = set(), []
    for x in lemmas:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def build_index_text(tags: List[str], extra: str = "") -> str:
    """В индекс кладём оригинальные токены + леммы."""
    tags = [t.lower() for t in tags]
    lemmas = lemmatize_tokens(tags)
    # порядок: сначала оригиналы, потом леммы (без дублей)
    all_tokens = []
    seen = set()
    for t in tags + lemmas:
        if t and t not in seen:
            seen.add(t)
            all_tokens.append(t)
    if extra:
        all_tokens.append(extra)
    return " ".join(all_tokens).strip()

async def fts_upsert(image_id: int, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        # стандартный delete → insert вместо спец-команды FTS5
        await db.execute("DELETE FROM image_fts WHERE rowid = ?", (image_id,))
        await db.execute("INSERT INTO image_fts(rowid, content) VALUES(?, ?)", (image_id, content))
        await db.commit()

async def fts_delete(image_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM image_fts WHERE rowid = ?", (image_id,))
        await db.commit()

async def ensure_fts_backfilled():
    """На старте: если FTS пуст/неполон — пересобрать индекс. Работает в одном соединении."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # сколько актуальных изображений
        c1 = await db.execute("SELECT COUNT(*) AS c FROM images WHERE deleted_at IS NULL")
        total = (await c1.fetchone())["c"]

        # сколько записей в FTS
        try:
            c2 = await db.execute("SELECT COUNT(*) AS c FROM image_fts")
            total_fts = (await c2.fetchone())["c"]
        except Exception:
            total_fts = 0

        if total_fts >= total:
            return  # индекс уже в норме

        # достраиваем индекс
        cur = await db.execute("SELECT id FROM images WHERE deleted_at IS NULL ORDER BY id")
        rows = await cur.fetchall()
        for r in rows:
            image_id = r["id"]
            tcur = await db.execute("SELECT tag FROM image_tags WHERE image_id=?", (image_id,))
            tags = [x[0] for x in await tcur.fetchall()]

            # upsert: удалить старую строку и вставить новую
            await db.execute("DELETE FROM image_fts WHERE rowid = ?", (image_id,))
            await db.execute(
                "INSERT INTO image_fts(rowid, content) VALUES(?, ?)",
                (image_id, build_index_text(tags))
            )
        await db.commit()

# --------- STATE FOR EDITING TAGS ----------
class EditStates(StatesGroup):
    waiting_new_tags = State()

# кеш тегов для медиа-групп (альбомов)
media_group_tags_cache: Dict[str, List[str]] = {}

# --------- HELPERS ----------
def tags_to_caption(tags: List[str]) -> str:
    return " ".join(f"#{t}" for t in tags) if tags else ""

def escape(s: Optional[str]) -> str:
    return html.escape(s or "")

def build_channel_caption_block(user_id: int, first_name: Optional[str], last_name: Optional[str], username: Optional[str]) -> str:
    # HTML parse_mode: используем <pre> ... </pre> блок, как вы просили
    uname = f"@{username}" if username else "—"
    return (
        f"<pre>"
        f"ID пользователя: {escape(str(user_id))}\n"
        f"Имя: {escape(first_name or '')} {escape(last_name or '')}\n"
        f"Ссылка: {escape(uname)}"
        f"</pre>"
    )

def compose_channel_caption(user_id: int, first_name: Optional[str], last_name: Optional[str], username: Optional[str],
                            tags: List[str], deleted_at: Optional[str] = None) -> str:
    header = build_channel_caption_block(user_id, first_name, last_name, username)
    tags_line = tags_to_caption(tags)
    deleted_line = f"\n<i>❌ Удалено пользователем {escape(deleted_at)}</i>" if deleted_at else ""
    return header + (("\n\n" + tags_line) if tags_line else "") + deleted_line

async def update_channel_caption(image_row: dict, tags: Optional[List[str]] = None, mark_deleted: bool = False):
    """Пересобираем подпись и обновляем её в канале."""
    user = await get_user(image_row["uploader_user_id"])
    if not user:
        # на всякий случай — возьмём пустые значения
        user = {"first_name": "", "last_name": "", "username": None}
    if tags is None:
        tags = await get_image_tags(image_row["id"])
    deleted_at = None
    if mark_deleted:
        deleted_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    caption = compose_channel_caption(
        image_row["uploader_user_id"], user.get("first_name"), user.get("last_name"), user.get("username"),
        tags, deleted_at
    )
    try:
        await bot.edit_message_caption(
            chat_id=image_row["channel_id"],
            message_id=image_row["channel_message_id"],
            caption=caption,
            parse_mode="HTML",
        )
    except Exception as e:
        logger.warning(f"Failed to edit channel caption: {e}")

def action_keyboard(image_id: int) -> types.InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="✏️ Редактировать теги", callback_data=f"edit:{image_id}")
    kb.button(text="🧹 Очистить теги", callback_data=f"clear:{image_id}")
    kb.button(text="🗑 Удалить", callback_data=f"del:{image_id}")
    kb.adjust(2, 1)  # 1-й ряд: редактировать/очистить; 2-й ряд: удалить
    return kb.as_markup()

async def send_photo_to_channel_from_file_id(
    source_file_id: str,
    uploader: types.User,
    tags: List[str],
) -> types.Message:
    caption = compose_channel_caption(
        uploader.id, uploader.first_name, uploader.last_name, uploader.username, tags
    )
    return await bot.send_photo(
        chat_id=CHANNEL_ID,
        photo=source_file_id,
        caption=caption,
        parse_mode="HTML",
    )

async def send_photo_to_channel_from_bytes(
    image_bytes: bytes,
    uploader: types.User,
    tags: List[str],
    filename: str = "image.jpg",
) -> types.Message:
    caption = compose_channel_caption(
        uploader.id, uploader.first_name, uploader.last_name, uploader.username, tags
    )
    return await bot.send_photo(
        chat_id=CHANNEL_ID,
        photo=BufferedInputFile(image_bytes, filename=filename),
        caption=caption,
        parse_mode="HTML",
    )

async def add_image_tags(image_id: int, tags: List[str]):
    """Добавляет новые теги, старые остаются (дубликаты игнорируются)."""
    if not tags:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR IGNORE INTO image_tags(image_id, tag) VALUES (?, ?)",
            [(image_id, t) for t in tags],
        )
        await db.commit()

async def clear_image_tags(image_id: int):
    """Полностью очищает теги картинки."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM image_tags WHERE image_id=?", (image_id,))
        await db.commit()

# --------- COMMANDS ----------
@dp.message(CommandStart())
async def cmd_start(m: Message):
    await upsert_user(m)
    await m.answer(
        "Привет! Я помогу хранить твои картинки с тегами.\n\n"
        "Как пользоваться:\n"
        "• Пришли фото с тегами в подписи — я сохраню их.\n"
        "• В inline-режиме (набор @бот в любом чате) ищи по своим тегам.\n"
        "• Чтобы отредактировать/удалить — пришли картинку в чат со мной через inline, я покажу кнопки."
    )

@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(
        "Добавление: пришли фото(а) с подписью вида: #кот, хвост, пушистый\n"
        "Поиск: в любом чате наберите @meetmemebot и теги\n"
        "Редактирование/удаление: пришлите найденную inline картинку сюда — появятся кнопки."
    )

# --------- ADDING PHOTOS (DIRECT CHAT) ----------
@dp.message(F.photo)
async def on_photo(m: Message):
    await upsert_user(m)

    # Если фото прислали в чат с ботом через inline этого же бота — не сохраняем, просто показываем меню
    me = await bot.get_me()
    if m.via_bot and m.via_bot.id == me.id:
        unique = m.photo[-1].file_unique_id
        row = await get_image_by_unique(unique)
        if not row:
            await m.reply("Я не нашёл эту картинку в базе (возможно, она удалена).")
            return
        if row["uploader_user_id"] != m.from_user.id:
            await m.reply("Это изображение принадлежит другому пользователю.")
            return
        user_tags = await get_image_tags(row["id"])
        tags_line = tags_to_caption(user_tags) or "—"
        await m.reply(f"Ваши теги: {tags_line}\nВыберите действие:", reply_markup=action_keyboard(row["id"]))
        return

    # Обычный сценарий добавления (НЕ через inline)
    tags = normalize_tags(m.caption)
    if m.media_group_id:
        gid = str(m.media_group_id)
        if tags:
            media_group_tags_cache[gid] = tags
        else:
            tags = media_group_tags_cache.get(gid, [])
    src_file_id = m.photo[-1].file_id
    sent = await send_photo_to_channel_from_file_id(src_file_id, m.from_user, tags)
    saved_photo = sent.photo[-1]
    image_id = await add_image_record(
        file_id=saved_photo.file_id,
        file_unique_id=saved_photo.file_unique_id,
        uploader_user_id=m.from_user.id,
        channel_id=CHANNEL_ID,
        channel_message_id=sent.message_id,
        tags=tags,
    )
    await fts_upsert(image_id, build_index_text(tags))
    await m.reply(
        "Сохранил ✅\n"
        f"Тегов: {len(tags)}\n"
        f"Подсказка: открой inline (@{(await bot.get_me()).username}) и введи теги для поиска.",
        reply_markup=action_keyboard(image_id),
    )

# --------- ADD FROM STICKER ----------
@dp.message(F.sticker)
async def on_sticker(m: Message):
    await upsert_user(m)
    tags = normalize_tags(m.caption)  # вдруг подпись есть
    sticker = m.sticker

    async def save_message_as_image_bytes(msg: Message, image_bytes: bytes, fname: str):
        sent = await send_photo_to_channel_from_bytes(image_bytes, msg.from_user, tags, filename=fname)
        saved_photo = sent.photo[-1]
        image_id = await add_image_record(
            file_id=saved_photo.file_id,
            file_unique_id=saved_photo.file_unique_id,
            uploader_user_id=msg.from_user.id,
            channel_id=CHANNEL_ID,
            channel_message_id=sent.message_id,
            tags=tags,
        )
        await msg.reply(
            "Сохранил из стикера ✅",
            reply_markup=action_keyboard(image_id),
        )

    try:
        if not sticker.is_animated and not sticker.is_video:
            # статичный стикер (.webp/.png) — конвертим в JPEG
            file = await bot.get_file(sticker.file_id)
            buf = io.BytesIO()
            await bot.download_file(file.file_path, destination=buf)
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
            out = io.BytesIO()
            img.save(out, format="JPEG", quality=95)
            await save_message_as_image_bytes(m, out.getvalue(), "sticker.jpg")
            return
        else:
            # анимированный/видео — пробуем взять превью (thumbnail)
            thumb = getattr(sticker, "thumbnail", None) or getattr(sticker, "thumb", None)
            if thumb:
                sent = await send_photo_to_channel_from_file_id(thumb.file_id, m.from_user, tags)
                saved_photo = sent.photo[-1]
                image_id = await add_image_record(
                    file_id=saved_photo.file_id,
                    file_unique_id=saved_photo.file_unique_id,
                    uploader_user_id=m.from_user.id,
                    channel_id=CHANNEL_ID,
                    channel_message_id=sent.message_id,
                    tags=tags,
                )
                await m.reply("Сохранил превью стикера ✅", reply_markup=action_keyboard(image_id))
                return
            else:
                await m.reply("Не могу обработать такой файл (анимированный/видео-стикер без превью).")
                return
    except Exception as e:
        logger.warning(f"Sticker processing failed: {e}")
        await m.reply("Не могу обработать такой файл (ошибка конвертации).")

# --------- INLINE QUERY (SEARCH) ----------
@dp.inline_query()
async def on_inline_query(q: InlineQuery):
    user_id = q.from_user.id
    query = q.query or ""
    limit = 25
    offset = int(q.offset) if q.offset and q.offset.isdigit() else 0

    rows = await search_images_for_user_fts(user_id, query, limit=limit, offset=offset)
    results = [
        InlineQueryResultCachedPhoto(
            id=str(r["id"]),
            photo_file_id=r["file_id"],
        ) for r in rows
    ]
    next_offset = str(offset + len(results)) if len(results) == limit else ""
    await q.answer(results=results, cache_time=0, is_personal=True, next_offset=next_offset)

# --------- RECEIVING INLINE MESSAGE IN BOT CHAT (FOR EDIT/DELETE) ----------
@dp.message(F.via_bot)
async def on_message_via_bot(m: Message):
    me = await bot.get_me()
    if not m.via_bot or m.via_bot.id != me.id:
        return

    # Фото из inline обрабатывает on_photo (показывает меню и не сохраняет)
    if m.photo:
        return

    # Оставим обработку для случая, если вдруг прилетит документ-картинка через inline
    if m.document and m.document.mime_type and m.document.mime_type.startswith("image/"):
        unique = m.document.file_unique_id
    else:
        return

    row = await get_image_by_unique(unique)
    if not row:
        await m.reply("Я не нашёл эту картинку в базе (возможно, она удалена).")
        return
    if row["uploader_user_id"] != m.from_user.id:
        await m.reply("Это изображение принадлежит другому пользователю.")
        return

    user_tags = await get_image_tags(row["id"])
    tags_line = tags_to_caption(user_tags) or "—"
    await m.reply(
        f"Ваши теги: {tags_line}\nВыберите действие:",
        reply_markup=action_keyboard(row["id"])
    )

# --------- EDIT TAGS ----------
@dp.callback_query(F.data.startswith("edit:"))
async def cb_edit(c: CallbackQuery, state: FSMContext):
    image_id = int(c.data.split(":")[1])
    row = await get_image_by_id(image_id)
    if not row:
        await c.answer("Не найдено/удалено.", show_alert=True)
        return
    if row["uploader_user_id"] != c.from_user.id:
        await c.answer("Недостаточно прав.", show_alert=True)
        return

    old_tags = await get_image_tags(image_id)
    await state.set_state(EditStates.waiting_new_tags)
    await state.update_data(image_id=image_id)
    await c.message.answer(
        f"Текущие теги: {tags_to_caption(old_tags) or '—'}\n"
        "Пришлите новые теги (через пробелы/запятые, можно с #)."
    )
    await c.answer()

@dp.message(EditStates.waiting_new_tags)
async def on_new_tags(m: Message, state: FSMContext):
    data = await state.get_data()
    image_id = data.get("image_id")
    row = await get_image_by_id(image_id)
    if not row:
        await m.reply("Картинка не найдена (возможно, удалена).")
        await state.clear()
        return
    if row["uploader_user_id"] != m.from_user.id:
        await m.reply("Недостаточно прав.")
        await state.clear()
        return

    new_tags = normalize_tags(m.text or "")
    await add_image_tags(image_id, new_tags)  # <-- добавляем, не затираем
    all_tags = await get_image_tags(image_id)
    await fts_upsert(image_id, build_index_text(all_tags))

    # обновим подпись в канале
    full_row = await get_image_row_any(image_id)
    if full_row:
        await update_channel_caption(full_row, tags=all_tags, mark_deleted=False)

    added_line = tags_to_caption(new_tags) or "—"
    total_line = tags_to_caption(all_tags) or "—"
    await m.reply(
        f"Готово. Добавил: {added_line}\nИтоговые теги: {total_line}",
        reply_markup=action_keyboard(image_id)
    )
    await state.clear()

@dp.callback_query(F.data.startswith("clear:"))
async def cb_clear(c: CallbackQuery):
    image_id = int(c.data.split(":")[1])
    row = await get_image_by_id(image_id)
    if not row:
        await c.answer("Не найдено/удалено.", show_alert=True)
        return
    if row["uploader_user_id"] != c.from_user.id:
        await c.answer("Недостаточно прав.", show_alert=True)
        return

    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, очистить", callback_data=f"clearc:{image_id}:yes")
    kb.button(text="↩️ Отмена", callback_data=f"clearc:{image_id}:no")
    kb.adjust(2)
    await c.message.answer("Очистить все теги у изображения?", reply_markup=kb.as_markup())
    await c.answer()

@dp.callback_query(F.data.startswith("clearc:"))
async def cb_clear_confirm(c: CallbackQuery):
    _, image_id_str, choice = c.data.split(":")
    image_id = int(image_id_str)
    if choice == "no":
        await c.answer("Отменено.")
        await c.message.edit_text("Очистка отменена.")
        return

    row = await get_image_by_id(image_id)
    if not row:
        await c.answer("Не найдено/удалено.", show_alert=True)
        return
    if row["uploader_user_id"] != c.from_user.id:
        await c.answer("Недостаточно прав.", show_alert=True)
        return

    await clear_image_tags(image_id)
    await fts_upsert(image_id, build_index_text([]))
    full_row = await get_image_row_any(image_id)
    if full_row:
        await update_channel_caption(full_row, tags=[], mark_deleted=False)

    await c.message.edit_text("Теги очищены. Теперь у изображения нет тегов.")
    await c.answer("Готово.")

# --------- DELETE (SOFT; KEEP IN CHANNEL WITH MARK) ----------
@dp.callback_query(F.data.startswith("del:"))
async def cb_delete(c: CallbackQuery):
    image_id = int(c.data.split(":")[1])
    row = await get_image_by_id(image_id)
    if not row:
        await c.answer("Не найдено/уже удалено.", show_alert=True)
        return
    if row["uploader_user_id"] != c.from_user.id:
        await c.answer("Недостаточно прав.", show_alert=True)
        return

    kb = InlineKeyboardBuilder()
    kb.button(text="✅ Да, удалить", callback_data=f"delc:{image_id}:yes")
    kb.button(text="↩️ Отмена", callback_data=f"delc:{image_id}:no")
    kb.adjust(2)
    await c.message.answer("Удалить изображение?", reply_markup=kb.as_markup())
    await c.answer()

@dp.callback_query(F.data.startswith("delc:"))
async def cb_delete_confirm(c: CallbackQuery):
    _, image_id_str, choice = c.data.split(":")
    image_id = int(image_id_str)
    if choice == "no":
        await c.answer("Отменено.")
        await c.message.edit_text("Удаление отменено.")
        return

    row = await get_image_by_id(image_id)
    if not row:
        await c.answer("Не найдено/уже удалено.", show_alert=True)
        return
    if row["uploader_user_id"] != c.from_user.id:
        await c.answer("Недостаточно прав.", show_alert=True)
        return

    # мягко помечаем удаление (в БД) и правим подпись в канале
    await soft_delete_image(image_id)
    full_row = await get_image_row_any(image_id)
    if full_row:
        await update_channel_caption(full_row, tags=None, mark_deleted=True)

    await c.message.edit_text("Элемент удален.")
    await c.answer("Готово.")

# --------- FALLBACK: неподдерживаемый контент ----------
@dp.message()
async def fallback(m: Message):
    # Отвечаем только на вложения, которые мы не обрабатываем (текст игнорируем)
    if m.content_type not in ("text", "photo", "sticker"):
        # Если документ с картинкой — можно расширить поддержку; пока выводим отказ
        await m.reply("Не могу обработать такой файл.")
        return

# --------- MAIN ----------
async def main():
    await init_db()
    await init_fts()
    await ensure_fts_backfilled()  # пройдётся по существующим и заполнит FTS

    me = await bot.get_me()
    logger.info(f"Bot started as @{me.username}")
    await dp.start_polling(bot, allowed_updates=[
        "message", "inline_query", "callback_query", "chosen_inline_result", "channel_post"
    ])

if __name__ == "__main__":
    asyncio.run(main())
