import sqlite3
import os
import numpy as np

# ── Built-in name-based gender classifier (no external packages needed) ──────
# Common female first names
_FEMALE_NAMES = {
    'mary','patricia','jennifer','linda','barbara','elizabeth','susan','jessica','sarah',
    'karen','lisa','nancy','betty','margaret','sandra','ashley','emily','dorothy','kimberly',
    'carol','michelle','amanda','melissa','deborah','stephanie','rebecca','sharon','laura',
    'cynthia','kathleen','amy','angela','shirley','anna','brenda','pamela','emma','nicole',
    'helen','samantha','katherine','christine','debra','rachel','carolyn','janet','catherine',
    'maria','heather','diane','ruth','julie','olivia','joyce','virginia','victoria','kelly',
    'lauren','christina','joan','evelyn','judith','megan','cheryl','andrea','hannah','martha',
    'jacqueline','lisa','frances','ann','gloria','theresa','kathryn','sara','janice','alice',
    'jean','danielle','abigail','julia','natalie','grace','brittany','amber','rose','beverly',
    'alexis','tiffany','kayla','crystal','autumn','brianna','vanessa','miranda','denise',
    'tina','renee','alicia','tamara','april','leah','dawn','jane','taylor','madison','sophia',
    'isabella','charlotte','amelia','mia','harper','evelyn','luna','ellie','chloe','penelope',
    'layla','riley','zoey','nora','lily','eleanor','hannah','aubrey','anna','lucy','zoe',
    'violet','scarlett','ivy','stella','aurora','bella','claire','skylar','queen','elena',
    'ada','bertha','clara','daisy','dolly','edith','eleanor','esther','ethel','flora','grace',
    'harriet','hattie','hazel','ida','irene','irma','josephine','julia','lillian','lilly',
    'lola','lottie','louise','mabel','maud','maude','millie','minnie','myrtle','nellie',
    'nettie','norma','opal','ora','pearl','pearly','phoebe','phyllis','priscilla','rhonda',
    'roberta','ruby','ruthie','sallie','sienna','sylvia','vera','viola','vivian','wanda',
    'wilma','winona','caroline','margaret','natasha','anastasia','katya','lena','irina',
    'tatiana','olga','svetlana','natalya','marina','elena','oksana','valeria','daria',
    'fatima','aisha','zainab','amina','layla','nadia','yasmin','samira','leila','rania',
    'priya','anita','sunita','kavita','pooja','neha','rai','anjali','divya','meena',
    'sita','rani','nisha','rekha','geeta','ritu','mona','radha','asha',
}

# Common male first names
_MALE_NAMES = {
    'james','john','robert','michael','william','david','richard','joseph','thomas','charles',
    'christopher','daniel','matthew','anthony','mark','donald','steven','paul','andrew','joshua',
    'kenneth','kevin','brian','george','timothy','ronald','edward','jason','jeffrey','ryan',
    'jacob','gary','nicholas','eric','jonathan','stephen','larry','justin','scott','brandon',
    'benjamin','samuel','raymond','gregory','frank','alexander','raymond','patrick','jack',
    'dennis','jerry','tyler','aaron','jose','henry','adam','douglas','nathan','peter','zachary',
    'kyle','walter','harold','jeremy','ethan','carl','keith','roger','gerald','christian',
    'terry','sean','arthur','austin','noah','lawrence','jesse','joe','bryan','billy','jordan',
    'albert','dylan','bruce','willie','wayne','alan','juan','elijah','roy','bobby','clarence',
    'russell','vincent','phillip','louis','gabriel','harry','liam','mason','oliver','logan',
    'lucas','caleb','owen','carter','wyatt','julian','leo','isaac','evan','nolan','elias',
    'muhammad','ahmed','ali','omar','khalid','hassan','ibrahim','yusuf','tariq','amir',
    'raj','ravi','suresh','ramesh','kumar','arun','vijay','sunil','manoj','deepak',
    'amit','rohit','sanjay','rajesh','naresh','ashok','vinod','mohan','prakash','girish',
    'ivan','dmitri','alexei','andrei','sergei','nikolai','vladimir','mikhail','boris','igor',
    'pierre','jean','claude','francois','alain','philippe','thierry','patrick','dominique',
    'liam','noah','oliver','william','elijah','james','benjamin','lucas','henry','alexander',
    'mason','ethan','daniel','jacob','logan','jackson','sebastian','jack','aiden','owen',
    'samuel','joseph','john','david','wyatt','matthew','luke','asher','carter','julian',
    'gavin','levi','isaac','anthony','dylan','lincoln','gabriel','grayson','leo','ryan',
    'thomas','charlie','caleb','christopher','jaxon','dominic','nolan','hunter','cameron',
    'aaron','eli','colton','landon','adam','joshua','easton','sawyer','evan','brayden',
    'scarface','al','george','clinton','bush','obama','trump','biden','boris','tony',
    'adolf','joseph','mao','napoleon','winston','franklin','abraham','theodore','woodrow',
}


def guess_gender(name):
    """Guess gender from a person's full name using first-name lookup.
    Returns 'male', 'female', or 'unknown'.
    """
    if not name:
        return 'unknown'
    # LFW names are typically 'First_Last' format
    first = name.replace('_', ' ').split()[0].lower().strip()
    if first in _FEMALE_NAMES:
        return 'female'
    if first in _MALE_NAMES:
        return 'male'
    return 'unknown'
# ─────────────────────────────────────────────────────────────────────────────


def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS criminals (
                 id INTEGER PRIMARY KEY,
                 name TEXT,
                 filepath TEXT,
                 encoding BLOB,
                 gender TEXT DEFAULT \'unknown\'
                 )''')
    # Migration: add gender column if it doesn't exist yet
    try:
        c.execute('ALTER TABLE criminals ADD COLUMN gender TEXT DEFAULT \'unknown\'')
        print('[DB] Migrated: added gender column.')
    except Exception:
        pass  # Column already exists
    conn.commit()
    conn.close()

# --- Tutorial / Step-by-Step Wrapper Functions ---
def connect_db():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'criminals.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)

def create_table():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'criminals.db')
    init_db(db_path)

def insert_data(name, image_path, embedding):
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'criminals.db')
    add_criminal(db_path, name, image_path, embedding)

def fetch_data():
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'criminals.db')
    return get_all_criminals(db_path)
# ---------------------------------------------------


def add_criminal(db_path, name, filepath, encoding):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        enc = np.asarray(encoding, dtype='float64')
        shape = np.array(enc.shape).astype('int32')
        blob = shape.tobytes() + enc.tobytes()
    except Exception:
        blob = None
    gender = guess_gender(name)
    c.execute('INSERT INTO criminals (name, filepath, encoding, gender) VALUES (?,?,?,?)', (name, filepath, blob, gender))
    conn.commit()
    conn.close()


def populate_gender_column(db_path):
    """One-time migration: fill the gender column for all existing rows using name heuristics."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name FROM criminals WHERE gender IS NULL OR gender = \'unknown\'')
    rows = c.fetchall()
    updated = 0
    for row_id, name in rows:
        g = guess_gender(name)
        c.execute('UPDATE criminals SET gender=? WHERE id=?', (g, row_id))
        updated += 1
    conn.commit()
    conn.close()
    print(f'[DB] Gender populated for {updated} records.')
    return updated


def delete_criminal(db_path, record_id):
    """Delete a single record by id. Returns True if a row was deleted."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DELETE FROM criminals WHERE id=?', (record_id,))
    deleted = c.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_db_stats(db_path):
    """Return quick summary stats about the database."""
    if not os.path.exists(db_path):
        return {'total': 0, 'with_encoding': 0, 'size_mb': 0}
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM criminals')
    total = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM criminals WHERE encoding IS NOT NULL')
    with_enc = c.fetchone()[0]
    conn.close()
    size_mb = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    return {'total': total, 'with_encoding': with_enc, 'size_mb': size_mb}


def search_criminals_by_name(db_path, query, limit=20):
    """Simple LIKE search on the name field (no encoding needed)."""
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute(
        'SELECT id, name, filepath FROM criminals WHERE name LIKE ? LIMIT ?',
        (f'%{query}%', limit)
    )
    rows = c.fetchall()
    conn.close()
    return [{'id': r[0], 'name': r[1], 'filepath': r[2]} for r in rows]


def get_paginated_criminals(db_path, page=1, per_page=20):
    """Return a page of records (without encoding blobs for speed)."""
    if not os.path.exists(db_path):
        return [], 0
    offset = (page - 1) * per_page
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM criminals')
    total = c.fetchone()[0]
    c.execute(
        'SELECT id, name, filepath FROM criminals ORDER BY id LIMIT ? OFFSET ?',
        (per_page, offset)
    )
    rows = c.fetchall()
    conn.close()
    records = [{'id': r[0], 'name': r[1], 'filepath': r[2]} for r in rows]
    return records, total


def _decode_encoding(blob):
    if blob is None:
        return None
    try:
        if len(blob) > 4:
            return np.frombuffer(blob[4:], dtype='float64')
        return None
    except Exception:
        return None


def get_all_criminals(db_path):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name, filepath, encoding, gender FROM criminals')
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        enc = _decode_encoding(r[3])
        out.append({'id': r[0], 'name': r[1], 'filepath': r[2], 'encoding': enc, 'gender': r[4] or 'unknown'})
    return out
