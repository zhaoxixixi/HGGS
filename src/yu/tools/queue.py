"""
@description: A queue implement by using sqlite
"""
from sqlite3 import Connection, connect
from typing import List

from yu.tools.misc import makedir


class SqliteQueue:
    """ A queue implemented using SQLite, not thread-safe. """
    db_path: str
    conn: Connection
    table: str
    counter: int = 0
    MAX_SIZE: int = 200

    def __init__(self, db_path: str, table: str):
        self.db_path = db_path
        self.table = table

    def __enter__(self):
        self.init()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def init(self):
        makedir(self.db_path)
        self.conn = connect(f'{self.db_path}/sqlite')
        cmd = f'''
            SELECT count(*) FROM sqlite_master
            WHERE tbl_name='{self.table}'
            AND type='table';
        '''
        n = self.conn.execute(cmd).fetchone()[0]
        if not n:
            try:
                cmd = f'''
                    CREATE TABLE IF NOT EXISTS {self.table}
                    (ID INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    updated_at DEFAULT (datetime('now','localtime')),
                    disabled BOOLEAN DEFAULT FALSE);
                '''
                self.conn.execute(cmd)
                cmd = f'''
                    create index idx_{self.table}_dsiabled on {self.table}(disabled);
                '''
                self.conn.execute(cmd)
            except Exception as ignored:
                pass
        return self

    def close(self):
        try:
            self._clear_disabled()
            cmd = f'''
                SELECT count(*) FROM {self.table}
                WHERE disabled=FALSE;
            '''
            n = self.conn.execute(cmd).fetchone()[0]
            if not n:
                cmd = f'''DROP TABLE {self.table}'''
                self.conn.execute(cmd)
        except Exception as ignored:
            pass
        self.conn.close()

    def put(self, contents: List[str]):
        cmd = f'''
        INSERT INTO {self.table} (ID, content)
        VALUES (NULL, ?)
        '''
        self.conn.executemany(cmd, [(item,) for item in contents])
        self.conn.commit()

    def get(self):
        try:
            cmd = f'''
                SELECT ID, content, updated_at FROM {self.table}
                WHERE disabled = FALSE LIMIT 1;
            '''
            ret = self.conn.execute(cmd).fetchone()
            if not ret:
                return None
            cmd = f'''
                UPDATE {self.table} SET disabled = TRUE, updated_at = datetime('now','localtime')
                WHERE ID = {ret[0]} AND updated_at = datetime('{ret[2]}');
            '''
            self.conn.execute(cmd)
            self.conn.commit()
            self.counter += 1
            if self.counter >= self.MAX_SIZE:
                self._clear_disabled()
            return ret[1]
        except Exception as ignored:
            pass
        return None

    def _clear_disabled(self):
        """ clear """
        cmd = f'''
            DELETE FROM {self.table} WHERE disabled = TRUE;
        '''
        self.conn.execute(cmd)
        self.conn.commit()

    def empty(self) -> bool:
        """ is empty? """
        cmd = f'''
            SELECT ID FROM {self.table}
            WHERE disabled = FALSE LIMIT 1;
        '''
        return not self.conn.execute(cmd).fetchone()
