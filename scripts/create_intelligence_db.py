import argparse
import os
import sqlite3


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def _read_schema(source_conn: sqlite3.Connection):
    cur = source_conn.cursor()
    cur.execute(
        """
        SELECT type, name, sql
        FROM sqlite_master
        WHERE sql IS NOT NULL
          AND name NOT LIKE 'sqlite_%'
        ORDER BY CASE type
            WHEN 'table' THEN 1
            WHEN 'index' THEN 2
            WHEN 'trigger' THEN 3
            WHEN 'view' THEN 4
            ELSE 5 END,
            name
        """
    )
    return cur.fetchall()


def _copy_data(source_conn: sqlite3.Connection, target_conn: sqlite3.Connection):
    src_cur = source_conn.cursor()
    dst_cur = target_conn.cursor()

    src_cur.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    tables = [row[0] for row in src_cur.fetchall()]

    for table_name in tables:
        src_cur.execute(f'PRAGMA table_info("{table_name}")')
        columns = [row[1] for row in src_cur.fetchall()]
        if not columns:
            continue

        quoted_columns = ", ".join(f'"{c}"' for c in columns)
        placeholders = ", ".join(["?"] * len(columns))

        src_cur.execute(f'SELECT {quoted_columns} FROM "{table_name}"')
        rows = src_cur.fetchall()
        if rows:
            dst_cur.executemany(
                f'INSERT INTO "{table_name}" ({quoted_columns}) VALUES ({placeholders})',
                rows,
            )

    target_conn.commit()
    return tables


def create_database_from_reference(source_db: str, target_db: str, include_data: bool = True):
    if not os.path.isfile(source_db):
        raise FileNotFoundError(f"Kaynak DB bulunamadı: {source_db}")

    _ensure_parent_dir(target_db)
    if os.path.exists(target_db):
        os.remove(target_db)

    source_conn = sqlite3.connect(source_db)
    target_conn = sqlite3.connect(target_db)
    try:
        schema_items = _read_schema(source_conn)
        dst_cur = target_conn.cursor()
        for _obj_type, _name, ddl in schema_items:
            dst_cur.execute(ddl)
        target_conn.commit()

        copied_tables = []
        if include_data:
            copied_tables = _copy_data(source_conn, target_conn)

    finally:
        target_conn.close()
        source_conn.close()

    print(f"Source DB: {source_db}")
    print(f"Target DB: {target_db}")
    print(f"Created object count: {len(schema_items)}")
    print(f"Data copied: {include_data}")
    if include_data:
        print(f"Copied table count: {len(copied_tables)}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect IntelligenceData.sqlite schema and create a new DB."
    )
    parser.add_argument(
        "--source",
        default=os.path.join("DataFolder", "IntelligenceData.sqlite"),
        help="Source sqlite file (default: DataFolder/IntelligenceData.sqlite)",
    )
    parser.add_argument(
        "--target",
        default=os.path.join("DataFolder", "IntelligenceData.generated.sqlite"),
        help="Target sqlite file to create",
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Also copy table data. Default behavior is schema-only.",
    )

    args = parser.parse_args()
    create_database_from_reference(
        source_db=args.source,
        target_db=args.target,
        include_data=args.with_data,
    )


if __name__ == "__main__":
    main()
