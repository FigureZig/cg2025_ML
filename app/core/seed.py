import pandas as pd
import os
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.transaction import Transaction


# --- ТВОЙ КАСТОМНЫЙ ПАРСЕР ---
def custom_date_parser(date_str):
    if pd.isna(date_str): return pd.NaT
    date_str = str(date_str).strip()
    # Перебираем разделители
    for sep in ['/', '.', '-']:
        if sep in date_str:
            parts = date_str.split(sep)
            if len(parts) == 3:
                # Жестко считаем порядок: День, Месяц, Год
                d, m, y = parts
                # Фикс 20xx года
                if len(y) == 2: y = '20' + y
                try:
                    return pd.Timestamp(year=int(y), month=int(m), day=int(d))
                except ValueError:
                    continue
    return pd.NaT


async def seed_data(db: AsyncSession):
    # Проверка на наличие данных (чтобы не дублировать)
    result = await db.execute(select(func.count(Transaction.id)))
    count = result.scalar()
    if count > 0:
        print(f"Database already has {count} transactions. Skipping seed.")
        return

    csv_path = "data/ci_data.csv"
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    try:
        print(f"Reading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        # Маппинг колонок
        col_map = {
            'Date': 'trx_date',
            'Category': 'category',
            'RefNo': 'ref_no',
            'Withdrawal': 'withdrawal',
            'Deposit': 'deposit',
            'Balance': 'balance_after'
        }
        df = df.rename(columns=col_map)

        # --- ПРИМЕНЯЕМ ТВОЙ ФИКС ---
        print("Applying custom date parser...")
        df['trx_date'] = df['trx_date'].apply(custom_date_parser)

        # Дропаем мусор, где дата не распарсилась
        initial_len = len(df)
        df = df.dropna(subset=['trx_date'])
        print(f"Date parsing finished. Rows: {initial_len} -> {len(df)}")

        transactions_to_add = []
        for _, row in df.iterrows():
            # Чистим числа (убираем возможные пробелы, запятые и т.д. если есть)
            try:
                w = float(row.get('withdrawal', 0) if pd.notna(row.get('withdrawal')) else 0)
                d = float(row.get('deposit', 0) if pd.notna(row.get('deposit')) else 0)
                bal = float(row.get('balance_after', 0) if pd.notna(row.get('balance_after')) else 0)
            except ValueError:
                continue

            if w > 0:
                amt, typ = w, 'withdrawal'
            elif d > 0:
                amt, typ = d, 'deposit'
            else:
                continue

            ref = str(row.get('ref_no', ''))

            # Чистим категории
            cat = row.get('category')
            if pd.isna(cat) or str(cat).strip() == '':
                cat = 'Misc'

            transactions_to_add.append(Transaction(
                trx_date=row['trx_date'],
                amount=amt,
                type=typ,
                category=str(cat).strip(),
                ref_no=ref,
                balance_after=bal
            ))

        if transactions_to_add:
            # Сортируем хронологически
            transactions_to_add.sort(key=lambda x: x.trx_date)

            db.add_all(transactions_to_add)
            await db.commit()
            print(f"Successfully seeded {len(transactions_to_add)} transactions.")
        else:
            print("No valid transactions found in CSV after parsing.")

    except Exception as e:
        print(f"Seeding failed: {e}")
        import traceback
        traceback.print_exc()