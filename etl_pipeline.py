import argparse
import re
from difflib import get_close_matches
from typing import Iterable, Set

import pandas as pd
from sqlalchemy import create_engine, text


CANONICAL_CATALOGS = ["Sports", "Pets", "Toys", "Gardening", "Software", "Collectibles"]
CATALOG_MAP = {
    "sport": "Sports",
    "sports": "Sports",
    "sprots": "Sports",
    "sporst": "Sports",
    "spots": "Sports",
    "sposts": "Sports",
    "pet": "Pets",
    "pets": "Pets",
    "pest": "Pets",
    "toy": "Toys",
    "toys": "Toys",
    "tosy": "Toys",
    "gardening": "Gardening",
    "gardenings": "Gardening",
    "garden": "Gardening",
    "gardning": "Gardening",
    "software": "Software",
    "softwares": "Software",
    "softwars": "Software",
    "softwar": "Software",
    "collectible": "Collectibles",
    "collectibles": "Collectibles",
    "colectibles": "Collectibles",
}


def read_catalog_orders(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def read_web_orders(path: str) -> pd.DataFrame:
    # Web dataset: header uses commas but rows are semicolon-separated and field order differs.
    names = ["ID", "INV", "PCODE", "DATE", "CATALOG", "QTY", "custnum"]
    return pd.read_csv(path, sep=";", quotechar='"', header=None, skiprows=1, names=names)


def read_products(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_catalog_date(value) -> pd.Timestamp:
    # Format observed: M/YY/D HH:MM:SS (e.g., 3/97/7 00:00:00)
    s = str(value).strip()
    m = re.match(r"^(\d{1,2})/(\d{2})/(\d{1,2})", s)
    if not m:
        return pd.NaT

    month = int(m.group(1))
    yy = int(m.group(2))
    day = int(m.group(3))
    year = 2000 + yy if yy <= 30 else 1900 + yy

    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        return pd.NaT


def parse_web_date(value) -> pd.Timestamp:
    # Format observed: DD/MM/YYYY HH:MM:SS
    return pd.to_datetime(value, dayfirst=True, errors="coerce")


def normalize_catalog(value):
    s = str(value).strip().lower()
    if s in CATALOG_MAP:
        return CATALOG_MAP[s]

    close = get_close_matches(s, list(CATALOG_MAP.keys()), n=1, cutoff=0.75)
    if close:
        return CATALOG_MAP[close[0]]
    return pd.NA


def normalize_customer(value):
    if pd.isna(value):
        return pd.NA
    s = str(value).strip()
    return re.sub(r"\s+", " ", s)


def normalize_pcode(value, valid_codes: Set[str]):
    s = str(value).strip().upper().replace(" ", "")
    m = re.match(r"^([A-Z]+)([A-Z0-9]+)$", s)
    if m:
        prefix, suffix = m.groups()
        s = prefix + suffix.replace("O", "0")

    if s in valid_codes:
        return s

    close = get_close_matches(s, list(valid_codes), n=1, cutoff=0.8)
    if close:
        return close[0]
    return pd.NA


def transform_orders(catalog_df: pd.DataFrame, web_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    valid_pcodes = set(products_df["PCODE"].astype(str).str.strip().str.upper())

    cat = catalog_df.copy()
    cat["source_channel"] = "Catalog"
    cat["order_date"] = cat["DATE"].apply(parse_catalog_date)
    cat["CATALOG_RAW"] = cat["CATALOG"]
    cat["CATALOG"] = cat["CATALOG"].apply(normalize_catalog)
    cat["PCODE_RAW"] = cat["PCODE"]
    cat["PCODE"] = cat["PCODE"].apply(lambda x: normalize_pcode(x, valid_pcodes))
    cat["customer_nk"] = cat["custnum"].apply(normalize_customer)

    web = web_df.copy()
    web["source_channel"] = "Web"
    web["order_date"] = web["DATE"].apply(parse_web_date)
    web["CATALOG_RAW"] = web["CATALOG"]
    web["CATALOG"] = web["CATALOG"].apply(normalize_catalog)
    web["PCODE_RAW"] = web["PCODE"]
    web["PCODE"] = web["PCODE"].apply(lambda x: normalize_pcode(x, valid_pcodes))
    web["customer_nk"] = web["custnum"].apply(normalize_customer)

    orders = pd.concat([cat, web], ignore_index=True)
    orders["ID"] = pd.to_numeric(orders["ID"], errors="coerce").astype("Int64")
    orders["INV"] = pd.to_numeric(orders["INV"], errors="coerce")
    orders["QTY"] = pd.to_numeric(orders["QTY"], errors="coerce")

    orders = orders.rename(
        columns={
            "ID": "transaction_id",
            "INV": "invoice_no",
            "QTY": "qty",
            "custnum": "customer_raw",
        }
    )

    # Records with missing core keys cannot be loaded into fact table reliably.
    orders = orders.dropna(subset=["transaction_id", "invoice_no", "order_date", "PCODE", "qty", "customer_nk"])
    orders = orders[orders["qty"] > 0].copy()

    return orders


def build_dimensions_and_fact(orders: pd.DataFrame, products_df: pd.DataFrame):
    products = products_df.copy()
    products["PCODE"] = products["PCODE"].astype(str).str.strip().str.upper()

    dim_product = (
        products[["PCODE", "TYPE", "DESCRIP", "PRICE", "COST", "supplier"]]
        .drop_duplicates()
        .rename(
            columns={
                "PCODE": "product_code",
                "TYPE": "product_type",
                "DESCRIP": "product_description",
                "PRICE": "unit_price",
                "COST": "unit_cost",
                "supplier": "supplier_name",
            }
        )
        .reset_index(drop=True)
    )
    dim_product.insert(0, "product_key", range(1, len(dim_product) + 1))

    dim_customer = (
        orders[["customer_nk", "customer_raw"]]
        .drop_duplicates()
        .rename(columns={"customer_nk": "customer_code", "customer_raw": "customer_name"})
        .reset_index(drop=True)
    )
    dim_customer.insert(0, "customer_key", range(1, len(dim_customer) + 1))

    dim_channel = pd.DataFrame(
        {
            "channel_key": [1, 2],
            "channel_name": ["Catalog", "Web"],
        }
    )

    dim_date = (
        orders[["order_date"]]
        .drop_duplicates()
        .assign(
            full_date=lambda d: pd.to_datetime(d["order_date"]).dt.date,
            year=lambda d: pd.to_datetime(d["order_date"]).dt.year,
            quarter=lambda d: pd.to_datetime(d["order_date"]).dt.quarter,
            month=lambda d: pd.to_datetime(d["order_date"]).dt.month,
            day=lambda d: pd.to_datetime(d["order_date"]).dt.day,
            day_name=lambda d: pd.to_datetime(d["order_date"]).dt.day_name(),
        )
        .drop(columns=["order_date"])
        .sort_values(["full_date"])
        .reset_index(drop=True)
    )
    dim_date.insert(0, "date_key", range(1, len(dim_date) + 1))

    fact = orders.copy()
    fact = fact.merge(
        dim_product[["product_key", "product_code", "unit_price"]],
        left_on="PCODE",
        right_on="product_code",
        how="left",
    )
    fact = fact.merge(dim_customer[["customer_key", "customer_code"]], left_on="customer_nk", right_on="customer_code", how="left")
    fact = fact.merge(dim_channel, left_on="source_channel", right_on="channel_name", how="left")
    fact = fact.merge(dim_date[["date_key", "full_date"]], left_on=fact["order_date"].dt.date, right_on="full_date", how="left")

    fact["total_amount"] = fact["qty"] * fact["unit_price"]
    fact_sales = fact[
        [
            "transaction_id",
            "invoice_no",
            "date_key",
            "product_key",
            "customer_key",
            "channel_key",
            "qty",
            "unit_price",
            "total_amount",
            "CATALOG",
            "CATALOG_RAW",
            "PCODE_RAW",
        ]
    ].rename(columns={"CATALOG": "catalog_clean", "CATALOG_RAW": "catalog_raw", "PCODE_RAW": "pcode_raw"})

    return dim_product, dim_customer, dim_channel, dim_date, fact_sales


def print_quality_report(orders: pd.DataFrame, catalog_df: pd.DataFrame, web_df: pd.DataFrame):
    raw_catalog = pd.concat([catalog_df["CATALOG"], web_df["CATALOG"]], ignore_index=True).astype(str).str.strip()
    catalog_errors = (~raw_catalog.str.lower().isin([c.lower() for c in CANONICAL_CATALOGS])).sum()

    print("=== Data Quality Summary ===")
    print(f"Total rows after transform: {len(orders)}")
    print(f"Unique customers: {orders['customer_nk'].nunique()}")
    print(f"Unique products in orders: {orders['PCODE'].nunique()}")
    print(f"Catalog typo frequency (raw): {int(catalog_errors)} / {len(raw_catalog)}")
    print(f"Date range: {orders['order_date'].min()} -> {orders['order_date'].max()}")


def run_load(
    db_url: str,
    dim_product: pd.DataFrame,
    dim_customer: pd.DataFrame,
    dim_channel: pd.DataFrame,
    dim_date: pd.DataFrame,
    fact_sales: pd.DataFrame,
):
    engine = create_engine(db_url)

    ddl = """
    CREATE TABLE IF NOT EXISTS dim_product (
        product_key INT PRIMARY KEY,
        product_code VARCHAR(20) UNIQUE NOT NULL,
        product_type VARCHAR(100),
        product_description VARCHAR(255),
        unit_price NUMERIC(12,2),
        unit_cost NUMERIC(12,2),
        supplier_name VARCHAR(255)
    );

    CREATE TABLE IF NOT EXISTS dim_customer (
        customer_key INT PRIMARY KEY,
        customer_code VARCHAR(255) UNIQUE NOT NULL,
        customer_name VARCHAR(255)
    );

    CREATE TABLE IF NOT EXISTS dim_channel (
        channel_key INT PRIMARY KEY,
        channel_name VARCHAR(30) UNIQUE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dim_date (
        date_key INT PRIMARY KEY,
        full_date DATE UNIQUE NOT NULL,
        year INT,
        quarter INT,
        month INT,
        day INT,
        day_name VARCHAR(20)
    );

    CREATE TABLE IF NOT EXISTS fact_sales (
        transaction_id BIGINT NOT NULL,
        invoice_no NUMERIC(18,2) NOT NULL,
        date_key INT NOT NULL REFERENCES dim_date(date_key),
        product_key INT NOT NULL REFERENCES dim_product(product_key),
        customer_key INT NOT NULL REFERENCES dim_customer(customer_key),
        channel_key INT NOT NULL REFERENCES dim_channel(channel_key),
        qty NUMERIC(12,2) NOT NULL,
        unit_price NUMERIC(12,2),
        total_amount NUMERIC(14,2),
        catalog_clean VARCHAR(100),
        catalog_raw VARCHAR(100),
        pcode_raw VARCHAR(50)
    );
    """

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fact_sales;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_date;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_product;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_customer;"))
        conn.execute(text("DROP TABLE IF EXISTS dim_channel;"))
        for statement in [s.strip() for s in ddl.split(";") if s.strip()]:
            conn.execute(text(statement))

    dim_product.to_sql("dim_product", engine, if_exists="append", index=False)
    dim_customer.to_sql("dim_customer", engine, if_exists="append", index=False)
    dim_channel.to_sql("dim_channel", engine, if_exists="append", index=False)
    dim_date.to_sql("dim_date", engine, if_exists="append", index=False)
    fact_sales.to_sql("fact_sales", engine, if_exists="append", index=False)
    print("Load completed in PostgreSQL.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline ETL para Practica 6")
    parser.add_argument("--catalog", default="Catalog_Orders.txt", help="Ruta Catalog_Orders")
    parser.add_argument("--web", default="Web_orders.txt", help="Ruta Web_orders")
    parser.add_argument("--products", default="products.txt", help="Ruta products")
    parser.add_argument("--db-url", default="postgresql://postgres:root@localhost:5432/etl_db", help="Conexion PostgreSQL")
    parser.add_argument("--no-load", action="store_true", help="Ejecuta solo extract+transform")
    args = parser.parse_args()

    catalog_df = read_catalog_orders(args.catalog)
    web_df = read_web_orders(args.web)
    products_df = read_products(args.products)

    orders = transform_orders(catalog_df, web_df, products_df)
    dim_product, dim_customer, dim_channel, dim_date, fact_sales = build_dimensions_and_fact(orders, products_df)
    print_quality_report(orders, catalog_df, web_df)

    # Persist local outputs for report/review
    orders.to_csv("orders_clean.csv", index=False)
    dim_product.to_csv("dim_product.csv", index=False)
    dim_customer.to_csv("dim_customer.csv", index=False)
    dim_channel.to_csv("dim_channel.csv", index=False)
    dim_date.to_csv("dim_date.csv", index=False)
    fact_sales.to_csv("fact_sales.csv", index=False)

    print("CSV outputs generated.")
    if not args.no_load:
        run_load(args.db_url, dim_product, dim_customer, dim_channel, dim_date, fact_sales)


if __name__ == "__main__":
    main()
