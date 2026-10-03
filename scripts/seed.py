import random
from datetime import date, timedelta
from pathlib import Path

from app.config import settings
from app.db.connection import get_write_connection


CUSTOMERS = [
    ("Aarav Sharma", "aarav.sharma@example.com"),
    ("Diya Patel", "diya.patel@example.com"),
    ("Arjun Mehta", "arjun.mehta@example.com"),
    ("Ananya Iyer", "ananya.iyer@example.com"),
    ("Kabir Singh", "kabir.singh@example.com"),
    ("Meera Nair", "meera.nair@example.com"),
    ("Rohan Gupta", "rohan.gupta@example.com"),
    ("Ishita Bose", "ishita.bose@example.com"),
    ("Vikram Rao", "vikram.rao@example.com"),
    ("Sara Khan", "sara.khan@example.com"),
    ("Noah Williams", "noah.williams@example.com"),
    ("Emma Johnson", "emma.johnson@example.com"),
    ("Liam Brown", "liam.brown@example.com"),
    ("Olivia Martin", "olivia.martin@example.com"),
    ("Lucas Garcia", "lucas.garcia@example.com"),
    ("Sofia Rossi", "sofia.rossi@example.com"),
    ("Ethan Wilson", "ethan.wilson@example.com"),
    ("Mia Anderson", "mia.anderson@example.com"),
    ("Kenji Tanaka", "kenji.tanaka@example.com"),
    ("Yuki Sato", "yuki.sato@example.com"),
    ("Amara Okafor", "amara.okafor@example.com"),
    ("Mateo Silva", "mateo.silva@example.com"),
    ("Leila Haddad", "leila.haddad@example.com"),
    ("Hannah Müller", "hannah.muller@example.com"),
    ("Chen Wei", "chen.wei@example.com"),
]


PRODUCTS = [
    ("Nova X1 Smartphone", "electronics", 34999.00, 24, 0),
    ("Pulse Wireless Earbuds", "electronics", 3999.00, 12, 0),
    ("Orbit 14 Laptop", "electronics", 62999.00, 24, 0),
    ("Beam Bluetooth Speaker", "electronics", 5499.00, 12, 0),
    ("ViewPro 27 Monitor", "electronics", 18999.00, 24, 0),
    ("Swift Mechanical Keyboard", "electronics", 6499.00, 12, 0),
    ("Glide Wireless Mouse", "electronics", 2499.00, 12, 0),
    ("Snap Action Camera", "electronics", 21999.00, 24, 1),
    ("Classic Cotton Shirt", "clothing", 1499.00, 0, 0),
    ("Everyday Denim Jeans", "clothing", 2799.00, 0, 0),
    ("Linen Summer Dress", "clothing", 3299.00, 0, 0),
    ("Trail Running Jacket", "clothing", 4499.00, 0, 0),
    ("Comfort Crew T-Shirt", "clothing", 899.00, 0, 0),
    ("Wool Blend Sweater", "clothing", 2999.00, 0, 0),
    ("Active Joggers", "clothing", 1999.00, 0, 0),
    ("Clearance Floral Skirt", "clothing", 799.00, 0, 1),
    ("Ceramic Dinner Set", "home", 3699.00, 0, 0),
    ("Cotton Bedsheet Set", "home", 2299.00, 0, 0),
    ("Bamboo Table Lamp", "home", 1899.00, 0, 0),
    ("Stainless Steel Cookware", "home", 5999.00, 0, 0),
    ("Memory Foam Pillow", "home", 1299.00, 0, 0),
    ("Woven Storage Basket", "home", 999.00, 0, 0),
    ("Soft Bath Towel Set", "home", 1599.00, 0, 0),
    ("Clearance Wall Clock", "home", 699.00, 0, 1),
    ("Hydrating Face Cream", "beauty", 1199.00, 0, 0),
    ("Vitamin C Serum", "beauty", 999.00, 0, 0),
    ("Matte Lip Colour", "beauty", 649.00, 0, 0),
    ("Gentle Daily Cleanser", "beauty", 749.00, 0, 0),
    ("Mineral Sunscreen SPF 50", "beauty", 899.00, 0, 0),
    ("Repair Hair Mask", "beauty", 799.00, 0, 0),
    ("Citrus Eau de Parfum", "beauty", 2499.00, 0, 0),
    ("Clearance Nail Kit", "beauty", 449.00, 0, 1),
    ("Canvas Travel Backpack", "accessories", 2999.00, 0, 0),
    ("Leather Card Holder", "accessories", 1299.00, 0, 0),
    ("Polarized Sunglasses", "accessories", 2199.00, 0, 0),
    ("Stainless Steel Watch", "accessories", 4999.00, 0, 0),
    ("Silk Printed Scarf", "accessories", 1599.00, 0, 0),
    ("Minimalist Tote Bag", "accessories", 1899.00, 0, 0),
    ("Braided Leather Belt", "accessories", 1399.00, 0, 0),
    ("Compact Travel Umbrella", "accessories", 1099.00, 0, 0),
]


def delivered_dates(today: date, age_range: tuple[int, int]) -> tuple[str, str, str]:
    delivered_date = today - timedelta(days=random.randint(*age_range))
    shipped_date = delivered_date - timedelta(days=random.randint(1, 5))
    order_date = shipped_date - timedelta(days=random.randint(0, 2))
    return order_date.isoformat(), shipped_date.isoformat(), delivered_date.isoformat()


def dates_for_status(
    status: str, today: date, delivered_age: tuple[int, int] | None
) -> tuple[str, str | None, str | None]:
    if status in {"delivered", "returned"}:
        age_range = delivered_age or (5, 90)
        return delivered_dates(today, age_range)

    if status == "shipped":
        shipped_date = today - timedelta(days=random.randint(0, 5))
        order_date = shipped_date - timedelta(days=random.randint(0, 2))
        return order_date.isoformat(), shipped_date.isoformat(), None

    max_age = 7 if status == "placed" else 120
    order_date = today - timedelta(days=random.randint(0, max_age))
    return order_date.isoformat(), None, None


def build_orders(today: date) -> tuple[list[tuple], list[tuple]]:
    customer_order_counts = [
        27, 25, 23, 21, 19, 17, 15, 14, 13, 12, 11, 10, 9,
        9, 8, 8, 7, 7, 6, 6, 5, 4, 2, 1, 1,
    ]
    customer_ids = [
        customer_id
        for customer_id, count in enumerate(customer_order_counts, start=1)
        for _ in range(count)
    ]
    random.shuffle(customer_ids)

    order_specs = (
        [("delivered", (1, 5))] * 35
        + [("delivered", (25, 35))] * 35
        + [("delivered", (60, 105))] * 35
        + [("delivered", (6, 24))] * 20
        + [("delivered", (36, 59))] * 20
        + [("placed", None)] * 55
        + [("shipped", None)] * 35
        + [("cancelled", None)] * 25
        + [("returned", (5, 90))] * 20
    )
    random.shuffle(order_specs)

    product_prices = {
        product_id: product[2]
        for product_id, product in enumerate(PRODUCTS, start=1)
    }
    final_sale_ids = [8, 16, 24, 32]
    orders = []
    order_items = []

    for order_id, (customer_id, order_spec) in enumerate(
        zip(customer_ids, order_specs), start=1
    ):
        status, delivered_age = order_spec
        order_date, shipped_date, delivered_date = dates_for_status(
            status, today, delivered_age
        )
        item_count = random.randint(1, 4)
        if order_id <= len(final_sale_ids):
            final_sale_id = final_sale_ids[order_id - 1]
            other_ids = [
                product_id
                for product_id in range(1, len(PRODUCTS) + 1)
                if product_id != final_sale_id
            ]
            product_ids = [final_sale_id] + random.sample(other_ids, item_count - 1)
        else:
            product_ids = random.sample(range(1, len(PRODUCTS) + 1), item_count)

        total = 0.0
        for product_id in product_ids:
            quantity = random.randint(1, 3)
            unit_price = product_prices[product_id]
            total += quantity * unit_price
            order_items.append((order_id, product_id, quantity, unit_price))

        orders.append(
            (
                customer_id,
                status,
                order_date,
                shipped_date,
                delivered_date,
                round(total, 2),
                random.choice(("standard", "express")),
            )
        )

    return orders, order_items


def main() -> None:
    random.seed(42)
    today = date.today()
    Path(settings.DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    schema_path = Path(__file__).parents[1] / "app" / "db" / "schema.sql"
    schema = schema_path.read_text(encoding="utf-8")

    customer_rows = [
        (name, email, (today - timedelta(days=random.randint(180, 1200))).isoformat())
        for name, email in CUSTOMERS
    ]
    orders, order_items = build_orders(today)

    connection = get_write_connection()
    try:
        connection.executescript(
            """
            DROP TABLE IF EXISTS order_items;
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS customers;
            """
            + schema
        )
        with connection:
            connection.executemany(
                "INSERT INTO customers (name, email, created_at) VALUES (?, ?, ?)",
                customer_rows,
            )
            connection.executemany(
                """
                INSERT INTO products
                    (name, category, price, warranty_months, is_final_sale)
                VALUES (?, ?, ?, ?, ?)
                """,
                PRODUCTS,
            )
            connection.executemany(
                """
                INSERT INTO orders
                    (customer_id, status, order_date, shipped_date, delivered_date,
                     total, shipping_method)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                orders,
            )
            connection.executemany(
                """
                INSERT INTO order_items
                    (order_id, product_id, quantity, unit_price)
                VALUES (?, ?, ?, ?)
                """,
                order_items,
            )

        for table in ("customers", "products", "orders", "order_items"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table}: {count}")

        print("orders by status:")
        status_counts = connection.execute(
            "SELECT status, COUNT(*) FROM orders GROUP BY status ORDER BY status"
        ).fetchall()
        for status, count in status_counts:
            print(f"  {status}: {count}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
