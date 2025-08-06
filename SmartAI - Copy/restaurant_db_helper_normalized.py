import mysql.connector
from datetime import datetime

class RestaurantDB:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host, user=user, password=password,
            database=database, autocommit=False
        )
        self.cursor = self.conn.cursor(dictionary=True)

    def place_order(self, table_no, items, total_amount):
        """
        Place an order and insert each item.
        """
        try:
            self.conn.start_transaction()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                """INSERT INTO orders (table_no, total_amount, order_time, status)
                   VALUES (%s, %s, %s, %s)""",
                (table_no, total_amount, now, 'processing')
            )
            order_id = self.cursor.lastrowid

            for item in items:
                self.cursor.execute(
                    """INSERT INTO order_items (order_id, item_name, quantity, price)
                       VALUES (%s, %s, %s, %s)""",
                    (order_id, item["name"], item["qty"], item["price"])
                )
            self.conn.commit()
            return order_id
        except Exception as e:
            self.conn.rollback()
            print("Error placing order:", e)
            return None

    def add_billing(self, order_id, bill_amount, paid=False):
        """
        Create billing record.
        """
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cursor.execute(
                """INSERT INTO billings (order_id, bill_amount, paid, bill_time)
                   VALUES (%s, %s, %s, %s)""",
                (order_id, bill_amount, int(paid), now)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except Exception as e:
            print("Error adding billing:", e)
            return None

    def get_order_status(self, order_id):
        self.cursor.execute("SELECT status FROM orders WHERE order_id=%s", (order_id,))
        row = self.cursor.fetchone()
        return row["status"] if row else None

    def update_order_status(self, order_id, new_status):
        valid = {'processing', 'ready', 'delivered', 'cancelled'}
        if new_status not in valid:
            raise ValueError(f"Invalid status '{new_status}'")
        self.cursor.execute("UPDATE orders SET status=%s WHERE order_id=%s", (new_status, order_id))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def get_billing_info(self, order_id):
        self.cursor.execute("SELECT * FROM billings WHERE order_id=%s", (order_id,))
        return self.cursor.fetchone()

    def mark_bill_paid(self, bill_id):
        self.cursor.execute("UPDATE billings SET paid=1 WHERE bill_id=%s", (bill_id,))
        self.conn.commit()
        return self.cursor.rowcount > 0

    def get_order_items(self, order_id):
        self.cursor.execute("SELECT * FROM order_items WHERE order_id=%s", (order_id,))
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()
        self.conn.close()
