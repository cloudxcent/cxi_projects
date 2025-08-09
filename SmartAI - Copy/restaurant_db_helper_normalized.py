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
        Place a new order and insert each item.
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

    def add_items_to_order(self, order_id, items, additional_amount):
        """
        Add items to an existing order and update the total amount.
        """
        try:
            self.conn.start_transaction()
            # Update the total_amount in the orders table
            self.cursor.execute(
                """UPDATE orders SET total_amount = total_amount + %s WHERE order_id = %s""",
                (additional_amount, order_id)
            )
            if self.cursor.rowcount == 0:
                raise Exception("Order ID not found")

            # Insert new items
            for item in items:
                self.cursor.execute(
                    """INSERT INTO order_items (order_id, item_name, quantity, price)
                       VALUES (%s, %s, %s, %s)""",
                    (order_id, item["name"], item["qty"], item["price"])
                )
            
            # Update the bill_amount in the billings table
            self.cursor.execute(
                """UPDATE billings SET bill_amount = bill_amount + %s WHERE order_id = %s""",
                (additional_amount, order_id)
            )
            if self.cursor.rowcount == 0:
                raise Exception("Billing record not found for order ID")

            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print("Error adding items to order:", e)
            return False

    def add_billing(self, order_id, bill_amount, paid=False):
        """
        Create or update a billing record.
        """
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Check if a billing record already exists
            self.cursor.execute("SELECT bill_id FROM billings WHERE order_id=%s", (order_id,))
            existing_bill = self.cursor.fetchone()
            
            if existing_bill:
                # Update existing bill
                self.cursor.execute(
                    """UPDATE billings SET bill_amount=%s, bill_time=%s, paid=%s
                       WHERE order_id=%s""",
                    (bill_amount, now, int(paid), order_id)
                )
            else:
                # Create new bill
                self.cursor.execute(
                    """INSERT INTO billings (order_id, bill_amount, paid, bill_time)
                       VALUES (%s, %s, %s, %s)""",
                    (order_id, bill_amount, int(paid), now)
                )
            self.conn.commit()
            return self.cursor.lastrowid if not existing_bill else existing_bill["bill_id"]
        except Exception as e:
            print("Error adding/updating billing:", e)
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

    def get_order_total(self, order_id):
        self.cursor.execute("SELECT total_amount FROM orders WHERE order_id=%s", (order_id,))
        row = self.cursor.fetchone()
        return row["total_amount"] if row else None

    def close(self):
        self.cursor.close()
        self.conn.close()
