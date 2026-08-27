-- Sample product catalog for local testing.
-- Copied or replaced by /Documents/ChatBot/data.sql on the target machine.
CREATE TABLE products (
  id INT,
  sku VARCHAR(20),
  name VARCHAR(120),
  description TEXT,
  price DECIMAL(10,2),
  category VARCHAR(60),
  stock INT
);

INSERT INTO products (id, sku, name, description, price, category, stock) VALUES
(1, 'A001', 'Wireless Mouse', 'Comfortable ergonomic wireless mouse with USB receiver and long battery life.', 24.99, 'Electronics', 120),
(2, 'A002', 'Mechanical Keyboard', 'RGB backlit mechanical keyboard with hot-swappable switches and aluminum frame.', 89.50, 'Electronics', 45),
(3, 'B101', 'Noise Cancelling Headphones', 'Over-ear headphones with active noise cancellation and 30-hour battery.', 199.00, 'Audio', 30),
(4, 'B102', 'Portable Bluetooth Speaker', 'Compact waterproof Bluetooth speaker with deep bass and 12-hour playtime.', 49.99, 'Audio', 200),
(5, 'C201', 'Stainless Steel Water Bottle', 'Insulated 750ml stainless steel bottle that keeps drinks cold for 24 hours.', 19.99, 'Kitchen', 300),
(6, 'C202', 'Coffee Maker', 'Programmable 12-cup drip coffee maker with brew-strength selector.', 55.25, 'Kitchen', 60),
(7, 'D301', 'Yoga Mat', 'Non-slip eco-friendly yoga mat with carrying strap, 6mm thick.', 29.00, 'Fitness', 150),
(8, 'D302', 'Adjustable Dumbbells', 'Set of adjustable dumbbells from 5lb to 52.5lb with locking mechanism.', 149.99, 'Fitness', 25);
