-- Creates a trigger that decreases quantity of an item

CREATE TRIGGER decrease_quantity
AFTER INSERT ON orders
FOR EACH ROW
UPDATE items 
SET quantity = quantity - NEW.number 
WHERE name = NEW.item_name;