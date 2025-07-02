-- Create SafeDiv function that safely divides two integers

CREATE OR REPLACE FUNCTION SafeDiv(a INT, b INT)
RETURNS INT
DETERMINISTIC
READS SQL DATA
BEGIN
    IF b = 0 THEN
        RETURN 0;
    ELSE
        RETURN a / b;
    END IF;
END;
