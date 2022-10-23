# Write your code here
class Shirt:
    """ A Super Shirt """
    def __init__(self, shirt_color, shirt_size, shirt_style, shirt_price):
        self.color = shirt_color
        self.size = shirt_size
        self.style = shirt_style
        self.price = shirt_price

    def change_price(self, new_price):
        self.price = new_price

    def discount(self, discount):
        return self.price * (1 - discount)

shirt_one = Shirt("red", "S", "long-sleeve", 25)

print("Items:\n")
print("  Shirt with price: {}".format(shirt_one.price))
shirt_one.change_price(10)
print("  new price: {}".format(shirt_one.price))
print("  discounted price: {}".format(shirt_one.discount(0.12)))

shirt_two = Shirt("orange", "L", "short-sleeve", 10)
total = sum([shirt_one.price, shirt_two.price])
print("\nTotal: {}".format(total))
total_discount = shirt_one.discount(0.14) + shirt_two.discount(0.06);
print("Total (discounted): {}".format(total_discount))

from tests import run_tests
run_tests(shirt_one, shirt_two, total, total_discount)
