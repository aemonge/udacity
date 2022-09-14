# from tests import check_results, run_tests

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

class Pants:
    """ A Super Pants """
    def __init__(self, color, waist_size, length, price):
        self.color = color
        self.waist_size = waist_size
        self.length = length
        self.price = price

    def change_price(self, new_price):
        self.price = new_price

    def discount(self, discount):
        return self.price * (1 - discount)

def check_results():
    pants = Pants('red', 35, 36, 15.12)
    assert pants.color == 'red'
    assert pants.waist_size == 35
    assert pants.length == 36
    assert pants.price == 15.12

    pants.change_price(10) == 10
    assert pants.price == 10

    assert pants.discount(.1) == 9

    print('You made it to the end of the check. Nice job!')

check_results()
