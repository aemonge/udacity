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
        self.size = waist_size
        self.length = length
        self.price = price

    def change_price(self, new_price):
        self.price = new_price

    def discount(self, discount):
        return self.price * (1 - discount)

class SalesPerson:

    def __init__(self, first_name, last_name, employee_id, salary):
        self.first_name = first_name
        self.last_name = last_name
        self.employee_id = employee_id
        self.salary = salary
        self.pants_sold = []
        self.total_sales = 0

    def sell_pants(self, pant):
        self.pants_sold.append(pant)

    def calculate_sales(self):
        self.total_sales = sum([pant.price for pant in self.pants_sold])
        return self.total_sales;

    def display_sales(self):
        for pants in self.pants_sold:
            print("  color: {}, waist_size: {}, length: {}, price: {}"\
                  .format(pants.color, pants.size, pants.length, pants.price)
            )

    def calculate_commission(self, commision):
        if self.total_sales == 0 and len(self.pants_sold) == 0:
            self.calculate_sales()

        return self.total_sales * commision

def check_results():
    pants_one = Pants('red', 35, 36, 15.12)
    pants_two = Pants('blue', 40, 38, 24.12)
    pants_three = Pants('tan', 28, 30, 8.12)

    salesperson = SalesPerson('Amy', 'Gonzalez', 2581923, 40000)

    assert salesperson.first_name == 'Amy'
    assert salesperson.last_name == 'Gonzalez'
    assert salesperson.employee_id == 2581923
    assert salesperson.salary == 40000
    assert salesperson.pants_sold == []
    assert salesperson.total_sales == 0

    salesperson.sell_pants(pants_one)
    salesperson.pants_sold[0] == pants_one.color

    salesperson.sell_pants(pants_two)
    salesperson.sell_pants(pants_three)

    assert len(salesperson.pants_sold) == 3
    assert round(salesperson.calculate_sales(),2) == 47.36
    assert round(salesperson.calculate_commission(.1),2) == 4.74

    print('Great job, you made it to the end of the code checks!')

check_results()

pants_one = Pants('red', 35, 36, 15.12)
pants_two = Pants('blue', 40, 38, 24.12)
pants_three = Pants('tan', 28, 30, 8.12)

salesperson = SalesPerson('Amy', 'Gonzalez', 2581923, 40000)

salesperson.sell_pants(pants_one)
salesperson.sell_pants(pants_two)
salesperson.sell_pants(pants_three)

salesperson.display_sales()

