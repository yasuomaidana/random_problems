from numpy import fft, inf, sqrt, pi
from numpy import abs as np_abs
from numpy.ma.core import argmax


def generate_product_sales(days: int, periodicity: int, max_sales: int) -> list[int]:
    import random
    from numpy import cos, pi
    sales = []
    average_sales = max_sales // 2
    for i in range(days):
        if i % periodicity == 0:
            sales.append(random.randint(0, max_sales))
        else:
            day_factor = cos(i * (1 / periodicity) * 2 * pi)
            random_sales = random.randint(-average_sales // 2, average_sales // 2)
            day_sales = max(0, average_sales + average_sales*day_factor + random_sales)
            if day_sales > max_sales:
                if random.random() < 0.95:
                    day_sales = max_sales
            sales.append(int(day_sales))
    return sales


def generate_data(days: int, max_sales: list[int], products: list[str], periodicity: list[int]):
    return [generate_product_sales(days, periodicity[i], max_sales[i])
            for i in range(len(products))]

def print_list_as_cpp_vector(py_list):
    cpp_vector = "{" + ", ".join(map(str, py_list)) + "};"
    print(cpp_vector)

def formatter(days, m_sales, products, periodicity, test_i):
    print(f"Unit Test {test_i} Input:")
    sales_data = generate_data(days, m_sales, products, periodicity)
    print(f"sales_data = {sales_data}")
    print(f"products ={products}")
    print(f"Unit Test {test_i} Output:")
    print("result = ")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.clf()
    i = 1
    days_ = 365*2
    max_sales_ = [500]
    products_ = ["potato"]
    periodicity_ = [26]
    data = generate_data(days_, max_sales_, products_, periodicity_)

    formatter(days_, max_sales_, products_, periodicity_, i)
    print()


    i +=1
    days_ = 365*2
    max_sales_ = [500]
    products_ = ["potato"]
    periodicity_ = [400]
    data = generate_data(days_, max_sales_, products_, periodicity_)
    formatter(days_, max_sales_, products_, periodicity_, i)

    i +=1
    days_ = 365*2
    max_sales_ = [16,30,50,25]
    products_ = ["potato","tomato","cucumber","carrot"]
    periodicity_ = [8, 12, 24, 48]
    data = generate_data(days_, max_sales_, products_, periodicity_)
    formatter(days_, max_sales_, products_, periodicity_, i)

    i +=1
    days_ = 365*2
    max_sales_ = [16,30,50,25]
    products_ = ["potato","tomato","cucumber","carrot"]
    periodicity_ = [350, 360, 37, 400]
    data = generate_data(days_, max_sales_, products_, periodicity_)
    formatter(days_, max_sales_, products_, periodicity_, i)







