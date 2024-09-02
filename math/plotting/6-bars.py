#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4,3))
    plt.figure(figsize=(6.4, 4.8))

    plt.ylabel('Quantity of Fruit')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))

    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_types = ['apples', 'bananas', 'oranges', 'peaches']

    people = ['Farrah', 'Fred', 'Felicia']

    bottom = np.zeros(3)
    for i, fruit_type in enumerate(fruit_types):
        plt.bar(people, fruit[i], bottom=bottom, color=colors[i], width=0.5, label=fruit_type)
        bottom += fruit[i]

    plt.title('Number of Fruit per Person')
    plt.legend()
    plt.show()
