import pandas as pd
import numpy as np
from student import *

class Checker:
    def __init__(self, exam):
        # Assuming model is an instance of a class with methods Q1, Q2, Q3, and attributes like y_test
        self.exam = exam

    def Q1(self):
        y = self.exam.Q1()
        print(y)

    def Q2(self):
        self.exam.Q1()
        y1 = self.exam.Q2()
        print(f"{y1:.2f}")

    def Q3(self):
        self.exam.Q1()
        self.exam.Q2()
        y = self.exam.Q3()
        print(y)

def main():
    data = pd.read_csv('./data/titanic.csv')
    exam_titanic = exam(data)
    checker = Checker(exam_titanic)
    eval(f"checker.{input().strip()}()")


if __name__ == "__main__":
    main()