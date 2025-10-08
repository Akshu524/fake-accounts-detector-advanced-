class mobile:
    company='moto'
    model='50 fushion'
    prize='20,000'
M1=mobile()
print(M1.model)
print(M1.company)
print(M1.prize)


class mobile:
    def __init__(self,model,prize):
        self.model=model
        self.prize=prize

M1=mobile('moto','20,000')
print(M1.model)
print(M1.prize)


class student:
    college_name='abc'
    def __init__(self,name,marks):
        self.name=name
        self.marks=marks
      
s1=student('soni','99')
print(s1.name,s1.marks)



class student:
    college_name='abc'
    def __init__(self,name,marks):
        self.name=name
        self.marks=marks
    def welcome(self):
        print('welcome student',self.name)
    def get_marks(self):
        return self.marks
    s1=student('karan',97)
    print(s1.name)
    print(s1.marks)
        





# abstract....

from abc import ABC, abstractmethod

# Abstract class
class Shape(ABC):
    # Abstract method
    @abstractmethod
    def area(self):
        pass

    # Abstract method
    @abstractmethod
    def perimeter(self):
        pass

    # Concrete method
    def description(self):
        return "This is a shape."

# Subclass inheriting from the abstract class
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    # Implementing the abstract methods
    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

# Subclass inheriting from the abstract class
class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius

    # Implementing the abstract methods
    def area(self):
        return 3.14 * self.radius * self.radius

    def perimeter(self):
        return 2 * 3.14 * self.radius

# Using the subclasses
rectangle = Rectangle(10, 5)
print("Rectangle Area:", rectangle.area())
print("Rectangle Perimeter:", rectangle.perimeter())
print(rectangle.description())

circle = Circle(7)
print("Circle Area:", circle.area())
print("Circle Perimeter:", circle.perimeter())
print(circle.description())
