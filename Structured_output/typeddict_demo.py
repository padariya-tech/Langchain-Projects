from typing import TypedDict

class Person(TypedDict):
    name:str
    age:int

new_person : Person = {'name':'nij','age':21}

new_person_2 : Person = {'name':'nij','age':'24'}

print(new_person)

print(new_person_2)