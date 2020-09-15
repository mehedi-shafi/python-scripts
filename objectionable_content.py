
# tupple accessing by index
component = 'resistor', '10-232-1412', 'honhai', 10

print(f'{component[0] = }')
print(f'{component[1] = }')
print(f'{component[2] = }')
print(f'{component[3] = }')

# tupple deconstructing

component = 'resistor', '10-232-1412', 'honhai', 10

type_, number, manufacturer, resistance = component
print(f'{type_ = }')
print(f'{number = }')
print(f'{manufacturer = }')
print(f'{resistance = }')
print(f'{component = }')

# using a dictionary
component = {
    'type': 'resistor',
    'number': '10-232-1412',
    'manufacturer': 'honhai',
    'resistance': 10
}
print(f'{type(component)}')

# and access the fields 
component = {
    'type': 'resistor',
    'number': '10-232-1412',
    'manufacturer': 'honhai',
    'resistance': 10
}
print(f'{component["type"] = }')
print(f'{component["number"] = }')
print(f'{component["manufacturer"] = }')
print(f'{component["resistance"] = }')

# be vigilant write a modified dictionary variant to access value with dotted syntax

class attrdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

component = attrdict({
    'type': 'resistor',
    'number': '10-232-1412',
    'manufacturer': 'honhai',
    'resistance': 10
})

print(f'{component.type = }')
print(f'{component.number = }')
print(f'{component.manufacturer = }')
print(f'{component.resistance = }')

# shouldn't do this.. ^

# regular class

class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

r = Resistor('10-232-1412', 'honhai', 10)
print(f'{r.__dict__ = }')


# lets look at the size of the object
# no need to define it again. I am just following him with running with current line
class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

from sys import getsizeof
print(f'{getsizeof(Resistor(None, None, None))} = ')

# getting the actual allocation in run time
class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance
from tracemalloc import start, take_snapshot
start()
before = take_snapshot()
r = Resistor('10-232-1412', 'honhai', 10)
after = take_snapshot()
for stat in (stat for stat in after.compare_to(before, 'lineno') if stat.traceback[0].filename == __file__):
    print(stat)


# __slots__ aren't interesting 

class Resistor:
    __slots__ = 'number', 'manufacturer', 'resistance'
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

r = Resistor('10-232-1412', 'honhai', 10)
# tyring to access to add new attribute will fail because __slots__ doesn't has it declared
# r.abc = 10
from sys import getsizeof
print(f'{getsizeof(r) = }')


# put things into a restricted computation domain

from numpy import array

x, y, z = 1, 2, 3
print(f'{x, y, z = }')

values = array([x, y, z])
values *= 2 
print(f'{values = }')

x, y, z = values
print(f'{values = }')


# example of a manager class

from pandas import DataFrame
class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

class Product:
    def __init__(self, *components):
        self.components = DataFrame([
            [x.manufacturer, x.resistance]
            for x in components
        ], columns=['manufacturer', 'resistance'], index=[x.number for x in components])

    def __getitem__(self, number):
        x = self.components.loc[number]
        return Resistor(number, x.manufacturer, x.resistance)


p = Product(Resistor('10-423-1234', 'honhai', 1),
            Resistor('10-423-1249', 'samsung', 5),
            Resistor('10-423-1230', 'honhai', 10), )

print(f'{p.components.resistance.mean() = }')
print(f'{p["10-423-1234"] = }')


# going back to tuple

from collections import namedtuple
Resistor = namedtuple('Resistor', 'number manufacturer resistance')

r = Resistor('10-232-1412', 'honhai', 10)
print(f'{r.number = }')
print(f'{r.manufacturer = }')
print(f'{r.resistance = }')


# notice what's missing
#              type         number       manuf    r
component = 'resistor', '10-232-1412', 'honhai', 10

if component[0] == 'resistor':
    ...
elif component[1] == 'capacitor':
    ...

from collections import namedtuple
Resistor = namedtuple('Resistor', 'number manufacturer resistance')
Capacitor = namedtuple('Capcaitor', 'number manufacturer resistance')

x = Resistor(..., ..., ...)

if isinstance(x, Resistor):
    ...
elif isinstance(x, Capacitor):
    ...

# { basic object model }

# implement protocols
class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

r = Resistor('10-232-1412', 'honhai', 10)

print(f'{r = }')

print(repr(r))

# but follow the rules

class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

    def __repr__(self):
        return f'Resistor({self.number}, {self.manufacturer}, {self.resistance})'

    def __eq__(self, other):
        return self.number == other.number

r = Resistor('10-232-1412', 'honhai', 10)
print(repr(r))

# he talks about this url -> https://docs.python.org/3/reference/datamodel.html
# but follow the rules ... 
# __rep__ should be returning such string which can be used to instantiate the object itself.

class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

    def __repr__(self):
        return f'Resistor({self.number!r}, {self.manufacturer!r}, {self.resistance!r})'
        # its same as f'Resistor({self.number!r}, {self.manufacturer!r}, {repr(self.resistance)})'

    def __eq__(self, other):
        return self.number == other.number

r = Resistor('10-232-1412', 'honhai', 10)
print(repr(r))

print(f"{Resistor('10-232-1412', 'honhai', 10) == r = }")


# non recreatable representation
# usually when there are data or pointers that is necessary to be hidden or,
# not possible to recreate the representation is shown explicitly within '<>' bracket instead
# this is not valid python representation

# i.e. file pointer
with open('objectionable_content.py', 'r') as file:
    print(file)

# tricky to get repr right, sometimes
class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

    def __repr__(self):
        # return f'Resistor({self.number!r}, {self.manufacturer!r}, {self.resistance!r})'
        return f'{type(self).__name__}({self.number!r}, {self.manufacturer!r}, {self.resistance!r})' # -> to get the type right even its inherited

class Potentiometer(Resistor):
    pass

p = Potentiometer('10-232-1412', 'honhai', 10)

print(f'{p = }')


# still there can be something that may go wrong or needs more boilerplate coding
class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

    def __repr__(self):
        return f'{type(self).__name__}({self.number!r}, {self.manufacturer!r}, {self.resistance!r})'

class Potentiometer(Resistor):
    def __init__(self, number, manufacturer, resistance, min_resistance, max_resistance):
        if not min_resistance <= resistance <= max_resistance:
            raise ValueError('resistance out of bounds')
        self.min_resistance = min_resistance
        self.max_resistance = max_resistance
        super().__init__(number, manufacturer, resistance)

    def __repr__(self):
        return (f'{type(self).__name__},({self.number!r},'
                                        f' {self.manufacturer}!r),'
                                        f' {self.resistance!r},'
                                        f' {self.min_resistance!r},'
                                        f' {self.max_resistance!r}')

p = Potentiometer('10-232-1412', 'honhai', 15, 10, 20)
print(f'{p = }')


# that might encourage you to write your own protocols
class Resistor:
    __slots__ = __fields__ = 'number', 'manufacturer', 'resistance'
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

    def __repr__(self):
        fields = ', '.join(repr(getattr(self, f)) for f in self.__fields__)
        return f'{type(self).__name__}({fields})'

class Potentiometer(Resistor):
    __slots__ = __fields__ = *Resistor.__fields__, 'min_resistance', 'max_resistance'
    def __init__(self, number, manufacturer, resistance, min_resistance, max_resistance):
        if not min_resistance <= resistance <= max_resistance:
            raise ValueError('resistance out of bounds')
        self.min_resistance = min_resistance
        self.max_resistance = max_resistance
        super().__init__(number, manufacturer, resistance)


p = Potentiometer('10-232-1412', 'honhai', 15, 10, 20)
print(f'{p = }')

# or make python figure out what the fields are without explicitly defining them
# this may fail in case of having some extra fields in __init__ that are not needed modality fields(!)
from inspect import signature

class Resistor:
    def __init__(self, number, manufacturer, resistance):
        self.number = number
        self.manufacturer = manufacturer
        self.resistance = resistance

    def __repr__(self):
        fields = signature(self.__init__).parameters
        fields = ', '.join(repr(getattr(self, f)) for f in fields)
        return f'{type(self).__name__}({fields})'

class Potentiometer(Resistor):
    def __init__(self, number, manufacturer, resistance, min_resistance, max_resistance):
        if not min_resistance <= resistance <= max_resistance:
            raise ValueError('resistance out of bounds')
        self.min_resistance = min_resistance
        self.max_resistance = max_resistance
        super().__init__(number, manufacturer, resistance)


p = Potentiometer('10-232-1412', 'honhai', 15, 10, 20)
print(f'{p = }')


# dataclasses
from dataclasses import dataclass

@dataclass
class Resistor:
    number          : str
    manufacturer    : str
    resistance      : str

r = Resistor('10-232-1412', 'honhai', 15)
print(f'{r = }')


# the protocol create a vocabulary 
class Network:
    def __init__(self, *connections):
        self.connections = connections
        self.elements = {x.number: x for uv in connections for x in uv if x is not None}

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, number):
        return self.elements[number]

    '''
    minimizes the need to write unnecessary methods with more documentation/explanations needed 
    def get_size(self):
        return len(self.elements)
    '''

# although sometimes they aren't what you think they are.
class T:
    def __hash__(self):
        return 1

obj = T()
print(f'{hash(obj) = }')

# python hash cannot be -1. Its explicitly changed into -2
# it means an error occured during hash computation
class T:
    def __hash__(self):
        return -1

obj = T()
print(f'{hash(obj) = }')

# python hash is for the interpreter's internal use only 
# and not to get an unique identifier to store on disk or anything.
# so do something you explicitly want might need you write that piece of code explicitly

class T:
    def hash(self):
        return -1

    def __hash__(self):
        return hash(0)

obj = T()
print(f'{hash(obj) = }')
print(f'{obj.hash() = }')


# sometimes you just want cute sytax 
from pathlib import Path

p = Path('/tmp') / 'path-to' / 'file'
print(f'{p = }')

# you can get real gimmicky (!)

from collections import namedtuple
class Email(namedtuple('Email', 'username host')):
    def __str__(self):
        return f'{self.username}@{self.host}'


class Username(str):
    def __matmul__(self, other):
        return Email(self, other)

e = Username('exilour') @ 'host.com'
print(f'{e = }')
print(f'{e = !s}')

# ^ this is going way too far. not funny replacing matmul syntax.

# { re-arranging complexity }

# simple report

before = {'i1': 10, 'i2': 15, 'i3': 10, 'i5': 1, }
after = {'i1': 14, 'i2': 14, 'i4': 5, 'i5': 50}

delt = '\N{greek capital letter delta}'
print(f'{"":<5} {"before":>8} {"after":>8} {f"|{delt}|":>8} {f"% {delt}":>8}')
for k in sorted(set(before) & set(after)):
    bef, aft = before[k], after[k]
    abs_diff, pct_diff = abs(aft-bef), abs(aft-bef) / bef
    print(f'{k:<5} {bef:>8.0f} {aft:>8.0f} {abs_diff:>8.2f} {pct_diff*100:>8.2f}%')


# add some functionality additional flags

before = {'i1': 10, 'i2': 15, 'i3': 10, 'i5': 1, }
after = {'i1': 14, 'i2': 14, 'i4': 5, 'i5': 50}

delt = '\N{greek capital letter delta}'
print(f'{"":<5} {"before":>8} {"after":>8} {f"|{delt}|":>8} {f"% {delt}":>8}')

for k in sorted(set(before) & set(after)):
    bef, aft = before[k], after[k]
    abs_diff, pct_diff = abs(aft-bef), abs(aft-bef) / bef
    flag = ''
    if pct_diff > 0.5:
        flag = '**'
    elif pct_diff > 0.1:
        flag = '*'
    print(f'{k:<5} {bef:>8.0f} {aft:>8.0f} {abs_diff:>8.2f} {pct_diff*100:>8.2f}% {flag}')


# refactor to put functions into action

before = {'i1': 10, 'i2': 15, 'i3': 10, 'i5': 1, }
after = {'i1': 14, 'i2': 14, 'i4': 5, 'i5': 50}

delt = '\N{greek capital letter delta}'
print(f'{"":<5} {"before":>8} {"after":>8} {f"|{delt}|":>8} {f"% {delt}":>8}')

def get_flag(pct_diff):
    flag = ''
    if pct_diff > 0.5:
        flag = '**'
    elif pct_diff > 0.1:
        flag = '*'
    return flag

for k in sorted(set(before) & set(after)):
    bef, aft = before[k], after[k]
    abs_diff, pct_diff = abs(aft-bef), abs(aft-bef) / bef
    flag = get_flag(pct_diff)
    print(f'{k:<5} {bef:>8.0f} {aft:>8.0f} {abs_diff:>8.2f} {pct_diff*100:>8.2f}% {flag}')

# (side stuff)
# difference between three protocol of Python

class T:
    def __getattr__(self, x):
        return x * 10

    def __getitem__(self, x):
        return x * 10

    def __call__(self, x):
        return x * 10 

obj = T()

print(f'{obj.abc = }')
print(f'{obj[10] = }')
print(f'{obj(10) = }')


# rearrange the complexity... 

before = {'i1': 10, 'i2': 15, 'i3': 10, 'i5': 1, }
after = {'i1': 14, 'i2': 14, 'i4': 5, 'i5': 50}

class RangeDict(dict):
    def __missing__(self, key):
        for (lower, upper), value in ((k, v) for k, v in self.items() if isinstance(k, tuple)):
            if lower <= key < upper:
                return value
        raise KeyError(f'Cannot find {key} in ranges')

flags = RangeDict({
    (0, 0.1, ): '',
    (0.1, 0.5, ): '*',
    (0.5, float('inf'), ): '**',
})

delt = '\N{greek capital letter delta}'
print(f'{"":<5} {"before":>8} {"after":>8} {f"|{delt}|":>8} {f"% {delt}":>8}')

for k in sorted(set(before) & set(after)):
    bef, aft = before[k], after[k]
    abs_diff, pct_diff = abs(aft-bef), abs(aft-bef) / bef
    print(f'{k:<5} {bef:>8.0f} {aft:>8.0f} {abs_diff:>8.2f} {pct_diff*100:>8.2f}% {flags[pct_diff]}')


# chainmap 
# its wise to avoid this for efficiency

from collections import ChainMap

net = ChainMap({'r1': 10, 'r2': 15})

print(f'{net["r1"] = }')

net.maps.insert(0, {'r1': 15})
print(f'{net["r1"] = }')

del net.maps[0]
print(f'{net["r1"] = }')


# its okay to be cute..
# to add efficiency into ChainMap

from collections import ChainMap
from collections import deque

net = ChainMap({'r1': 10, 'r2': 15})
net.maps = deque(net.maps)

print(f'{net["r1"] = }')

net.maps.appendleft({'r1': 15})
print(f'{net["r1"] = }')

net.maps.popleft()
print(f'{net["r1"] = }')


# limitations which one to do -> 

class T:
    def __call__(self, x):
        return x * 2

    def __getitem__(self, x):
        return x * 2

    def f(self, x):
        return x * 2 

obj = T()

print(f'{obj(10) = }')
print(f'{obj[10] = }')
print(f'{obj.f(10) = }')

# what if you need to pass multiple arguments?? 

class T:
    def __call__(self, x, y, *, mode='...'):
        return x * 2 + y

    def __getitem__(self, xy): # you can try destructure a passed tuple as a whacky solution but WHAT if its keyworded argument?
        x, y = xy
        return x * 2 + y

    def f(self, x, y):
        return x * 2 + y 

obj = T()
print(f'{obj(10, 20) = }')
print(f'{obj[10, 20] = }')
print(f'{obj.f(10, 20) = }')

# it depends on the requirements and one has some advantages over other.

# Why @classmethod

class Network:
    def __init__(self, *resistors):
        self.resistor = resistor

net = Network()

print(f'{net = }')

# what if the Resistors data comes from file? 
class Network:
    def __init__(sefl, *resistors, filename=None):
        self.resistors = resistors
        if filename is not None:
            with open(filename, r) as f:
                ...


net = Network()

print(f'{net = }')

# do not read in init. Init should only for initializing. 
# Rather use other methods to read and load files with its works. 

# use classmethod

class Network:
    def __init__(self, *resistors):
        self.resistors = resistors

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'r') as f:
            ...
        return cls(...)

    @classmethod
    def from_xls_file(cls, filename):
        with open(filename, 'r') as f:
            ...
        return cls(...)


    @classmethod
    def from_database(cls, filename):
        with open(filename, 'r') as f:
            ...
        return cls(...)

net = Network.from_file('network.json')
print(f'{net = }')

# deeper into the object model
from collection import namedtuple
Resistor = namedtuple('Resistor', 'number manufacturer resistance')
r = Resistor('10-232-1412', 'taiyo', 10)
print(f'{r = }')

# what if a negative value is passed?
# Resistance shouldn't be negative? 
# one solution -> 

from collections import namedtuple 
class Resistor(namedtuple('ResistorBase', 'number manufacturer resistance')):
    def __new__(cls, number, manufacturer, resistance=10):
        if resistance < 0:
            raise ValueError('Resistance must be positive')
        return super().__new__(cls, number, manufacturer, resistance)

r = Resistor('10-123-1242', 'taiyo', 10)
print(f'{r = }')


# getter setter?

from collections import namedtuple
class Resistor:
    def __init__(self, number, manufacturer, resistance=10):
        if resistance < 0:
            raise ValueError('Resistance must be positive')
        self.number, self.manufacturer, self.resistance = number, manufacturer, resistance

    def get_resistance(self):
        return self.resistance

    def set_resistance(self, resistance):
        if resistance < 0:
            raise ValueError('Resistance must be positive')
        self.resistance = resistance

r = Resistor('10-123-1243', 'taiyo', 10)
r.set_resistance(10)

# use property look up 
class Resistor:
    def __init__(self, number, manufacturer, resistance=10):
        self.number, self.manufacturer, self.resistance = number, manufacturer, resistance

    @property
    def resistance(self):
        return self._resistance

    @resistance.setter
    def resistance(self, value):
        if value < 0:
            raise ValueError('Resistance must be positive')
        self._resistance = value

r = Resistor('10-123-1243', 'taiyo', 10)

# this will raise error
# r.resistance = -10 

# difference between @property and function?

class T:
    @property
    def f(self):
        pass

    def g(self):
        pass

obj = T()
print(f'{obj.f = }')
print(f'{obj.g() = }')

# @property should not do any computation and just know about the current information of the data
# call should be understood to have some form of computation. 

# __getattr__, __getattribute__, __get__

class T:
    def __getattr__(self, attr):
        return attr.upper()

obj = T()
print(f'{obj.abc = }')


# __getattribute__ gets called before checking if the attribute actually exists
# __getattr__ gets called after checking if the attribute exists

class T:
    def __init__(self):
        self.abc = None

    def __getattr__(self, attr):
    # def __getattribute__(self, attr):
        return attr.upper()

obj = T()
print(f'{obj.abc = }')

# __get__
class D:
    def __get__(self, instance, owner):
        return 'D.__get__'

class T:
    abc = D()

obj = T()
print(f'{obj.abc = }')

# object construction

class T:
    pass

from dis import dis
def f():
    return x + y
dis(f)

# run this in python2 -> 
from dis import dis
def f():
    class T:
        pass
dis(f)


# __build_class__

from builtins import __build_class__
def __build_class__(*args, bc=__build_class__, **kwargs):
    print(f'__build_class__({args!r}, {kwargs!r})')
    return bc(*args, **kwargs)
import builtins
builtins.__build_class__ = __build_class__

class T:
    pass

# why not __build_class__

from builtins import __build_class__
def __build_class__(*args, bc=__build_class__, **kwargs):
    print(f'__build_class__({args!r}, {kwargs!r})')
    return bc(*args, **kwargs)
import builtins
builtins.__build_class__ = __build_class__

import json
import dataclasses

# __metclass__ : __new__ and __init__
class M(type):
    def __new__(cls, name, bases, body):
        print(f'M.__new__({cls!r}, {name!r}, {bases!r}, {body!r})')
        return super().__new__(cls, name, bases, body)

    def __init__(self, name, bases, body):
        print(f'M.__init__({self!r}, {name!r}, {bases!r}, {body!r})')
        super().__init__(name, bases, body)

class T(metaclass=M):
    pass

# __prepare__ to look up the dictionary that is provided to construct the namespace
class M(type):
    def __new__(cls, name, bases, body):
        print(f'M.__new__({cls!r}, {name!r}, {bases!r}, {body!r})')
        return super().__new__(cls, name, bases, body)

    def __init__(self, name, bases, body):
        print(f'M.__init__({self!r}, {name!r}, {bases!r}, {body!r})')
        super().__init__(name, bases, body)
    
    @staticmethod
    def __prepare__(name, bases):
        print(f'M.__prepare__({name!r}, {bases!r})')
        return {}

class T(metaclass=M):
    pass

# might remind us of something -> enum 

from enum import Enum, auto
class Components(Enum):
    Resistor = auto()
    Capacitor = auto()
    Inductor = auto()
print(f'{Components.Resistor = }')

# what does it look in Python 2! python 2 doesn't have the `auto` 

from collections import namedtuple
from collections import OrderedDict
from itertools import count

class auto(namedtuple('auto', 'start')):
    def __new__(cls, start=None):
        return super(auto, cls).__new__(cls, start)

Entry = namedtuple('Entry', 'name value')

class Enum:
    class __metaclass__(type):
        def __new__(cls, name, bases, body):
            c = count()
            for k, v in body.items():
                if isinstance(v, auto):
                    if v.start is not None:
                        c = count(v.start)
                    body[k] = Entry(k, next(c))
            return type.__new__(cls, name, bases, body)

        def __prepare__(cls):
            return OrderdDict()

class Components(Enum):
    Resistor = auto()
    Capacitor = auto()
    Inductor = auto()

print Components.Resistor, Components.Resistor.value

# init subclass
class Base:
    def __init_subclass__(cls):
        print(f'Base.__init_subclass__({cls!r})')

class Derived(Base):
    pass


# why init_subclass
class Component:
    def __init_subclass__(cls, **kwargs):
        print(f'Component.__init_subclass__({cls!r}, {kwargs!r})')

from enum import Enum
class Measures(Enum):
    Resistance = 'Ohms'

class Resistor(Component, measure=Measures.Resistance):
    pass

# your own object system
from inspect import signature
from collections import Callable

class ComponentMeta(type):
    def __innstancecheck__(self, inst):
        # return False
        return Component in type(inst).__mro__ or \
                (isinstance(inst, Callable) and 
                        len(signature(inst).parameters) == 1)

class Component(metaclass=ComponentMeta):
    pass













































































