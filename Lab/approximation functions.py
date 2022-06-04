import math

# interval for approximation
a, b = -6, 0
# index last node point
n = 3
node_point = [0.5 * (b + a) - (b - a)* 0.5 * math.cos(math.pi * (2 * i + 1) / (2 * n + 2)) for i in range(n + 1)]

print(list(map(lambda x: round(x, 2), node_point)))
